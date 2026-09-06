"""CPU/static closure for the P8 coupled scale component; no CUDA imports."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
MODULE_PATH = PATCH / "p8_coupled_scales.py"
spec = importlib.util.spec_from_file_location("p8_coupled_scales", MODULE_PATH)
scales_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scales_mod
spec.loader.exec_module(scales_mod)


def _fixture() -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    hidden, intermediate, experts = 128, 128, 2
    generator = torch.Generator().manual_seed(5308)
    tensors = {
        "gate_up_suh_fp16": (
            torch.rand(hidden, generator=generator, dtype=torch.float16) + 0.25
        ),
        "intermediate_scales_fp16": (
            torch.rand(experts, 3 * intermediate, generator=generator,
                       dtype=torch.float16) + 0.25
        ),
        "down_svh_fp16": (
            torch.rand(hidden, generator=generator, dtype=torch.float16) + 0.25
        ),
    }
    # Format evidence requires signed scales to be accepted.
    tensors["gate_up_suh_fp16"][3] *= -1
    tensors["intermediate_scales_fp16"][1, 17] *= -1
    tensors["down_svh_fp16"][11] *= -1
    metadata = {
        "schema": scales_mod.SCHEMA,
        "layer": "3",
        "rank": "0",
        "world_size": "4",
        "component": scales_mod.COMPONENT,
        "composition_target": scales_mod.COMPOSITION_TARGET,
        "boundary": "h128-suh-svh-scale-component",
        "cast_order": scales_mod.CAST_ORDER,
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        "full_coupled": "false",
        "signed_scales": "true",
    }
    metadata.update(
        {f"sha256_{name}": scales_mod.tensor_sha256(tensor)
         for name, tensor in tensors.items()}
    )
    return metadata, tensors


def test_schema_shapes_hashes_signed_values_and_packed_offsets() -> None:
    metadata, tensors = _fixture()
    checked = scales_mod.validate_scale_component(
        metadata, tensors, layer=3, rank=0, experts=2,
        hidden=128, intermediate=128,
    )
    gate, up, down = checked.split_intermediate()
    assert gate.shape == up.shape == down.shape == (2, 128)
    assert checked.packed.dtype == torch.float16
    assert checked.packed.numel() == 128 + 2 * 3 * 128 + 128
    assert torch.equal(checked.packed[:128], tensors["gate_up_suh_fp16"])
    assert torch.equal(checked.packed[-128:], tensors["down_svh_fp16"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m, t: m.__setitem__("full_coupled", "true"), "metadata"),
        (lambda m, t: m.__setitem__("signed_scales", "false"), "signed"),
        (lambda m, t: t["gate_up_suh_fp16"].__setitem__(0, 0), "zero-valued"),
        (lambda m, t: t["down_svh_fp16"].__setitem__(0, float("nan")), "non-finite"),
        (lambda m, t: m.__setitem__("sha256_down_svh_fp16", "0" * 64), "hash mismatch"),
    ],
)
def test_validator_fails_closed(mutation, match: str) -> None:
    metadata, tensors = _fixture()
    mutation(metadata, tensors)
    with pytest.raises(RuntimeError, match=match):
        scales_mod.validate_scale_component(
            metadata, tensors, layer=3, rank=0, experts=2,
            hidden=128, intermediate=128,
        )


def test_luke_input_cast_is_multiply_then_fp16_then_h128() -> None:
    generator = torch.Generator().manual_seed(53128)
    value = torch.randn(3, 256, generator=generator, dtype=torch.float16)
    suh = torch.randn(256, generator=generator, dtype=torch.float16)
    actual = scales_mod.had128_luke(value, suh=suh, store_fp16=False)
    had = scales_mod._hadamard_128()
    rounded = (value.float() * suh.float()).to(torch.float16).float()
    expected = (rounded.view(3, 2, 128) @ had).view(3, 256)
    wrong = ((value.float().view(3, 2, 128) @ had).view(3, 256)
             * suh.float())
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong)


def test_post_hadamard_values_determine_e4m3_payload_and_ue8m0_bytes() -> None:
    generator = torch.Generator().manual_seed(5332)
    value = torch.randn(2, 256, generator=generator, dtype=torch.bfloat16)
    suh = torch.randn(256, generator=generator, dtype=torch.float16)
    transformed = scales_mod.had128_luke(value, suh=suh, store_fp16=True)
    payload, scales, reconstruction = scales_mod.quantize_e4m3_ue8m0_per32(
        transformed
    )

    blocks = transformed.float().reshape(2, 8, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum / 448.0)).clamp(-127, 127)
    power = torch.pow(torch.tensor(2.0), exponent)
    expected_payload = (blocks / power).to(torch.float8_e4m3fn).view(torch.uint8)
    expected_scales = (exponent.squeeze(-1).to(torch.int16) + 127).to(torch.uint8)
    assert torch.equal(payload, expected_payload)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(
        reconstruction,
        expected_payload.view(torch.float8_e4m3fn).float().mul(power).reshape(2, 256),
    )

    # Amax before suh/H128 is the prohibited ordering and changes the bytes.
    wrong_payload, wrong_scales, _ = scales_mod.quantize_e4m3_ue8m0_per32(
        value.to(torch.float16)
    )
    assert not (
        torch.equal(payload, wrong_payload) and torch.equal(scales, wrong_scales)
    )


def test_complete_scale_sequence_is_bit_exact_to_explicit_luke_order() -> None:
    generator = torch.Generator().manual_seed(53128128)
    x = torch.randn(2, 128, generator=generator, dtype=torch.bfloat16)
    weights = [
        torch.randn(128, 128, generator=generator, dtype=torch.float16) / 16
        for _ in range(3)
    ]
    component = scales_mod.P8ScaleSandwich(
        gate_up_suh=torch.randn(128, generator=generator, dtype=torch.float16),
        intermediate_scales=torch.randn(
            1, 384, generator=generator, dtype=torch.float16
        ),
        down_svh=torch.randn(128, generator=generator, dtype=torch.float16),
    )
    actual = scales_mod.scale_sandwich_reference(
        x, weights[0], weights[1], weights[2], component
    )
    gate_svh, up_svh, down_suh = component.split_intermediate()
    source = scales_mod.had128_luke(
        x, suh=component.gate_up_suh, store_fp16=True
    )
    gate = (source.float() @ weights[0].float().T).to(torch.float16)
    up = (source.float() @ weights[1].float().T).to(torch.float16)
    gate = scales_mod.had128_luke(
        gate, svh=gate_svh[0], store_fp16=True
    )
    up = scales_mod.had128_luke(up, svh=up_svh[0], store_fp16=True)
    gate_work = gate.float().clamp(max=10.0)
    up_work = up.float().clamp(-10.0, 10.0)
    activation = (
        gate_work * torch.sigmoid(gate_work) * up_work
    ).to(torch.float16)
    down_input = scales_mod.had128_luke(
        activation, suh=down_suh[0], store_fp16=True
    )
    down = (down_input.float() @ weights[2].float().T).to(torch.float16)
    expected = scales_mod.had128_luke(
        down, svh=component.down_svh, store_fp16=False
    )
    assert torch.equal(actual, expected)


def test_runtime_sources_are_syntactic_and_scale_hooks_have_exact_seams() -> None:
    paths = {
        "wrapper": PATCH / "p8_native_kernel.py",
        "dynamic": PATCH / "b12x_h16/b12x/moe/_shared/kernels/dynamic.py",
        "fc1": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py",
        "fc2": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py",
    }
    source = {}
    for name, path in paths.items():
        source[name] = path.read_text()
        compile(source[name], str(path), "exec")

    assert "_p8_scale_input_before_h128" in source["dynamic"]
    assert "return cutlass.Float16(scaled).to(cutlass.Float32)" in source["dynamic"]
    assert "_scale_fc1_after_h128" in source["fc1"]
    assert "_scale_down_before_h128" in source["fc1"]
    assert "_scale_down_after_h128" in source["fc2"]
    assert "scale_component[down_svh_base + output_col]" in source["fc2"]
    assert "trellis_lut, scale_component, smem_base" in source["fc2"]
    assert "trellis_lut,\n                    trellis_rotations," in source["dynamic"]

    # Scale-only remains M1; full coupling additionally owns exact grouped
    # M64/N128 CTAs rather than moving svh across an ownership boundary.
    assert "One-CTA N128 owner for the H128 scale-sandwich boundary" in source["fc1"]
    assert "P8 scale sandwich requires exact M1 or full-coupled M64/N128 ownership" in source["dynamic"]
    assert "P8 scale sandwich currently supports M=1 only" in source["wrapper"]
    assert "p8_scale_sandwich=self.scale_component is not None" in source["wrapper"]
    assert "_w4a8_had128_quad" in source["fc1"]
    assert "if cutlass.const_expr(self.scale_sandwich):" in source["fc1"]
    assert "m1_block_max" in source["dynamic"]
    assert source["dynamic"].index("_p8_scale_input_before_h128(") < source[
        "dynamic"
    ].index("m1_block_max = cutlass.Float32(0.0)", source["dynamic"].index("m1_h128 ="))
    assert '"full_coupled": "false"' in MODULE_PATH.read_text()
