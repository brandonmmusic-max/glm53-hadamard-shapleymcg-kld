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


def test_runtime_sources_are_syntactic_and_scale_hooks_have_exact_seams() -> None:
    paths = {
        "wrapper": PATCH / "p8_native_kernel.py",
        "dynamic": PATCH / "b12x_h16/b12x/moe/_shared/kernels/dynamic.py",
        "fc1": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py",
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

    # Current N64 ownership cannot close an H128 split across two CTAs.  The
    # wrapper must reject this component before compiling or launching it.
    tree = ast.parse(source["wrapper"])
    call = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "P8NativeTPMoE")
    invoke = next(node for node in call.body if isinstance(node, ast.FunctionDef)
                  and node.name == "__call__")
    statements = [ast.unparse(node) for node in invoke.body]
    reject_index = next(i for i, text in enumerate(statements)
                        if "requires the coupled H512/H128/sign runtime" in text)
    compile_index = next(i for i, text in enumerate(statements)
                         if "self._compile" in text)
    assert reject_index < compile_index
    assert '"full_coupled": "false"' in MODULE_PATH.read_text()
