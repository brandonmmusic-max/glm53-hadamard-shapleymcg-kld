from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
import sys
sys.path.insert(0, str(PATCH))

import p8_coupled_scales as coupled


def _fixture(*, rank: int = 0, experts: int = 2, hidden: int = 512, intermediate: int = 128):
    generator = torch.Generator().manual_seed(905128)
    tensors = {
        "gate_up_suh_fp16": torch.randn(hidden, generator=generator, dtype=torch.float16),
        "intermediate_scales_fp16": torch.randn(
            experts, 3 * intermediate, generator=generator, dtype=torch.float16
        ),
        "down_svh_fp16": torch.randn(hidden, generator=generator, dtype=torch.float16),
    }
    transform_sha = hashlib.sha256(b"encoder-coupled-transform-receipt").hexdigest()
    metadata = {
        "schema": coupled.COUPLED_SCHEMA,
        "layer": "3",
        "rank": str(rank),
        "world_size": "4",
        "bits": "4",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "ldlq": "false",
        "component": coupled.COUPLED_COMPONENT,
        "composition_target": coupled.COMPOSITION_TARGET,
        "boundary": coupled.COMPOSITION_TARGET,
        "cast_order": coupled.COUPLED_CAST_ORDER,
        "full_coupled": "true",
        "h512": "normalized-sylvester-512-v1",
        "h128": "normalized-sylvester-128-v1",
        "transform_id": coupled.TRANSFORM_ID,
        "fc1_interleave": "slot0-atom32-slot1-atom32-v1",
        "sign_generator": coupled.SIGN_GENERATOR,
        "sign_draw": str(coupled.SIGN_DRAW),
        "sign_pre_axis": "1",
        "sign_post_axis": "2",
        "activation": "silu-cap10",
        "global_intermediate": str(4 * intermediate),
        "local_atom_begin": str(rank * (intermediate // 32)),
        "tp_slice": "contiguous-atom32-v1",
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        "coupled_signs_shared": "true",
        "signed_scales": "true",
        "encoder_transform_sha256": transform_sha,
    }
    for name, tensor in tensors.items():
        metadata[f"sha256_{name}"] = coupled.tensor_sha256(tensor)
    signs = coupled.rank_local_coupled_signs(intermediate=intermediate, rank=rank)
    metadata["sha256_coupled_signs_fp16"] = coupled.tensor_sha256(signs)
    return metadata, tensors, transform_sha


def test_full_schema_regenerates_fixed_rank_local_signs_bit_exactly() -> None:
    metadata, tensors, transform_sha = _fixture(rank=2)
    result = coupled.validate_coupled_component(
        metadata, tensors, layer=3, rank=2, experts=2,
        hidden=512, intermediate=128,
        expected_transform_sha256=transform_sha,
    )
    assert result.full_coupled
    assert result.coupled_signs is not None
    assert result.coupled_signs.dtype == torch.float16
    assert set(result.coupled_signs.tolist()) == {1.0}
    assert torch.equal(
        result.coupled_signs,
        coupled.rank_local_coupled_signs(intermediate=128, rank=2),
    )


@pytest.mark.parametrize(
    "rank,expected",
    [
        *[(rank, "95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0")
          for rank in range(4)],
    ],
)
def test_glm_tp4_fixed_sign_bytes_have_frozen_hash(rank: int, expected: str) -> None:
    signs = coupled.rank_local_coupled_signs(intermediate=512, rank=rank)
    assert coupled.tensor_sha256(signs) == expected


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("sign_draw", "5", "invalid full-coupled"),
        ("local_atom_begin", "0", "invalid full-coupled"),
        ("sha256_coupled_signs_fp16", "0" * 64, "sign seed/hash"),
        ("encoder_transform_sha256", "0" * 64, "transform identity mismatch"),
    ],
)
def test_full_schema_fails_closed_on_encoder_or_sign_identity(
    field: str, value: str, match: str
) -> None:
    metadata, tensors, transform_sha = _fixture(rank=1)
    metadata[field] = value
    with pytest.raises(RuntimeError, match=match):
        coupled.validate_coupled_component(
            metadata, tensors, layer=3, rank=1, experts=2,
            hidden=512, intermediate=128,
            expected_transform_sha256=transform_sha,
        )


def test_scale_only_component_cannot_enter_full_reference() -> None:
    metadata, tensors, _ = _fixture()
    metadata.update(
        {
            "schema": coupled.SCHEMA,
            "component": coupled.COMPONENT,
            "boundary": "h128-suh-svh-scale-component",
            "cast_order": coupled.CAST_ORDER,
            "gate_up_suh_shared": "true",
            "down_svh_shared": "true",
            "full_coupled": "false",
        }
    )
    component = coupled.validate_scale_component(
        metadata, tensors, layer=3, rank=0, experts=2,
        hidden=512, intermediate=128,
    )
    z = torch.zeros
    with pytest.raises(RuntimeError, match="scale-only"):
        coupled.coupled_reference(
            z(1, 512), z(128, 512), z(128, 512), z(512, 128), component
        )


def test_h512_fwht_matches_explicit_normalized_sylvester_matrix() -> None:
    generator = torch.Generator().manual_seed(512)
    value = torch.randn(2, 1024, generator=generator)
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    had = h2
    for _ in range(8):
        had = torch.kron(had, h2)
    had /= 512.0**0.5
    expected = (value.view(-1, 512) @ had).view_as(value)
    actual = coupled.hadamard_blocks(value, 512)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_complete_coupled_reference_has_exact_boundary_order() -> None:
    generator = torch.Generator().manual_seed(5121286)
    x = torch.randn(2, 512, generator=generator, dtype=torch.bfloat16)
    gate = torch.randn(128, 512, generator=generator, dtype=torch.float16) / 16
    up = torch.randn(128, 512, generator=generator, dtype=torch.float16) / 16
    down = torch.randn(512, 128, generator=generator, dtype=torch.float16) / 16
    scales = coupled.P8ScaleSandwich(
        gate_up_suh=torch.randn(512, generator=generator, dtype=torch.float16),
        intermediate_scales=torch.randn(1, 384, generator=generator, dtype=torch.float16),
        down_svh=torch.randn(512, generator=generator, dtype=torch.float16),
        coupled_signs=coupled.rank_local_coupled_signs(intermediate=128, rank=0),
        full_coupled=True,
        transform_sha256="1" * 64,
    )
    actual = coupled.coupled_reference(x, gate, up, down, scales)

    gate_svh, up_svh, down_suh = scales.split_intermediate()
    pre_sign, post_sign = scales.split_signs()
    source = coupled.hadamard_blocks(x.to(torch.float16).float(), 512)
    source = coupled.hadamard_blocks(
        source * scales.gate_up_suh.float(), 128
    )
    gp = (source.float() @ gate.float().T).to(torch.float16)
    uph = (source.float() @ up.float().T).to(torch.float16)
    raw = torch.stack((gp.view(2, 4, 32), uph.view(2, 4, 32)), dim=2).reshape(2, 256)
    raw_scale = torch.stack(
        (gate_svh[0].view(4, 32), up_svh[0].view(4, 32)), dim=1
    ).reshape(256)
    pre = coupled.hadamard_blocks(raw.float(), 128)
    pre = coupled.hadamard_blocks(pre * raw_scale.float(), 128) * pre_sign.float()
    g, u = pre[:, 0::2], pre[:, 1::2]
    gc = g.clamp(max=10.0)
    uc = u.clamp(-10.0, 10.0)
    activated = gc * torch.sigmoid(gc) * uc
    middle = coupled.hadamard_blocks(activated * post_sign.float(), 128)
    middle = coupled.hadamard_blocks(middle * down_suh[0].float(), 128)
    physical = (middle @ down.float().T).to(torch.float16)
    route = coupled.hadamard_blocks(physical.float(), 128) * scales.down_svh.float()
    expected = coupled.hadamard_blocks(route, 512)
    assert torch.equal(actual, expected)


def test_runtime_source_has_n128_owner_and_replacement_reducer() -> None:
    paths = {
        "wrapper": PATCH / "p8_native_kernel.py",
        "dynamic": PATCH / "b12x_h16/b12x/moe/_shared/kernels/dynamic.py",
        "fc1": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py",
        "fc2": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py",
        "sum": PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py",
    }
    source = {name: path.read_text() for name, path in paths.items()}
    for name, path in paths.items():
        compile(source[name], str(path), "exec")
    assert "p8_full_coupled=self.full_coupled" in source["wrapper"]
    assert "trellis_identity_boundary=not self.full_coupled" in source["wrapper"]
    assert "BF16 -> FP16 -> H512 -> signed suh -> H128 (FP32)" in source["dynamic"]
    assert "slot0-atom32-slot1-atom32-v1" in (PATCH / "p8_coupled_scales.py").read_text()
    assert "self._coupled_activation" in source["fc1"]
    assert "if cutlass.const_expr(self.full_coupled):" in source["fc2"]
    assert "route values unweighted and FP32" in source["fc2"]
    assert "This replaces the ordinary B12X top-k reducer" in source["sum"]
    assert "self._compile_full_coupled_reducer()" in source["wrapper"]
    assert "full-coupled P8 requires an externally pinned encoder transform" in source["wrapper"]
