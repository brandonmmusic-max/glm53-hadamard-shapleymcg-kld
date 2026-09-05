from __future__ import annotations

import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = (
    ROOT
    / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
)


def _had128_unnormalized(value: torch.Tensor) -> torch.Tensor:
    """CPU transcription of `_p8_had128_quad_unnormalized`."""

    shape = value.shape
    lanes = value.float().reshape(-1, 32, 4)
    v0, v1, v2, v3 = lanes.unbind(-1)
    s0, d0 = v0 + v1, v0 - v1
    s1, d1 = v2 + v3, v2 - v3
    work = [s0 + s1, d0 + d1, s0 - s1, d0 - d1]
    lane_ids = torch.arange(32)
    for stride in (1, 2, 4, 8, 16):
        peer = lane_ids ^ stride
        work = [
            torch.where(
                (lane_ids & stride) != 0,
                item.index_select(-1, peer) - item,
                item.index_select(-1, peer) + item,
            )
            for item in work
        ]
    return torch.stack(work, dim=-1).reshape(shape)


def _canonical_h512(value: torch.Tensor) -> torch.Tensor:
    quarters = _had128_unnormalized(value.reshape(-1, 4, 128))
    x0, x1, x2, x3 = quarters.unbind(1)
    r0, r1 = x0 + x1, x0 - x1
    r2, r3 = x2 + x3, x2 - x3
    return (
        torch.stack((r0 + r2, r1 + r3, r0 - r2, r1 - r3), dim=1)
        / math.sqrt(512)
    ).reshape_as(value)


def _canonical_h128(value: torch.Tensor) -> torch.Tensor:
    return (
        _had128_unnormalized(value.reshape(-1, 128)) / math.sqrt(128)
    ).reshape_as(value)


def test_torch_terminal_normalization_is_f32_division_not_reciprocal_multiply():
    values = torch.tensor([1.0, 3.0, 127.0, 1e-7, -31.0], dtype=torch.float32)
    expected = {
        128: (0x413504F3, 0x3DB504F3),
        512: (0x41B504F3, 0x3D3504F3),
    }
    for size, (divisor_bits, reciprocal_bits) in expected.items():
        divisor = torch.tensor(math.sqrt(size), dtype=torch.float32)
        reciprocal = torch.tensor(1.0 / math.sqrt(size), dtype=torch.float32)
        assert int(divisor.view(torch.int32)) == divisor_bits
        assert int(reciprocal.view(torch.int32)) == reciprocal_bits
        assert torch.equal(values / math.sqrt(size), values / divisor)
        assert not torch.equal(values / divisor, values * reciprocal)


def test_canonical_decomposition_is_bit_exact_to_cpu_hadamard_reference():
    from runtime_patch.p8_coupled_scales import hadamard_blocks

    generator = torch.Generator(device="cpu").manual_seed(20260905128)
    value = (torch.randn(2, 4096, generator=generator) * 0.03125).to(
        torch.bfloat16
    ).to(torch.float16).float()
    suh = torch.randn(4096, generator=generator).to(torch.float16).float()

    outer = _canonical_h512(value)
    assert torch.equal(outer, hadamard_blocks(value, 512))
    actual = _canonical_h128(outer * suh)
    expected = hadamard_blocks(hadamard_blocks(value, 512) * suh, 128)
    assert torch.equal(actual, expected)


def test_both_full_coupled_input_owners_use_canonical_operation_order():
    source = DYNAMIC.read_text()
    assert "div_rn_f32," in source
    assert "_P8_SQRT128_F32 = 11.313708305358887" in source
    assert "_P8_SQRT512_F32 = 22.627416610717773" in source

    # Definition + normalized wrapper + one materialized/prefill call + one
    # monolithic M1 call.
    assert source.count("def _p8_had128_quad_unnormalized(") == 1
    assert source.count("_p8_had128_quad_unnormalized(") == 4
    assert source.count("_p8_h512_mix_reference_order(") == 3
    assert source.count("_p8_had128_quad_reference_order(") == 3

    materialized = source[
        source.index("def _store_p8_full_coupled_input_row(") :
        source.index("    @cute.jit\n    def __call__(", source.index("def _store_p8_full_coupled_input_row("))
    ]
    assert "cutlass.Float32(0.5)" not in materialized
    assert "_w4a8_had128_quad(" not in materialized
