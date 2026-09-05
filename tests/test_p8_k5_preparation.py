"""CPU structural gates for the 5.25-bpw P8 quality arm."""
from pathlib import Path

import torch

from glm53_nvfp4.trellis_mxf import state_lut
from glm53_nvfp4.trellis_nvfp4 import (
    pack_trellis_edges,
    reconstruct_trellis_states,
    unpack_trellis_edges,
)


ROOT = Path(__file__).resolve().parents[1]


def test_k5_stream_pack_and_sliding_state_roundtrip() -> None:
    generator = torch.Generator().manual_seed(5305)
    edges = torch.randint(0, 32, (7, 256), generator=generator, dtype=torch.int16)
    packed = pack_trellis_edges(edges, 5)
    assert packed.shape == (7, 80)
    assert packed.numel() * packed.element_size() * 8 / edges.numel() == 5.0
    assert torch.equal(unpack_trellis_edges(packed, 5), edges)
    expected = reconstruct_trellis_states(edges, 5)
    actual = reconstruct_trellis_states(unpack_trellis_edges(packed, 5), 5)
    assert torch.equal(actual, expected)


def test_k5_mcg_alphabet_remains_native_e4m3() -> None:
    lut = state_lut(5, alphabet="e4m3", law="mcg", compander_scale=2.0, device="cpu")
    decoded = lut.view(torch.float8_e4m3fn).float()
    assert lut.shape == (65536,)
    assert torch.isfinite(decoded).all()
    assert torch.equal(decoded.to(torch.float8_e4m3fn).view(torch.uint8), lut)


def test_fused_runtime_admits_k5_but_small_m_remains_k4_only() -> None:
    dynamic = (
        ROOT / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
    ).read_text()
    decoder = (ROOT / "runtime_patch/p8_mcg/w4a8_mcg_decode.py").read_text()
    wrapper = (ROOT / "runtime_patch/p8_native_kernel.py").read_text()
    assert "trellis_bits not in (2, 3, 4, 5)" in dynamic
    assert "bits not in (3, 4, 5)" in dynamic
    assert "bits not in (3, 4, 5)" in decoder
    assert 'bits_text not in {"4", "5"}' in wrapper
    assert "P8 K5 does not use the K4-only small-M specialization" in wrapper
    assert "trellis_bits=self.trellis_bits" in wrapper
