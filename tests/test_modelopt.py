import torch

from glm53_nvfp4.modelopt import dequantize, pack_codes, quantize_gate_up_pair, swizzle_block_scale, unpack_codes, unswizzle_block_scale


def test_code_pack_roundtrip_both_orders():
    codes = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    for low_first in (True, False):
        packed = pack_codes(codes, low_first=low_first)
        values = unpack_codes(packed, low_first=low_first)
        expected_mag = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2).reshape(2, 8)
        expected = expected_mag.clone()
        expected[1] *= -1
        assert torch.equal(values, expected)


def test_gate_up_share_global_scale_and_shapes():
    generator = torch.Generator().manual_seed(17)
    gate = torch.randn(32, 64, generator=generator)
    up = torch.randn(32, 64, generator=generator) * 2
    qg, qu = quantize_gate_up_pair(gate, up)
    assert torch.equal(qg.weight_scale_2, qu.weight_scale_2)
    assert qg.weight.shape == (32, 32)
    assert qg.weight_scale.shape == (128, 4)
    assert qg.weight.dtype == torch.uint8
    assert qg.weight_scale.dtype == torch.float8_e4m3fn
    assert dequantize(qg).shape == gate.shape
    assert (dequantize(qg) - gate).pow(2).mean().sqrt() < gate.pow(2).mean().sqrt() * 0.2


def test_modelopt_scale_swizzle_roundtrip_with_padding():
    logical = torch.arange(130 * 5, dtype=torch.int32).reshape(130, 5).to(torch.float8_e4m3fn)
    physical = swizzle_block_scale(logical)
    assert physical.shape == (256, 8)
    recovered = unswizzle_block_scale(physical, 130, 5)
    assert torch.equal(recovered.view(torch.uint8), logical.view(torch.uint8))
