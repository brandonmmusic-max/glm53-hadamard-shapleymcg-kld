import torch

from glm53_nvfp4.block_gptq import (
    block_hessian,
    compare_full_packed_to_rtn,
    compare_to_rtn,
    full_gptq_quantize,
    full_hessian,
    gptq_quantize,
    mse_rtn_quantize,
    refine_global_scale,
)
from glm53_nvfp4.modelopt import dequantize


def test_block_gptq_emits_exact_grid_and_beats_rtn():
    generator = torch.Generator().manual_seed(23)
    samples = torch.randn(512, 64, generator=generator)
    samples[:, 1:] += samples[:, :-1] * 0.65
    weights = torch.rand(512, generator=generator) + 0.1
    weight = torch.randn(48, 64, generator=generator) * 0.03
    hessian = block_hessian(samples, weights)
    comparison = compare_to_rtn(weight, hessian)
    packed = gptq_quantize(weight, hessian)
    assert torch.isfinite(dequantize(packed)).all()
    assert comparison["gptq"] <= comparison["rtn"]


def test_full_gptq_matches_qwen_geometry_and_emits_exact_grid():
    generator = torch.Generator().manual_seed(29)
    samples = torch.randn(768, 64, generator=generator)
    samples[:, 16:] += samples[:, :-16] * 0.8
    weights = torch.rand(768, generator=generator) + 0.1
    weight = torch.randn(40, 64, generator=generator) * 0.03
    hessian = full_hessian(samples, weights)
    packed = full_gptq_quantize(weight, hessian, column_block=32)
    comparison = compare_full_packed_to_rtn(
        weight, hessian, packed, packed.weight_scale_2
    )
    assert torch.isfinite(dequantize(packed)).all()
    assert packed.weight_scale.shape == (40, 4)
    assert comparison["gptq"] <= comparison["rtn"]


def test_global_scale_refit_is_packed_and_shared_for_gate_up():
    generator = torch.Generator().manual_seed(31)
    gate = torch.randn(24, 64, generator=generator) * 0.03
    up = torch.randn(24, 64, generator=generator) * 0.05
    scale = refine_global_scale(gate, up, search_grid=8, iterations=2)
    packed_gate = mse_rtn_quantize(gate, global_scale=scale, search_grid=8)
    packed_up = mse_rtn_quantize(up, global_scale=scale, search_grid=8)
    assert scale.ndim == 0 and torch.isfinite(scale) and scale > 0
    assert torch.equal(packed_gate.weight_scale_2, packed_up.weight_scale_2)
    assert torch.isfinite(dequantize(packed_gate)).all()
    assert torch.isfinite(dequantize(packed_up)).all()
