import torch

from glm53_nvfp4.block_gptq import block_hessian, compare_to_rtn, gptq_quantize
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
