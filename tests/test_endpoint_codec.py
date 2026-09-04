import pytest
import torch

from glm53_nvfp4.block_gptq import full_gptq_quantize
from glm53_nvfp4.endpoint_codec import optimize_identity_k4_endpoint, pack_identity_k4_stream
from glm53_nvfp4.modelopt import dequantize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_identity_k4_endpoint_coordinate_sweeps_are_monotone_and_native():
    generator = torch.Generator(device="cuda").manual_seed(2026090418)
    weight = torch.randn((16, 32), generator=generator, device="cuda") * 0.05
    samples = torch.randn((64, 32), generator=generator, device="cuda")
    hessian = samples.T @ samples / samples.shape[0]
    initial = full_gptq_quantize(weight, hessian)
    endpoint, history = optimize_identity_k4_endpoint(
        weight, hessian, initial, sweeps=2, ridge_ratio=0.1
    )
    assert len(history) == 2
    assert all(row.objective_after_codes <= row.objective_before * (1 + 1e-6) for row in history)
    assert all(row.objective_after_scales <= row.objective_after_codes * (1 + 1e-6) for row in history)
    assert history[-1].objective_after_scales <= history[0].objective_before
    assert endpoint.weight.dtype == torch.uint8
    assert endpoint.weight_scale.dtype == torch.float8_e4m3fn
    assert dequantize(endpoint).shape == weight.shape
    stream = pack_identity_k4_stream(endpoint)
    assert stream.trellis_bpw == 4.0
    assert stream.scale_bpw == 0.5
