import torch

from glm53_nvfp4.block_gptq import (
    _fpquant_mse_scales,
    block_hessian,
    compare_full_packed_to_rtn,
    compare_packed_to_rtn,
    compare_to_rtn,
    full_gptq_quantize,
    full_hessian,
    fpquant_rtn_quantize,
    gptq_quantize,
    mr_gptq_quantize,
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


def test_rtn_candidate_is_explicit_and_matches_its_control():
    generator = torch.Generator().manual_seed(37)
    samples = torch.randn(256, 64, generator=generator)
    route = torch.rand(256, generator=generator) + 0.1
    weight = torch.randn(32, 64, generator=generator) * 0.04
    hessian = block_hessian(samples, route)
    packed = mse_rtn_quantize(weight, search_grid=8)
    comparison = compare_packed_to_rtn(
        weight,
        hessian,
        packed,
        packed.weight_scale_2,
        search_grid=8,
        candidate_method="rtn",
    )
    assert comparison["candidate_method"] == "rtn"
    assert comparison["candidate"] == comparison["rtn"]
    assert comparison["ratio"] == 1.0
    assert "gptq" not in comparison


def test_mr_gptq_uses_fixed_native_group_scales_and_beats_rtn():
    generator = torch.Generator().manual_seed(41)
    samples = torch.randn(1024, 64, generator=generator)
    samples[:, 16:] += samples[:, :-16] * 0.72
    route = torch.rand(1024, generator=generator) + 0.1
    weight = torch.randn(48, 64, generator=generator) * 0.04
    hessian = full_hessian(samples, route)
    scale = refine_global_scale(weight, search_grid=8, iterations=2)
    rtn = mse_rtn_quantize(weight, global_scale=scale, search_grid=8)
    packed = mr_gptq_quantize(
        weight,
        hessian,
        global_scale=scale,
        column_block=32,
        search_grid=8,
    )
    comparison = compare_full_packed_to_rtn(
        weight,
        hessian,
        packed,
        scale,
        search_grid=8,
        candidate_method="mr-gptq",
    )
    assert torch.equal(packed.weight_scale, rtn.weight_scale)
    assert torch.isfinite(dequantize(packed)).all()
    assert comparison["candidate_method"] == "mr-gptq"
    assert comparison["candidate"] <= comparison["rtn"]


def test_fpquant_observer_searches_below_modelopt_grid_and_stays_native():
    blocks = torch.tensor(
        [[[6.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4] * 2]],
        dtype=torch.float32,
    )
    global_scale = torch.tensor(1.0 / 448.0)
    scales = _fpquant_mse_scales(blocks, global_scale)
    assert scales.dtype == torch.float8_e4m3fn
    assert scales.shape == (1, 1)
    assert torch.isfinite(scales.float()).all()
    assert 0 < float(scales.float()[0, 0] * global_scale) <= 1.0


def test_fpquant_mr_gptq_uses_matched_fpquant_rtn_control():
    generator = torch.Generator().manual_seed(43)
    samples = torch.randn(512, 32, generator=generator)
    route = torch.rand(512, generator=generator) + 0.1
    weight = torch.randn(24, 32, generator=generator) * 0.04
    hessian = full_hessian(samples, route)
    scale = torch.tensor(float(weight.abs().max()) / (6.0 * 448.0))
    packed = mr_gptq_quantize(
        weight,
        hessian,
        global_scale=scale,
        column_block=32,
        scale_observer="fpquant-mse-l2.4",
    )
    baseline = fpquant_rtn_quantize(weight, global_scale=scale)
    comparison = compare_full_packed_to_rtn(
        weight,
        hessian,
        packed,
        scale,
        candidate_method="fpquant-mr-gptq",
        rtn_scale_observer="fpquant-mse-l2.4",
    )
    assert torch.equal(packed.weight_scale, baseline.weight_scale)
    assert comparison["rtn_scale_observer"] == "fpquant-mse-l2.4"
    assert comparison["candidate"] <= comparison["rtn"]
