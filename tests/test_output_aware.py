import torch

from glm53_nvfp4.output_aware import (
    output_aware_weight_target,
    output_nmse,
    route_weighted_hessian,
)


def test_output_aware_target_reduces_held_out_linear_carrier_error() -> None:
    torch.manual_seed(7)
    weight = torch.randn(12, 8)
    carrier_map = torch.eye(8) + 0.08 * torch.randn(8, 8)
    fit_source = torch.randn(64, 8)
    fit_carrier = fit_source @ carrier_map
    validation_source = torch.randn(64, 8)
    validation_carrier = validation_source @ carrier_map
    route_weights = torch.linspace(0.2, 1.0, 64)
    target, receipt = output_aware_weight_target(
        weight,
        fit_source,
        fit_carrier,
        route_weights,
        ridge_ratio=1e-5,
    )
    baseline = output_nmse(
        weight, weight, validation_source, validation_carrier, route_weights
    )
    candidate = output_nmse(
        weight, target, validation_source, validation_carrier, route_weights
    )
    assert candidate < baseline * 0.02
    assert receipt.correction_weight_nmse > 0
    assert receipt.effective_sample_size < 64


def test_route_weighted_hessian_is_symmetric() -> None:
    samples = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    hessian = route_weighted_hessian(samples, torch.ones(6))
    assert hessian.shape == (4, 4)
    assert torch.allclose(hessian, hessian.T)
