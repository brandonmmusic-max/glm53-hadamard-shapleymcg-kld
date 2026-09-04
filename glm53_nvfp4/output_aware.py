"""Output-aware targets for native MXF trellis encoders.

The runtime still stores and decodes one native MXF weight matrix. Calibration
only changes the matrix presented to the trellis encoder so that its output on
the quantized activation carrier approximates the BF16 checkpoint output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class OutputAwareFit:
    ridge_ratio: float
    ridge_value: float
    samples: int
    effective_sample_size: float
    correction_scale: float
    correction_weight_nmse: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _normalized_route_weights(
    route_weights: torch.Tensor | None, samples: int, device: torch.device
) -> torch.Tensor:
    if route_weights is None:
        return torch.full((samples,), 1.0 / samples, device=device)
    weights = route_weights.to(device=device, dtype=torch.float32).square()
    if weights.ndim != 1 or weights.numel() != samples:
        raise ValueError("route weights must have one value per calibration sample")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("route weights must have positive finite energy")
    return weights / total


@torch.no_grad()
def route_weighted_hessian(
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Return the full input Hessian for the actual quantized carrier."""
    if carrier_samples.ndim != 2:
        raise ValueError("carrier samples must be a 2D matrix")
    x = carrier_samples.float()
    weights = _normalized_route_weights(route_weights, x.shape[0], x.device)
    xw = x * weights.sqrt()[:, None]
    return xw.transpose(0, 1) @ xw


@torch.no_grad()
def output_aware_weight_target(
    weight: torch.Tensor,
    source_samples: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor | None,
    *,
    ridge_ratio: float,
    correction_scale: float = 1.0,
    row_chunk: int = 256,
) -> tuple[torch.Tensor, OutputAwareFit]:
    """Fit a single-matmul target using a route-weighted sample-space solve.

    For checkpoint weight ``W``, source activation ``X``, and native carrier
    activation ``Xq``, this fits ``C`` such that ``Xq @ (W + C).T`` predicts
    ``X @ W.T``. The sample-space solve remains well-defined when calibration
    has many fewer rows than hidden dimensions.
    """
    if weight.ndim != 2 or source_samples.ndim != 2 or carrier_samples.ndim != 2:
        raise ValueError("weight and samples must be 2D")
    if source_samples.shape != carrier_samples.shape:
        raise ValueError("source and carrier sample shapes must match")
    if weight.shape[1] != source_samples.shape[1]:
        raise ValueError("weight width must match sample width")
    if ridge_ratio <= 0 or not torch.isfinite(torch.tensor(ridge_ratio)):
        raise ValueError("ridge ratio must be positive and finite")
    if correction_scale < 0 or not torch.isfinite(torch.tensor(correction_scale)):
        raise ValueError("correction scale must be nonnegative and finite")
    if row_chunk <= 0:
        raise ValueError("row chunk must be positive")

    device = weight.device
    source = source_samples.to(device=device, dtype=torch.float32)
    carrier = carrier_samples.to(device=device, dtype=torch.float32)
    weights = _normalized_route_weights(route_weights, source.shape[0], device)
    sqrt_weights = weights.sqrt()
    carrier_weighted = carrier * sqrt_weights[:, None]
    gram = carrier_weighted @ carrier_weighted.transpose(0, 1)
    ridge_value = float(
        ridge_ratio * torch.diagonal(gram).mean().clamp_min(1e-20).item()
    )
    gram.diagonal().add_(ridge_value)
    chol = torch.linalg.cholesky(gram)
    delta_input = source - carrier
    target = weight.float().clone()
    correction_energy = 0.0
    weight_energy = float(weight.float().double().square().sum().item())
    for start in range(0, weight.shape[0], row_chunk):
        stop = min(start + row_chunk, weight.shape[0])
        original = weight[start:stop].float()
        residual_output = delta_input @ original.transpose(0, 1)
        residual_output.mul_(sqrt_weights[:, None])
        coefficients = torch.cholesky_solve(residual_output, chol)
        correction = coefficients.transpose(0, 1) @ carrier_weighted
        correction.mul_(correction_scale)
        target[start:stop].add_(correction)
        correction_energy += float(correction.double().square().sum().item())
    effective_n = float(1.0 / weights.double().square().sum().item())
    return target, OutputAwareFit(
        ridge_ratio=float(ridge_ratio),
        ridge_value=ridge_value,
        samples=int(source.shape[0]),
        effective_sample_size=effective_n,
        correction_scale=float(correction_scale),
        correction_weight_nmse=correction_energy / max(weight_energy, 1e-30),
    )


@torch.no_grad()
def output_nmse(
    weight: torch.Tensor,
    candidate: torch.Tensor,
    source_samples: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor | None = None,
) -> float:
    """Route-weighted projection-output NMSE against the BF16 source path."""
    source = source_samples.to(weight.device, dtype=torch.float32)
    carrier = carrier_samples.to(weight.device, dtype=torch.float32)
    weights = _normalized_route_weights(route_weights, source.shape[0], weight.device)
    reference = source @ weight.float().transpose(0, 1)
    actual = carrier @ candidate.float().transpose(0, 1)
    error = actual - reference
    numerator = (error.double().square().sum(1) * weights.double()).sum()
    denominator = (reference.double().square().sum(1) * weights.double()).sum()
    return float((numerator / denominator.clamp_min(1e-30)).item())
