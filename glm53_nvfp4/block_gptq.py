"""Group-16 GPTQ that emits the exact packed ModelOpt payload."""
from __future__ import annotations

import torch

from .modelopt import E2M1_LEVELS, E4M3_MAX, PackedNVFP4, choose_global_scale, dequantize, pack_codes, quantize


@torch.no_grad()
def block_hessian(samples: torch.Tensor, route_weights: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    x = samples.float()
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(f"invalid sample shape {tuple(x.shape)}")
    weights = route_weights.float().square()
    weights = weights / weights.sum().clamp_min(1e-20)
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    return torch.einsum("sbg,sbh,s->bgh", blocks, blocks, weights)


def _prepare_inverse(hessian: torch.Tensor, percdamp: float) -> tuple[torch.Tensor, torch.Tensor]:
    h = hessian.float().clone()
    group = h.shape[-1]
    idx = torch.arange(group, device=h.device)
    diagonal = h[:, idx, idx]
    diagonal = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal))
    h[:, idx, idx] = diagonal
    h[:, idx, idx] += percdamp * diagonal.mean(-1, keepdim=True)
    permutation = torch.argsort(diagonal, dim=-1, descending=True)
    h = torch.gather(h, 1, permutation[:, :, None].expand_as(h))
    h = torch.gather(h, 2, permutation[:, None, :].expand_as(h))
    chol = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inverse, upper=True), permutation


def _best_scales(blocks: torch.Tensor, global_scale: torch.Tensor, search_grid: int) -> torch.Tensor:
    # blocks [rows, blocks, 16]; result is E4M3 [rows, blocks]
    maximum = blocks.abs().amax(-1).clamp_min(1e-12)
    levels = E2M1_LEVELS.to(blocks.device)
    mids = (levels[1:] + levels[:-1]) / 2
    best_error = None
    best_scale = None
    for range_max in (6.0, 4.0):
        for shrink in torch.linspace(0.65, 1.0, search_grid, device=blocks.device):
            scale = (maximum / range_max * shrink / global_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
            real = scale.float() * global_scale
            code = torch.bucketize((blocks / real[..., None]).abs(), mids)
            deq = levels[code] * torch.sign(blocks) * real[..., None]
            error = (deq - blocks).square().sum(-1)
            if best_error is None:
                best_error, best_scale = error, scale
            else:
                better = error < best_error
                best_error = torch.where(better, error, best_error)
                best_scale = torch.where(better, scale, best_scale)
    return best_scale


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
    search_grid: int = 8,
    percdamp: float = 0.01,
) -> PackedNVFP4:
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError(f"invalid weight shape {tuple(weight.shape)}")
    device = weight.device
    x = weight.float()
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    if hessian.shape != (blocks.shape[1], group_size, group_size):
        raise ValueError(f"Hessian shape {tuple(hessian.shape)} does not match {tuple(blocks.shape)}")
    gs = choose_global_scale(x).to(device) if global_scale is None else torch.as_tensor(global_scale, dtype=torch.float32, device=device)
    hinv, permutation = _prepare_inverse(hessian.to(device), percdamp)
    work = torch.gather(blocks, 2, permutation[None, :, :].expand(blocks.shape[0], -1, -1)).clone()
    block_scale = _best_scales(work, gs, search_grid)
    real_scale = block_scale.float() * gs
    levels = E2M1_LEVELS.to(device)
    mids = (levels[1:] + levels[:-1]) / 2
    for column in range(group_size):
        current = work[:, :, column]
        code = torch.bucketize((current / real_scale).abs(), mids)
        quantized = levels[code] * torch.sign(current) * real_scale
        diagonal = hinv[:, column, column]
        error = (current - quantized) / diagonal[None, :]
        work[:, :, column:] -= error[:, :, None] * hinv[None, :, column, column:]
        work[:, :, column] = quantized
    inverse = torch.empty_like(permutation)
    inverse.scatter_(1, permutation, torch.arange(group_size, device=device)[None, :].expand_as(permutation))
    deq = torch.gather(work, 2, inverse[None, :, :].expand_as(work))
    # Scales are invariant to within-group permutation. Encode the final grid exactly.
    codes = torch.bucketize((deq / real_scale[..., None]).abs(), mids).to(torch.uint8)
    codes |= ((deq < 0).to(torch.uint8) << 3)
    return PackedNVFP4(
        weight=pack_codes(codes.reshape_as(x)).cpu(),
        weight_scale=block_scale.cpu(),
        weight_scale_2=gs.reshape(()).cpu(),
    )


@torch.no_grad()
def weighted_error(weight: torch.Tensor, packed: PackedNVFP4, hessian: torch.Tensor) -> float:
    delta = (weight.float().cpu() - dequantize(packed)).reshape(weight.shape[0], -1, 16)
    value = torch.einsum("nbg,bgh,nbh->", delta, hessian.float().cpu(), delta)
    return float(value)


@torch.no_grad()
def compare_to_rtn(weight: torch.Tensor, hessian: torch.Tensor, global_scale: torch.Tensor | float | None = None) -> dict:
    gs = choose_global_scale(weight) if global_scale is None else torch.as_tensor(global_scale, dtype=torch.float32)
    candidate = gptq_quantize(weight, hessian, global_scale=gs)
    return compare_packed_to_rtn(weight, hessian, candidate, gs)


@torch.no_grad()
def compare_packed_to_rtn(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    candidate: PackedNVFP4,
    global_scale: torch.Tensor | float,
) -> dict:
    gs = torch.as_tensor(global_scale, dtype=torch.float32)
    baseline = quantize(weight.cpu(), global_scale=gs)
    candidate_error = weighted_error(weight, candidate, hessian)
    baseline_error = weighted_error(weight, baseline, hessian)
    return {"gptq": candidate_error, "rtn": baseline_error, "ratio": candidate_error / baseline_error}
