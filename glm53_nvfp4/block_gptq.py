"""Group-16 GPTQ that emits the exact packed ModelOpt payload."""
from __future__ import annotations

import torch

from .modelopt import E2M1_LEVELS, E4M3_MAX, PackedNVFP4, choose_global_scale, dequantize, pack_codes


@torch.no_grad()
def block_hessian(samples: torch.Tensor, route_weights: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    x = samples.float()
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(f"invalid sample shape {tuple(x.shape)}")
    weights = route_weights.float().square()
    weights = weights / weights.sum().clamp_min(1e-20)
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    return torch.einsum("sbg,sbh,s->bgh", blocks, blocks, weights)


@torch.no_grad()
def full_hessian(samples: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
    """Route-weight-squared full Hessian used by the Qwen reference GPTQ."""
    x = samples.float()
    if x.ndim != 2:
        raise ValueError(f"invalid sample shape {tuple(x.shape)}")
    weights = route_weights.float().square()
    weights = weights / weights.sum().clamp_min(1e-20)
    xw = x * weights.sqrt()[:, None]
    return xw.transpose(0, 1) @ xw


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


def _in_group_act_order(diagonal: torch.Tensor, group_size: int) -> torch.Tensor:
    """Sort columns by Hessian diagonal without changing fixed group membership."""
    width = diagonal.numel()
    permutation = torch.empty(width, dtype=torch.long, device=diagonal.device)
    for start in range(0, width, group_size):
        order = torch.argsort(diagonal[start : start + group_size], descending=True)
        permutation[start : start + group_size] = start + order
    return permutation


def _prepare_full_inverse(
    hessian: torch.Tensor, percdamp: float, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h = hessian.float().clone()
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"full Hessian must be square, got {tuple(h.shape)}")
    diagonal = torch.diagonal(h)
    dead = diagonal <= 0
    idx = torch.arange(h.shape[0], device=h.device)
    h[idx, idx] = torch.where(dead, torch.ones_like(diagonal), diagonal)
    h[idx, idx] += percdamp * torch.diagonal(h).mean()
    permutation = _in_group_act_order(torch.diagonal(h), group_size)
    h = h[permutation][:, permutation]
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


def _fpquant_mse_scales(
    blocks: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    scale_search_iters: int = 100,
    max_scale_shrink_factor: float = 0.80,
    error_norm: float = 2.4,
) -> torch.Tensor:
    """Reproduce FP-Quant's MSE observer before its final E4M3 cast.

    Despite the observer's historical ``MSE`` name, the reference defaults to
    an L2.4 reconstruction objective and searches 100 linearly shrinking
    versions of the absmax/6 scale, down to 20.8% of that scale.
    """
    if scale_search_iters <= 0:
        raise ValueError("scale_search_iters must be positive")
    maximum = blocks.abs().amax(-1).clamp_min(1e-12)
    initial = maximum / 6.0
    levels = E2M1_LEVELS.to(blocks.device)
    mids = (levels[1:] + levels[:-1]) / 2
    best_error = torch.full_like(maximum, float("inf"), dtype=torch.float32)
    best_real_scale = initial.clone()
    for index in range(scale_search_iters):
        shrink = 1.0 - index * max_scale_shrink_factor / scale_search_iters
        candidate = initial * shrink
        code = torch.bucketize((blocks / candidate[..., None]).abs(), mids)
        dequantized = levels[code] * torch.sign(blocks) * candidate[..., None]
        error = (blocks - dequantized).abs().pow(error_norm).sum(-1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_real_scale = torch.where(better, candidate, best_real_scale)
    return (best_real_scale / global_scale).clamp(max=E4M3_MAX).to(
        torch.float8_e4m3fn
    )


@torch.no_grad()
def refine_global_scale(
    *weights: torch.Tensor,
    search_grid: int = 12,
    iterations: int = 2,
) -> torch.Tensor:
    """Refit one packed-representable FP32 scale for aligned weights."""
    if not weights or iterations < 0:
        raise ValueError("weights are required and iterations must be nonnegative")
    width = weights[0].shape[1]
    if any(weight.ndim != 2 or weight.shape[1] != width for weight in weights):
        raise ValueError("global-scale refit requires aligned 2D weights")
    combined = torch.cat([weight.float() for weight in weights], dim=0)
    global_scale = choose_global_scale(*weights).to(combined.device)
    levels = E2M1_LEVELS.to(combined.device)
    mids = (levels[1:] + levels[:-1]) / 2
    blocks = combined.reshape(combined.shape[0], width // 16, 16)
    for _ in range(iterations):
        block_scale = _best_scales(blocks, global_scale, search_grid)
        real_scale = block_scale.float() * global_scale
        codes = torch.bucketize((blocks / real_scale[..., None]).abs(), mids)
        dequantized = levels[codes] * torch.sign(blocks) * real_scale[..., None]
        gamma = (
            (blocks * dequantized).sum()
            / dequantized.square().sum().clamp_min(1e-20)
        ).clamp(0.8, 1.25)
        global_scale = global_scale * gamma
    return global_scale.reshape(())


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
def full_gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
    column_block: int = 128,
    search_grid: int = 8,
    percdamp: float = 0.01,
) -> PackedNVFP4:
    """Packed NVFP4 using the Qwen full-Hessian sequential GPTQ geometry.

    This preserves fixed 16-column storage groups, applies static act-order
    only inside each group, searches the group's scale after preceding GPTQ
    error updates, and carries error across 128-column processing slabs.  The
    only deliberate adaptation from the Qwen pseudo-quantizer is that every
    block scale remains exactly E4M3 times one checkpoint-representable FP32
    global scale.
    """
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError(f"invalid weight shape {tuple(weight.shape)}")
    rows, width = weight.shape
    if hessian.shape != (width, width):
        raise ValueError(
            f"Hessian shape {tuple(hessian.shape)} does not match width {width}"
        )
    if column_block % group_size:
        raise ValueError("column block must preserve fixed NVFP4 groups")
    device = weight.device
    gs = (
        choose_global_scale(weight).to(device)
        if global_scale is None
        else torch.as_tensor(global_scale, dtype=torch.float32, device=device)
    )
    hinv, permutation = _prepare_full_inverse(
        hessian.to(device), percdamp, group_size
    )
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(width, device=device)
    work = weight.float()[:, permutation].clone()
    block_scales = torch.empty(
        (rows, width // group_size), dtype=torch.float8_e4m3fn, device=device
    )
    levels = E2M1_LEVELS.to(device)
    mids = (levels[1:] + levels[:-1]) / 2

    for block_start in range(0, width, column_block):
        block_end = min(block_start + column_block, width)
        slab = work[:, block_start:block_end]
        errors = torch.zeros_like(slab)
        inverse_slab = hinv[block_start:block_end, block_start:block_end]
        for group_start in range(0, block_end - block_start, group_size):
            group_end = group_start + group_size
            current_group = slab[:, group_start:group_end]
            scale = _best_scales(
                current_group[:, None, :], gs, search_grid
            )[:, 0]
            group_index = (block_start + group_start) // group_size
            block_scales[:, group_index] = scale
            real_scale = scale.float() * gs
            for offset in range(group_size):
                column = group_start + offset
                current = slab[:, column]
                code = torch.bucketize((current / real_scale).abs(), mids)
                quantized = levels[code] * torch.sign(current) * real_scale
                diagonal = inverse_slab[column, column]
                error = (current - quantized) / diagonal
                slab[:, column:] -= (
                    error[:, None] * inverse_slab[column, column:][None, :]
                )
                slab[:, column] = quantized
                errors[:, column] = error
        if block_end < width:
            work[:, block_end:] -= (
                errors @ hinv[block_start:block_end, block_end:]
            )

    dequantized = work[:, inverse]
    real_scales = block_scales.float() * gs
    blocks = dequantized.reshape(rows, width // group_size, group_size)
    codes = torch.bucketize((blocks / real_scales[..., None]).abs(), mids).to(
        torch.uint8
    )
    codes |= ((blocks < 0).to(torch.uint8) << 3)
    return PackedNVFP4(
        weight=pack_codes(codes.reshape(rows, width)).cpu(),
        weight_scale=block_scales.cpu(),
        weight_scale_2=gs.reshape(()).cpu(),
    )


@torch.no_grad()
def mr_gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
    column_block: int = 128,
    search_grid: int = 8,
    percdamp: float = 0.01,
    scale_observer: str = "modelopt-mse46",
) -> PackedNVFP4:
    """Reference-faithful FP-Quant static-ActOrder GPTQ for NVFP4.

    FP-Quant computes each fixed group's scale before GPTQ, globally orders
    columns by the Hessian diagonal, and keeps the original group scale bound
    to each column while error feedback is applied in the permuted order.  The
    columns and codes are returned to their original order for native NVFP4
    storage.  This is deliberately separate from :func:`full_gptq_quantize`,
    whose earlier Qwen-style adaptation only orders within each fixed group and
    searches scales after preceding GPTQ updates.
    """
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError(f"invalid weight shape {tuple(weight.shape)}")
    rows, width = weight.shape
    if hessian.shape != (width, width):
        raise ValueError(
            f"Hessian shape {tuple(hessian.shape)} does not match width {width}"
        )
    if column_block % group_size:
        raise ValueError("column block must be a multiple of the NVFP4 group")

    device = weight.device
    x = weight.float()
    gs = (
        choose_global_scale(x).to(device)
        if global_scale is None
        else torch.as_tensor(global_scale, dtype=torch.float32, device=device)
    )

    # Ingredient 1: one fixed MSE-optimized grid per native group, computed
    # before any GPTQ feedback changes the working weights.
    original_blocks = x.reshape(rows, width // group_size, group_size)
    if scale_observer == "modelopt-mse46":
        block_scales = _best_scales(original_blocks, gs, search_grid)
    elif scale_observer == "fpquant-mse-l2.4":
        block_scales = _fpquant_mse_scales(original_blocks, gs)
    else:
        raise ValueError(f"unsupported MR-GPTQ scale observer {scale_observer!r}")

    # Ingredient 2: global static activation ordering.  A permuted column keeps
    # the scale of the native group to which it belongs in the stored tensor.
    h = hessian.float().clone()
    diagonal = torch.diagonal(h)
    dead = diagonal <= 0
    idx = torch.arange(width, device=device)
    h[idx, idx] = torch.where(dead, torch.ones_like(diagonal), diagonal)
    h[idx, idx] += percdamp * torch.diagonal(h).mean()
    permutation = torch.argsort(torch.diagonal(h), descending=True)
    inverse = torch.argsort(permutation)
    original_group = torch.arange(width, device=device) // group_size
    permuted_group = original_group[permutation]
    h = h[permutation][:, permutation]
    chol = torch.linalg.cholesky(h)
    h_inverse_chol = torch.linalg.cholesky(
        torch.cholesky_inverse(chol), upper=True
    )

    work = x[:, permutation].clone()
    permuted_codes = torch.empty((rows, width), dtype=torch.uint8, device=device)
    levels = E2M1_LEVELS.to(device)
    mids = (levels[1:] + levels[:-1]) / 2
    for block_start in range(0, width, column_block):
        block_end = min(block_start + column_block, width)
        slab = work[:, block_start:block_end]
        errors = torch.zeros_like(slab)
        inverse_slab = h_inverse_chol[block_start:block_end, block_start:block_end]
        for column in range(block_end - block_start):
            absolute_column = block_start + column
            group_index = permuted_group[absolute_column]
            real_scale = block_scales[:, group_index].float() * gs
            current = slab[:, column]
            magnitude_code = torch.bucketize(
                (current / real_scale).abs(), mids
            ).to(torch.uint8)
            quantized = levels[magnitude_code.long()] * torch.sign(current) * real_scale
            code = magnitude_code | ((current < 0).to(torch.uint8) << 3)
            diagonal_inverse = inverse_slab[column, column]
            error = (current - quantized) / diagonal_inverse
            slab[:, column:] -= (
                error[:, None] * inverse_slab[column, column:][None, :]
            )
            slab[:, column] = quantized
            errors[:, column] = error
            permuted_codes[:, absolute_column] = code
        if block_end < width:
            work[:, block_end:] -= (
                errors @ h_inverse_chol[block_start:block_end, block_end:]
            )

    codes = permuted_codes[:, inverse].contiguous()
    return PackedNVFP4(
        weight=pack_codes(codes).cpu(),
        weight_scale=block_scales.cpu(),
        weight_scale_2=gs.reshape(()).cpu(),
    )


@torch.no_grad()
def mse_rtn_quantize(
    weight: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
    search_grid: int = 8,
) -> PackedNVFP4:
    """RTN control with the exact same global scale and MSE scale-search grid."""
    x = weight.float()
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(f"invalid weight shape {tuple(x.shape)}")
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    gs = choose_global_scale(x).to(x.device) if global_scale is None else torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    block_scale = _best_scales(blocks, gs, search_grid)
    real_scale = block_scale.float() * gs
    levels = E2M1_LEVELS.to(x.device)
    mids = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize((blocks / real_scale[..., None]).abs(), mids).to(torch.uint8)
    codes |= ((blocks < 0).to(torch.uint8) << 3)
    return PackedNVFP4(
        weight=pack_codes(codes.reshape_as(x)).cpu(),
        weight_scale=block_scale.cpu(),
        weight_scale_2=gs.reshape(()).cpu(),
    )


@torch.no_grad()
def fpquant_rtn_quantize(
    weight: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
) -> PackedNVFP4:
    """RTN using the exact FP-Quant default L2.4 scale observer."""
    x = weight.float()
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(f"invalid weight shape {tuple(x.shape)}")
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    gs = (
        choose_global_scale(x).to(x.device)
        if global_scale is None
        else torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    )
    block_scale = _fpquant_mse_scales(blocks, gs)
    real_scale = block_scale.float() * gs
    levels = E2M1_LEVELS.to(x.device)
    mids = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize((blocks / real_scale[..., None]).abs(), mids).to(
        torch.uint8
    )
    codes |= ((blocks < 0).to(torch.uint8) << 3)
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
def full_weighted_error(
    weight: torch.Tensor, packed: PackedNVFP4, hessian: torch.Tensor
) -> float:
    delta = weight.float().cpu() - dequantize(packed)
    return float(torch.einsum("nk,kl,nl->", delta, hessian.float().cpu(), delta))


@torch.no_grad()
def compare_full_packed_to_rtn(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    candidate: PackedNVFP4,
    global_scale: torch.Tensor | float,
    search_grid: int = 8,
    candidate_method: str = "gptq",
    rtn_scale_observer: str = "modelopt-mse46",
) -> dict:
    gs = torch.as_tensor(global_scale, dtype=torch.float32)
    if rtn_scale_observer == "modelopt-mse46":
        baseline = mse_rtn_quantize(
            weight, global_scale=gs, search_grid=search_grid
        )
    elif rtn_scale_observer == "fpquant-mse-l2.4":
        baseline = fpquant_rtn_quantize(weight, global_scale=gs)
    else:
        raise ValueError(f"unsupported RTN scale observer {rtn_scale_observer!r}")
    candidate_error = full_weighted_error(weight, candidate, hessian)
    baseline_error = full_weighted_error(weight, baseline, hessian)
    result = {
        "candidate": candidate_error,
        "candidate_method": candidate_method,
        "rtn": baseline_error,
        "rtn_scale_observer": rtn_scale_observer,
        "ratio": candidate_error / baseline_error,
    }
    if candidate_method == "gptq":
        result["gptq"] = candidate_error
    return result


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
    search_grid: int = 8,
    candidate_method: str = "gptq",
) -> dict:
    gs = torch.as_tensor(global_scale, dtype=torch.float32)
    baseline = mse_rtn_quantize(weight, global_scale=gs, search_grid=search_grid)
    candidate_error = weighted_error(weight, candidate, hessian)
    baseline_error = weighted_error(weight, baseline, hessian)
    result = {
        "candidate": candidate_error,
        "candidate_method": candidate_method,
        "rtn": baseline_error,
        "ratio": candidate_error / baseline_error,
    }
    if candidate_method == "gptq":
        result["gptq"] = candidate_error
    return result
