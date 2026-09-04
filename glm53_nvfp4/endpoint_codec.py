"""Output-aware optimization of a directly consumable native-NVFP4 endpoint.

K4 has exactly as many branch symbols as E2M1 has physical nibble codes.  The
lossless K4 law is therefore the identity law: the branch is the endpoint
nibble.  This module optimizes those nibbles and their E4M3/16 scales against
the routed full Hessian.  It is a codec encoder, not LDLQ, and needs no runtime
dequantization into BF16.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .modelopt import E2M1_LEVELS, PackedNVFP4, pack_codes, unpack_codes
from .trellis_nvfp4 import (
    TrellisNVFP4,
    decode_trellis_endpoint,
    pack_trellis_edges,
    tensor_core_permutation,
)


@dataclass(frozen=True)
class EndpointSweep:
    sweep: int
    objective_before: float
    objective_after_codes: float
    objective_after_scales: float
    changed_codes: int
    changed_scales: int
    global_scale_changed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _positive_e4m3_levels(device: torch.device) -> torch.Tensor:
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    values = raw.view(torch.float8_e4m3fn).float()
    return torch.unique(values[torch.isfinite(values) & (values > 0)]).sort().values.to(device)


def _nearest(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    targets = values.float().clamp(min=float(levels[0]), max=float(levels[-1]))
    upper_index = torch.searchsorted(levels, targets).clamp(max=levels.numel() - 1)
    lower_index = (upper_index - 1).clamp(min=0)
    lower, upper = levels[lower_index], levels[upper_index]
    return torch.where((targets - lower).abs() <= (upper - targets).abs(), lower, upper)


def _objective(error: torch.Tensor, hessian: torch.Tensor) -> float:
    return float(torch.einsum("ri,ij,rj->", error.double(), hessian.double(), error.double()).item())


@torch.no_grad()
def optimize_identity_k4_endpoint(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    initial: PackedNVFP4,
    *,
    sweeps: int = 2,
    ridge_ratio: float = 0.0,
) -> tuple[PackedNVFP4, list[EndpointSweep]]:
    """Monotonically optimize exact E2M1 codes and scales for ``E H E^T``.

    Each code update tests all 16 physical nibbles.  Each block-scale update is
    the exact continuous conditional minimizer projected to positive E4M3, and
    the tensor-global FP32 scale is then conditionally minimized.  Updates are
    accepted only when they do not increase the registered Hessian objective.
    """

    if weight.ndim != 2 or weight.device.type != "cuda" or weight.shape[1] % 16:
        raise ValueError("endpoint optimization requires a CUDA [out,in] tensor with K divisible by 16")
    rows, width = weight.shape
    if hessian.shape != (width, width) or sweeps <= 0 or ridge_ratio < 0:
        raise ValueError("invalid Hessian or sweep count")
    device = weight.device
    h = hessian.to(device=device, dtype=torch.float32)
    if ridge_ratio:
        h = h.clone()
        h.diagonal().add_(ridge_ratio * h.diagonal().mean())
    normalized = unpack_codes(initial.weight.to(device)).float()
    block_scales = initial.weight_scale.to(device).float().clone()
    global_scale = initial.weight_scale_2.to(device).float().reshape(()).clone()
    if block_scales.shape != (rows, width // 16):
        raise ValueError("initial endpoint has incompatible scale geometry")
    code_values = torch.cat((E2M1_LEVELS, -E2M1_LEVELS)).to(device)
    e4m3_levels = _positive_e4m3_levels(device)
    order = []
    diagonal = torch.diagonal(h)
    for start in range(0, width, 16):
        order.extend((start + torch.argsort(diagonal[start:start + 16], descending=True)).tolist())
    history = []

    def rebuild() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (normalized.reshape(rows, -1, 16) * block_scales[..., None] * global_scale).reshape(rows, width)
        error = q - weight.float()
        return q, error, error @ h

    q, error, gradient = rebuild()
    for sweep in range(sweeps):
        before = _objective(error, h)
        changed_codes = 0
        for column in order:
            group = column // 16
            scale = block_scales[:, group] * global_scale
            candidates = code_values[None, :] * scale[:, None]
            current = q[:, column]
            cross = gradient[:, column] - error[:, column] * h[column, column]
            delta_error = candidates - weight[:, column, None]
            costs = h[column, column] * delta_error.square() + 2.0 * delta_error * cross[:, None]
            selected = costs.argmin(1)
            replacement = candidates.gather(1, selected[:, None]).squeeze(1)
            delta = replacement - current
            changed_codes += int((delta != 0).sum().item())
            normalized[:, column] = code_values[selected]
            q[:, column] = replacement
            error[:, column] += delta
            gradient += delta[:, None] * h[column, :][None, :]
        after_codes = _objective(error, h)

        changed_scales = 0
        for group, start in enumerate(range(0, width, 16)):
            stop = start + 16
            values = normalized[:, start:stop]
            current_scale = block_scales[:, group] * global_scale
            numerator = torch.einsum("ri,ri->r", values, gradient[:, start:stop])
            denominator = torch.einsum("ri,ij,rj->r", values, h[start:stop, start:stop], values).clamp_min(1e-30)
            target_scale = (current_scale - numerator / denominator).clamp_min(1e-30)
            proposed_block = _nearest(target_scale / global_scale, e4m3_levels)
            proposed_scale = proposed_block * global_scale
            delta = (proposed_scale - current_scale)[:, None] * values
            old_obj = torch.einsum("ri,ri->r", error, gradient)
            candidate_error = error.clone()
            candidate_error[:, start:stop] += delta
            candidate_gradient = gradient + delta @ h[start:stop, :]
            new_obj = torch.einsum("ri,ri->r", candidate_error, candidate_gradient)
            accept = new_obj <= old_obj + old_obj.abs() * 1e-7
            if bool(accept.any()):
                changed_scales += int(
                    (accept & (proposed_block != block_scales[:, group])).sum().item()
                )
                accepted_delta = torch.where(accept[:, None], delta, torch.zeros_like(delta))
                block_scales[:, group] = torch.where(accept, proposed_block, block_scales[:, group])
                q[:, start:stop] += accepted_delta
                error[:, start:stop] += accepted_delta
                gradient += accepted_delta @ h[start:stop, :]

        basis = (normalized.reshape(rows, -1, 16) * block_scales[..., None]).reshape(rows, width)
        denominator = torch.einsum("ri,ij,rj->", basis.double(), h.double(), basis.double()).clamp_min(1e-30)
        proposed_global = (
            torch.einsum("ri,ij,rj->", basis.double(), h.double(), weight.double()) / denominator
        ).float().clamp_min(1e-30)
        candidate_error = basis * proposed_global - weight.float()
        current_objective = _objective(error, h)
        candidate_objective = _objective(candidate_error, h)
        global_changed = candidate_objective <= current_objective
        if global_changed:
            global_scale = proposed_global
            q, error, gradient = rebuild()
        after_scales = _objective(error, h)
        if after_codes > before * (1.0 + 1e-6) or after_scales > after_codes * (1.0 + 1e-6):
            raise RuntimeError("endpoint coordinate sweep violated monotonicity")
        history.append(EndpointSweep(sweep, before, after_codes, after_scales, changed_codes, changed_scales, bool(global_changed)))

    levels = E2M1_LEVELS.to(device)
    magnitude = (normalized.abs()[..., None] - levels).abs().argmin(-1).to(torch.uint8)
    codes = magnitude | (((normalized < 0) & (magnitude != 0)).to(torch.uint8) << 3)
    endpoint = PackedNVFP4(
        weight=pack_codes(codes).cpu(),
        weight_scale=block_scales.to(torch.float8_e4m3fn).cpu(),
        weight_scale_2=global_scale.cpu(),
    )
    return endpoint, history


@torch.no_grad()
def pack_identity_k4_stream(endpoint: PackedNVFP4) -> TrellisNVFP4:
    """Represent native nibbles as a bit-exact K4 sliding-state stream.

    The law reads the low four state bits, so every predecessor offers every
    physical E2M1 code.  This is lossless and shows why the decoded endpoint
    itself is a strictly cheaper hot-path representation than a K4 prologue.
    """

    packed = endpoint.weight.cpu()
    low, high = packed & 0x0F, packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    rows, width = codes.shape
    if rows % 16 or width % 16:
        raise ValueError("identity K4 stream requires 16x16-aligned endpoint")
    out_tiles, input_tiles = rows // 16, width // 16
    tiles = codes.reshape(out_tiles, 16, input_tiles, 16).permute(
        2, 0, 3, 1
    ).contiguous().reshape(-1, 256)
    tiles = tiles.index_select(1, tensor_core_permutation())
    states = tiles.reshape(input_tiles, out_tiles, 256).to(torch.int16)
    state_codes = (torch.arange(1 << 16, dtype=torch.int64) & 0x0F).to(torch.uint8)
    magnitudes = E2M1_LEVELS[(state_codes & 7).long()]
    values = torch.where((state_codes & 8) != 0, -magnitudes, magnitudes)
    values[values == 0] = 0
    payload = TrellisNVFP4(
        endpoint=endpoint,
        trellis=pack_trellis_edges(states, 4),
        bits=4,
        codebook_e4m3=values.to(torch.float8_e4m3fn).view(torch.uint8),
        codebook_law="identity-low4",
    )
    decoded = decode_trellis_endpoint(payload)
    if not torch.equal(decoded.weight, endpoint.weight):
        raise RuntimeError("identity K4 stream failed bit-exact endpoint closure")
    return payload
