"""Exact ModelOpt-style NVFP4 E2M1 packing used by the GLM carrier."""
from __future__ import annotations

from dataclasses import dataclass

import torch

E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E4M3_MAX = 448.0


@dataclass(frozen=True)
class PackedNVFP4:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor


def _nearest_codes(x: torch.Tensor) -> torch.Tensor:
    levels = E2M1_LEVELS.to(x.device)
    mids = (levels[1:] + levels[:-1]) / 2
    mag = torch.bucketize(x.abs().float(), mids).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    return mag | sign


def pack_codes(codes: torch.Tensor, *, low_first: bool = True) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("NVFP4 input dimension must be even")
    a = codes[..., 0::2]
    b = codes[..., 1::2]
    return (a | (b << 4)) if low_first else ((a << 4) | b)


def unpack_codes(packed: torch.Tensor, *, low_first: bool = True) -> torch.Tensor:
    lo = packed & 0x0F
    hi = packed >> 4
    a, b = (lo, hi) if low_first else (hi, lo)
    codes = torch.stack((a, b), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    levels = E2M1_LEVELS.to(packed.device)
    magnitude = levels[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -magnitude, magnitude)


def choose_global_scale(*weights: torch.Tensor) -> torch.Tensor:
    if not weights:
        raise ValueError("at least one weight tensor is required")
    maximum = max(float(w.detach().abs().max()) for w in weights)
    return torch.tensor(max(maximum / (6.0 * E4M3_MAX), 1e-12), dtype=torch.float32)


@torch.no_grad()
def quantize(
    weight: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    group_size: int = 16,
    low_first: bool = True,
) -> PackedNVFP4:
    if weight.ndim != 2 or weight.shape[-1] % group_size:
        raise ValueError(f"expected [out,in] with input divisible by {group_size}, got {tuple(weight.shape)}")
    x = weight.detach().float().cpu()
    gs = choose_global_scale(x) if global_scale is None else torch.as_tensor(global_scale, dtype=torch.float32).cpu()
    blocks = x.reshape(x.shape[0], x.shape[1] // group_size, group_size)
    raw_scale = blocks.abs().amax(-1).clamp_min(1e-12) / (6.0 * gs)
    block_scale = raw_scale.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    real_scale = block_scale.float() * gs
    codes = _nearest_codes(blocks / real_scale[..., None]).reshape_as(x)
    return PackedNVFP4(
        weight=pack_codes(codes, low_first=low_first).contiguous(),
        weight_scale=block_scale.contiguous(),
        weight_scale_2=gs.reshape(()).contiguous(),
    )


@torch.no_grad()
def dequantize(packed: PackedNVFP4, *, group_size: int = 16, low_first: bool = True) -> torch.Tensor:
    codes = unpack_codes(packed.weight, low_first=low_first)
    blocks = codes.reshape(codes.shape[0], codes.shape[1] // group_size, group_size)
    return (blocks * packed.weight_scale.float()[..., None] * packed.weight_scale_2.float()).reshape_as(codes)


def quantize_gate_up_pair(gate: torch.Tensor, up: torch.Tensor) -> tuple[PackedNVFP4, PackedNVFP4]:
    shared = choose_global_scale(gate, up)
    return quantize(gate, global_scale=shared), quantize(up, global_scale=shared)

