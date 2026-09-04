"""Trellis pseudoquantization into SM120 MXF FP4/FP6/FP8 alphabets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from .block_gptq import _prepare_full_inverse
from .modelopt import E2M1_LEVELS
from .trellis_nvfp4 import (
    TILE_VALUES,
    _encode_tiles,
    _quantized_tiles_to_blocks,
    _validate_bits,
    pack_trellis_edges,
    procedural_state_values,
    sqg_xor_cheb_t12_e4m3_lut,
    tensor_core_permutation,
    reconstruct_trellis_states,
    unpack_trellis_edges,
)


@dataclass(frozen=True)
class TrellisMXF:
    reconstruction: torch.Tensor
    scales: torch.Tensor
    trellis: torch.Tensor
    codebook_e4m3: torch.Tensor
    bits: int
    alphabet: str
    block_size: int
    codebook_law: str
    compander_scale: float
    runtime_table_e4m3: torch.Tensor | None = None

    @property
    def stored_bpw(self) -> float:
        return self.bits + 8.0 / self.block_size


def pack_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Pack positive power-of-two scales using the UE8M0 exponent bias."""
    values = scales.float()
    exponents = torch.log2(values)
    if not bool(torch.isfinite(exponents).all()) or not torch.equal(exponents, exponents.round()):
        raise ValueError("UE8M0 scales must be finite positive powers of two")
    codes = exponents.round().to(torch.int16) + 127
    if bool(((codes < 0) | (codes > 254)).any()):
        raise ValueError("UE8M0 scale exponent is outside finite encoding range")
    return codes.to(torch.uint8).contiguous()


def unpack_ue8m0(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or bool((codes == 255).any()):
        raise ValueError("UE8M0 codes must be finite uint8 values")
    return torch.pow(2.0, codes.to(torch.int16).float() - 127)


def _finite_minifloat_levels(exponent_bits: int, mantissa_bits: int, bias: int) -> torch.Tensor:
    positive = []
    for exponent in range(1 << exponent_bits):
        for mantissa in range(1 << mantissa_bits):
            if exponent == 0:
                value = (mantissa / (1 << mantissa_bits)) * 2.0 ** (1 - bias)
            else:
                value = (1.0 + mantissa / (1 << mantissa_bits)) * 2.0 ** (exponent - bias)
            positive.append(value)
    values = torch.tensor(positive, dtype=torch.float32).unique(sorted=True)
    return torch.cat((-values[1:].flip(0), values))


@lru_cache(maxsize=None)
def alphabet_levels(alphabet: str) -> torch.Tensor:
    if alphabet == "e2m1":
        positive = E2M1_LEVELS.float()
        return torch.cat((-positive[1:].flip(0), positive))
    if alphabet == "e2m3":
        return _finite_minifloat_levels(2, 3, 1)
    if alphabet == "e3m2":
        return _finite_minifloat_levels(3, 2, 3)
    if alphabet == "e4m3":
        raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        values = raw.view(torch.float8_e4m3fn).float()
        return values[torch.isfinite(values)].unique(sorted=True)
    raise ValueError(f"unsupported MXF alphabet: {alphabet}")


def nearest_levels(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    levels = levels.to(values.device)
    upper_index = torch.searchsorted(levels, values.float()).clamp(max=levels.numel() - 1)
    lower_index = (upper_index - 1).clamp(min=0)
    lower = levels[lower_index]
    upper = levels[upper_index]
    return torch.where((values - lower).abs() <= (upper - values).abs(), lower, upper)


def state_lut(
    bits: int,
    *,
    alphabet: str,
    law: str,
    compander_scale: float,
    device: torch.device | str,
) -> torch.Tensor:
    _validate_bits(bits)
    if not math.isfinite(compander_scale) or compander_scale <= 0:
        raise ValueError("compander scale must be positive and finite")
    if law == "sqg-xor-cheb-t12":
        values = sqg_xor_cheb_t12_e4m3_lut(bits).view(torch.float8_e4m3fn).float()
    elif law in ("mcg", "mul1"):
        values = procedural_state_values(law).float()
    else:
        raise ValueError(f"unsupported trellis law: {law}")
    projected = nearest_levels(
        values * compander_scale,
        alphabet_levels(alphabet),
    )
    projected[projected == 0] = 0
    encoded = projected.to(torch.float8_e4m3fn)
    if not torch.equal(encoded.float(), projected):
        raise RuntimeError(f"{alphabet} contains values not exactly representable as E4M3")
    return encoded.view(torch.uint8).contiguous().to(device)


def expand_xor_t12_lut(
    bits: int,
    table_e4m3: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Expand one runtime-sized 4 KiB XOR-rank table for reference encoding."""
    _validate_bits(bits)
    table = table_e4m3.detach().to(dtype=torch.uint8, device="cpu").contiguous()
    if table.shape != (4096,):
        raise ValueError("T12 runtime table must contain exactly 4096 E4M3 bytes")
    values = table.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("T12 runtime table contains non-finite E4M3 values")
    from .trellis_nvfp4 import sqg_xor_rank_permutation

    buckets = sqg_xor_rank_permutation(bits) >> 4
    return table.index_select(0, buckets).to(device).contiguous()


def _best_power2_scales(weight: torch.Tensor, levels: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks = weight.float().reshape(weight.shape[0], -1, block_size)
    max_level = float(levels.abs().max())
    max_abs = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    base = torch.floor(torch.log2(max_abs / max_level))
    best_error = torch.full_like(max_abs, torch.inf)
    best_scale = torch.ones_like(max_abs)
    for offset in range(-3, 5):
        scale = torch.pow(2.0, base + offset)
        quantized = nearest_levels(blocks / scale[..., None], levels)
        error = (blocks - quantized * scale[..., None]).double().square().sum(-1)
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, scale, best_scale)
    return best_scale.contiguous()


def _refit_power2_scales(
    weight: torch.Tensor,
    normalized: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    blocks = weight.float().reshape_as(normalized)
    numerator = (blocks.double() * normalized.double()).sum(-1)
    denominator = normalized.double().square().sum(-1)
    target = (numerator / denominator.clamp_min(1e-30)).float().clamp_min(2.0**-126)
    lower = torch.pow(2.0, torch.floor(torch.log2(target)))
    upper = lower * 2.0
    lower_error = (blocks - normalized * lower[..., None]).double().square().sum(-1)
    upper_error = (blocks - normalized * upper[..., None]).double().square().sum(-1)
    fitted = torch.where(lower_error <= upper_error, lower, upper)
    return torch.where(denominator > 0, fitted, previous).contiguous()


def _prepare_mxf_tiles(weight: torch.Tensor, scales: torch.Tensor, block_size: int) -> torch.Tensor:
    rows, width = weight.shape
    normalized = (
        weight.float().reshape(rows, width // block_size, block_size)
        / scales.float()[..., None]
    ).reshape(rows, width)
    out_tiles, input_tiles = rows // 16, width // 16
    tiles = normalized.reshape(out_tiles, 16, input_tiles, 16).permute(
        2, 0, 3, 1
    ).contiguous().reshape(-1, TILE_VALUES)
    return tiles.index_select(1, tensor_core_permutation(weight.device))


def _prepare_native_group_tiles(normalized_group: torch.Tensor) -> torch.Tensor:
    """Map one native 16-column group to the frozen tensor-core lane order."""
    rows, width = normalized_group.shape
    if rows % 16 or width != 16:
        raise ValueError("native trellis group must have shape [16*N, 16]")
    tiles = normalized_group.reshape(rows // 16, 16, 16).permute(
        0, 2, 1
    ).contiguous()
    return tiles.reshape(-1, TILE_VALUES).index_select(
        1, tensor_core_permutation(normalized_group.device)
    )


def _decode_native_group_tiles(
    quantized_tiles: torch.Tensor, *, rows: int
) -> torch.Tensor:
    return _quantized_tiles_to_blocks(
        quantized_tiles, rows=rows, width=16
    ).reshape(rows, 16)


@torch.no_grad()
def _encode_mxf_with_gptq_feedback(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor,
    *,
    bits: int,
    block_size: int,
    tailbite_context: int,
    percdamp: float,
    column_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode native trellis groups while carrying GPTQ error between them.

    The trellis is atomic over each physical 16-column group, so its Viterbi
    decision is made in native order. Hessian feedback is evaluated in a
    fixed, within-group activation order; group membership and the runtime
    decoder layout never change.
    """
    rows, width = weight.shape
    if hessian.shape != (width, width):
        raise ValueError("Hessian shape must match the weight width")
    if column_block % 16:
        raise ValueError("column block must preserve native trellis groups")
    hinv, permutation = _prepare_full_inverse(hessian, percdamp, 16)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(width, device=weight.device)
    work = weight.float()[:, permutation].clone()
    input_tiles, out_tiles = width // 16, rows // 16
    states_native = torch.empty(
        (input_tiles, out_tiles, TILE_VALUES),
        dtype=torch.int16,
        device=weight.device,
    )

    for slab_start in range(0, width, column_block):
        slab_end = min(slab_start + column_block, width)
        slab = work[:, slab_start:slab_end]
        errors = torch.zeros_like(slab)
        inverse_slab = hinv[slab_start:slab_end, slab_start:slab_end]
        for local_start in range(0, slab_end - slab_start, 16):
            absolute_start = slab_start + local_start
            local_order = (
                permutation[absolute_start : absolute_start + 16] - absolute_start
            )
            inverse_local = torch.argsort(local_order)
            current_permuted = slab[:, local_start : local_start + 16]
            current_native = current_permuted[:, inverse_local]
            scale = scales[:, absolute_start // block_size].float()
            normalized_native = current_native / scale[:, None].clamp_min(1e-30)
            tiles = _prepare_native_group_tiles(normalized_native)
            quantized_tiles, states = _encode_tiles(
                tiles, lut, bits=bits, tailbite_context=tailbite_context
            )
            quantized_native = _decode_native_group_tiles(
                quantized_tiles, rows=rows
            ) * scale[:, None]
            quantized_permuted = quantized_native[:, local_order]
            states_native[absolute_start // 16] = states.reshape(
                out_tiles, TILE_VALUES
            )

            # Codes are a joint Viterbi decision for this native group. Apply
            # sequential GPTQ feedback against that frozen group decision.
            for column in range(16):
                slab_column = local_start + column
                current = slab[:, slab_column]
                quantized = quantized_permuted[:, column]
                diagonal = inverse_slab[slab_column, slab_column]
                error = (current - quantized) / diagonal
                slab[:, slab_column:] -= (
                    error[:, None]
                    * inverse_slab[slab_column, slab_column:][None, :]
                )
                slab[:, slab_column] = quantized
                errors[:, slab_column] = error
        if slab_end < width:
            work[:, slab_end:] -= errors @ hinv[slab_start:slab_end, slab_end:]
    return work[:, inverse].contiguous(), states_native


def decode_trellis_mxf(
    trellis: torch.Tensor,
    codebook_e4m3: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    *,
    bits: int,
    block_size: int,
    rows: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Reference-decode a frozen trellis/UE8M0 payload to a dense pseudoquant tensor."""
    input_tiles, out_tiles = width // 16, rows // 16
    if trellis.shape != (input_tiles, out_tiles, 16 * bits):
        raise ValueError("frozen trellis shape disagrees with matrix geometry")
    if scale_ue8m0.shape != (rows, width // block_size):
        raise ValueError("frozen UE8M0 scale shape disagrees with matrix geometry")
    edges = unpack_trellis_edges(trellis.to(device), bits)
    states = reconstruct_trellis_states(edges, bits).to(torch.int64) & 0xFFFF
    codebook = codebook_e4m3.to(device).view(torch.float8_e4m3fn).float()
    quantized = codebook.index_select(0, states.flatten()).reshape_as(states)
    logical = _quantized_tiles_to_blocks(
        quantized.reshape(-1, TILE_VALUES), rows=rows, width=width
    ).reshape(rows, width)
    scales = unpack_ue8m0(scale_ue8m0.to(device))
    return (
        logical.reshape(rows, width // block_size, block_size) * scales[..., None]
    ).reshape(rows, width)


@torch.no_grad()
def quantize_scalar_mxf(
    weight: torch.Tensor,
    *,
    alphabet: str,
    block_size: int = 32,
    scale_refinement_iterations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size not in (16, 32) or weight.shape[1] % block_size:
        raise ValueError("MXF block size must be 16 or 32 and divide K")
    levels = alphabet_levels(alphabet).to(weight.device)
    scales = _best_power2_scales(weight, levels, block_size)
    for _ in range(scale_refinement_iterations):
        normalized = nearest_levels(
            weight.float().reshape(weight.shape[0], -1, block_size) / scales[..., None],
            levels,
        )
        scales = _refit_power2_scales(weight, normalized, scales)
    normalized = nearest_levels(
        weight.float().reshape(weight.shape[0], -1, block_size) / scales[..., None],
        levels,
    )
    return (normalized * scales[..., None]).reshape_as(weight), scales


@torch.no_grad()
def quantize_trellis_mxf(
    weight: torch.Tensor,
    *,
    bits: int,
    alphabet: str,
    law: str,
    compander_scale: float,
    block_size: int = 32,
    tailbite_context: int = 128,
    scale_refinement_iterations: int = 2,
) -> TrellisMXF:
    _validate_bits(bits)
    if weight.ndim != 2 or weight.device.type != "cuda":
        raise ValueError("trellis MXF requires a CUDA 2D tensor")
    if weight.shape[0] % 16 or weight.shape[1] % math.lcm(16, block_size):
        raise ValueError("trellis MXF weight shape is not tile/block aligned")
    if block_size not in (16, 32):
        raise ValueError("MXF block size must be 16 or 32")
    lut = state_lut(
        bits,
        alphabet=alphabet,
        law=law,
        compander_scale=compander_scale,
        device=weight.device,
    )
    codebook_levels = lut.view(torch.float8_e4m3fn).float().unique(sorted=True)
    scales = _best_power2_scales(weight, codebook_levels, block_size)
    states = None
    normalized_blocks = None
    for _ in range(scale_refinement_iterations):
        tiles = _prepare_mxf_tiles(weight, scales, block_size)
        quantized, states = _encode_tiles(
            tiles, lut, bits=bits, tailbite_context=tailbite_context
        )
        logical = _quantized_tiles_to_blocks(
            quantized, rows=weight.shape[0], width=weight.shape[1]
        ).reshape_as(weight)
        normalized_blocks = logical.reshape(weight.shape[0], -1, block_size)
        scales = _refit_power2_scales(weight, normalized_blocks, scales)
    tiles = _prepare_mxf_tiles(weight, scales, block_size)
    quantized, states = _encode_tiles(
        tiles, lut, bits=bits, tailbite_context=tailbite_context
    )
    logical = _quantized_tiles_to_blocks(
        quantized, rows=weight.shape[0], width=weight.shape[1]
    ).reshape_as(weight)
    reconstruction = (
        logical.reshape(weight.shape[0], -1, block_size) * scales[..., None]
    ).reshape_as(weight)
    input_tiles, out_tiles = weight.shape[1] // 16, weight.shape[0] // 16
    return TrellisMXF(
        reconstruction=reconstruction,
        scales=scales.cpu(),
        trellis=pack_trellis_edges(
            states.reshape(input_tiles, out_tiles, TILE_VALUES), bits
        ).cpu(),
        codebook_e4m3=lut.cpu(),
        bits=bits,
        alphabet=alphabet,
        block_size=block_size,
        codebook_law=law,
        compander_scale=compander_scale,
    )


@torch.no_grad()
def quantize_trellis_mxf_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: int,
    alphabet: str,
    law: str,
    compander_scale: float,
    block_size: int = 32,
    tailbite_context: int = 128,
    scale_refinement_iterations: int = 2,
    percdamp: float = 0.01,
    column_block: int = 128,
    t12_codebook_e4m3: torch.Tensor | None = None,
) -> TrellisMXF:
    """Hessian-aware trellis MXF with static in-group activation ordering."""
    _validate_bits(bits)
    if weight.ndim != 2 or weight.device.type != "cuda":
        raise ValueError("trellis MXF requires a CUDA 2D tensor")
    if weight.shape[0] % 16 or weight.shape[1] % math.lcm(16, block_size):
        raise ValueError("trellis MXF weight shape is not tile/block aligned")
    if block_size not in (16, 32):
        raise ValueError("MXF block size must be 16 or 32")
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Hessian shape must match weight width")
    if t12_codebook_e4m3 is None:
        lut = state_lut(
            bits,
            alphabet=alphabet,
            law=law,
            compander_scale=compander_scale,
            device=weight.device,
        )
        runtime_table = None
        codebook_law = law
    else:
        if alphabet != "e4m3":
            raise ValueError("learned T12 runtime tables currently target E4M3")
        runtime_table = t12_codebook_e4m3.detach().cpu().to(torch.uint8).contiguous()
        lut = expand_xor_t12_lut(bits, runtime_table, device=weight.device)
        codebook_law = "learned-xor-t12"
    codebook_levels = lut.view(torch.float8_e4m3fn).float().unique(sorted=True)
    scales = _best_power2_scales(weight, codebook_levels, block_size)
    reconstruction = None
    states = None
    for iteration in range(scale_refinement_iterations + 1):
        reconstruction, states = _encode_mxf_with_gptq_feedback(
            weight,
            hessian.to(weight.device),
            scales,
            lut,
            bits=bits,
            block_size=block_size,
            tailbite_context=tailbite_context,
            percdamp=percdamp,
            column_block=column_block,
        )
        if iteration < scale_refinement_iterations:
            normalized = reconstruction.reshape(
                weight.shape[0], -1, block_size
            ) / scales[..., None]
            scales = _refit_power2_scales(weight, normalized, scales)
    assert reconstruction is not None and states is not None
    return TrellisMXF(
        reconstruction=reconstruction,
        scales=scales.cpu(),
        trellis=pack_trellis_edges(states, bits).cpu(),
        codebook_e4m3=lut.cpu(),
        bits=bits,
        alphabet=alphabet,
        block_size=block_size,
        codebook_law=codebook_law,
        compander_scale=compander_scale,
        runtime_table_e4m3=runtime_table,
    )
