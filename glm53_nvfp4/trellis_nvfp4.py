"""SQG trellis selection with an exact native-NVFP4 numerical endpoint.

The stored symbol plane uses ``bits`` branch bits per coefficient.  Decoding
the cyclic 16-bit state produces *only* legal E2M1 values.  Those values and
the ordinary per-16 UE4M3 scale plane can therefore be materialized directly
as ModelOpt NVFP4 nibbles; no dense FP16/BF16 weight tile is part of the
format or the intended runtime path.

The SQG graph and rank mixer are derived from the audited KQuant reference
snapshot.  That snapshot did not contain a license file; provenance and the
unverified license status are recorded in THIRD_PARTY_NOTICES.md.  KQuant's
CUDA Viterbi encoder is loaded lazily so the format,
packer, and reference decoder remain testable without that optional encoder.

These legacy encoder LUTs use lower-magnitude midpoint ties and canonical
positive zero. The independently versioned, table-free native RNE P4 matrix
interchange is in p4_codec.py; do not relabel legacy payloads as that law.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from .block_gptq import _best_scales, _prepare_full_inverse
from .modelopt import E2M1_LEVELS, PackedNVFP4, dequantize, pack_codes


TILE_CHANNELS = 16
TILE_VALUES = TILE_CHANNELS * TILE_CHANNELS
TRANSITIONS = 1 << 16
DEFAULT_KQUANT_ROOT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm52_fresh_sqg_3p0625/kquant"
)


@dataclass(frozen=True)
class TrellisNVFP4:
    """Compressed source payload plus its exact native-NVFP4 endpoint."""

    endpoint: PackedNVFP4
    trellis: torch.Tensor
    bits: int
    codebook_e4m3: torch.Tensor
    codebook_law: str
    scale_refinement_iterations: int = 0
    initial_reconstruction_mse: float | None = None
    final_reconstruction_mse: float | None = None

    @property
    def trellis_bpw(self) -> float:
        weights = self.endpoint.weight.numel() * 2
        return self.trellis.numel() * self.trellis.element_size() * 8 / weights

    @property
    def scale_bpw(self) -> float:
        weights = self.endpoint.weight.numel() * 2
        scale_bytes = (
            self.endpoint.weight_scale.numel()
            * self.endpoint.weight_scale.element_size()
        )
        return scale_bytes * 8 / weights

    @property
    def codebook_sha256(self) -> str:
        raw = self.codebook_e4m3.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CodebookFitStep:
    iteration: int
    assigned_mse: float
    visited_states: int
    changed_labels: int


@dataclass(frozen=True)
class HybridTrellisNVFP4:
    """One K4 trellis tile plus a charged selector into shared law LUTs."""

    endpoint: PackedNVFP4
    trellis: torch.Tensor
    selectors: torch.Tensor
    selector_shape: tuple[int, int]
    bits: int
    codebooks_e4m3: torch.Tensor
    codebook_laws: tuple[str, ...]

    @property
    def selector_bpw(self) -> float:
        weights = self.endpoint.weight.numel() * 2
        return math.prod(self.selector_shape) * 2 / weights

    @property
    def stored_bpw_excluding_shared_luts_and_global_scalar(self) -> float:
        weights = self.endpoint.weight.numel() * 2
        trellis_bits = self.trellis.numel() * self.trellis.element_size() * 8
        scale_bits = (
            self.endpoint.weight_scale.numel()
            * self.endpoint.weight_scale.element_size()
            * 8
        )
        return (trellis_bits + scale_bits) / weights + self.selector_bpw


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("trellis bits must be an integer from 2 through 6")


def tensor_core_permutation(
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """EXL3's 16x16 row-major to Tensor-Core lane ordering."""

    permutation = [0] * TILE_VALUES
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        permutation[thread * 8 + 0] = row0 * 16 + column0
        permutation[thread * 8 + 1] = row1 * 16 + column0
        permutation[thread * 8 + 2] = row2 * 16 + column0
        permutation[thread * 8 + 3] = row3 * 16 + column0
        permutation[thread * 8 + 4] = row0 * 16 + column1
        permutation[thread * 8 + 5] = row1 * 16 + column1
        permutation[thread * 8 + 6] = row2 * 16 + column1
        permutation[thread * 8 + 7] = row3 * 16 + column1
    return torch.tensor(permutation, dtype=torch.long, device=device)


def _mix_width(
    values: torch.Tensor,
    *,
    width: int,
    multiplier_a: int,
    multiplier_b: int,
    shift_a: int,
    shift_b: int,
    shift_c: int,
) -> torch.Tensor:
    mask = (1 << width) - 1
    result = values & mask
    for multiplier, shift in (
        (multiplier_a | 1, shift_a),
        (multiplier_b | 1, shift_b),
    ):
        result ^= result >> min(max(shift, 1), width - 1)
        result = (result * multiplier) & mask
    result ^= result >> min(max(shift_c, 1), width - 1)
    return result & mask


def _reverse_low_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    result = torch.zeros_like(values)
    for index in range(bits):
        result |= ((values >> index) & 1) << (bits - 1 - index)
    return result


def sqg_rank_permutation(bits: int) -> torch.Tensor:
    """Map each transition state to KQuant's shared Gaussian rank."""

    _validate_bits(bits)
    width = 16 - bits
    branches = 1 << bits
    states = torch.arange(TRANSITIONS, dtype=torch.int64)
    history = states >> bits
    branch = states & (branches - 1)
    phase = _mix_width(
        history,
        width=width,
        multiplier_a=0x65AF,
        multiplier_b=0x16BF,
        shift_a=6,
        shift_b=4,
        shift_c=5,
    )
    syndrome_hash = _mix_width(
        history ^ 0x5105,
        width=width,
        multiplier_a=0x8693,
        multiplier_b=0x2A21,
        shift_a=2,
        shift_b=4,
        shift_c=4,
    )
    syndrome = syndrome_hash & (branches - 1)
    stratum = (7 * (_reverse_low_bits(branch, bits) ^ syndrome)) & (branches - 1)
    return ((stratum << width) | phase).contiguous()


def sqg_xor_rank_permutation(bits: int) -> torch.Tensor:
    """Return the mature SQG carry-mixed rank permutation.

    This is a byte-for-byte port of KQuant/QSRT's ``sqg_xor_rank_permutation``.
    Keeping it local makes the exact state-label contract testable without
    importing a mutable research checkout at runtime.
    """

    _validate_bits(bits)
    width = 16 - bits
    history_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    codeword = torch.arange(TRANSITIONS, dtype=torch.int64)
    history = codeword >> bits
    branch = codeword & branch_mask
    mixed = history ^ (history >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & history_mask
    syndrome = product >> (32 - bits)
    stratum = _reverse_low_bits(branch, bits) ^ syndrome
    return ((stratum << width) | phase).contiguous()


@lru_cache(maxsize=1)
def _sqg_xor_cheb_t12_rank_lut_e4m3() -> torch.Tensor:
    """Reproduce the frozen 4-KiB modal T12 E4M3 staircase."""

    ranks = torch.arange(TRANSITIONS, dtype=torch.float64)
    probability = (ranks + 0.5) / TRANSITIONS
    exact = (1.5 * torch.special.ndtri(probability)).float()
    raw = exact.to(torch.float8_e4m3fn).view(torch.uint8).contiguous().clone()
    raw[(raw & 0x7F) == 0] = 0
    blocks = raw.reshape(1 << 12, 16)
    result = torch.empty(1 << 12, dtype=torch.uint8)
    for index, block in enumerate(blocks):
        labels, counts = torch.unique(block, return_counts=True)
        result[index] = labels[counts == counts.max()].min()
    return result.contiguous()


def sqg_xor_cheb_t12_e4m3_lut(bits: int) -> torch.Tensor:
    """Return mature SQG-XOR-Cheb-T12's exact pre-projection E4M3 LUT."""

    ranks = sqg_xor_rank_permutation(bits)
    return _sqg_xor_cheb_t12_rank_lut_e4m3().index_select(
        0, ranks >> 4
    ).contiguous()


def e2m1_rank_lut(
    bits: int = 3,
    *,
    compander_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a 65,536-entry E4M3 byte LUT whose values are all legal E2M1.

    The monotone law is a normal quantile staircase scaled by 1.5, matching
    SQG's calibration domain. ``compander_scale`` is an encoder-selection
    parameter; it changes which E2M1 labels are offered but never changes the
    native endpoint alphabet.
    """

    _validate_bits(bits)
    if not math.isfinite(compander_scale) or compander_scale <= 0:
        raise ValueError("compander_scale must be positive and finite")
    rank = sqg_rank_permutation(bits)
    probability = ((rank.double() + 0.5) / TRANSITIONS).clamp(
        1.0 / 2048.0, 1.0 - 1.0 / 2048.0
    )
    # erfinv is the direct normal inverse CDF and is stable over the clamp.
    gaussian = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    values = (1.5 * compander_scale * gaussian).float()
    levels = E2M1_LEVELS
    magnitude = levels[(values.abs()[:, None] - levels[None, :]).abs().argmin(1)]
    quantized = torch.where(values < 0, -magnitude, magnitude)
    quantized[quantized == 0] = 0
    raw = quantized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    decoded = raw.view(torch.float8_e4m3fn).float()
    legal_values = torch.cat((-levels[1:].flip(0), levels))
    legal = (decoded[:, None] == legal_values[None, :]).any(1)
    if not bool(legal.all()):
        raise RuntimeError("codebook escaped the E2M1 alphabet")
    return raw.to(device=device)


def procedural_state_values(law: str) -> torch.Tensor:
    """Reconstruct the original EXL3 procedural state alphabet in FP16."""

    states = torch.arange(TRANSITIONS, dtype=torch.int64)
    if law == "mcg":
        product = (states * 0xCBAC1FED) & 0xFFFFFFFF
        word = (product & 0x8FFF8FFF) ^ 0x3B603B60
        low = (word & 0xFFFF).to(torch.uint16).view(torch.float16)
        high = ((word >> 16) & 0xFFFF).to(torch.uint16).view(torch.float16)
        return (low + high).to(torch.float16).contiguous()
    if law == "mul1":
        product = (states * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = sum((product >> shift) & 0xFF for shift in (0, 8, 16, 24))
        source = ((byte_sum + 0x6400) & 0xFFFF).to(torch.uint16).view(
            torch.float16
        )
        inverse = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16)
        bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16)
        return (source.float() * inverse.float() + bias.float()).to(
            torch.float16
        ).contiguous()
    raise ValueError("procedural codebook law must be 'mcg' or 'mul1'")


def e2m1_state_lut(
    bits: int = 3,
    *,
    law: str = "mcg",
    compander_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Project an optimized EXL3 state law onto the exact E2M1 alphabet."""

    _validate_bits(bits)
    if not math.isfinite(compander_scale) or compander_scale <= 0:
        raise ValueError("compander_scale must be positive and finite")
    if law == "sqg-normal":
        return e2m1_rank_lut(
            bits, compander_scale=compander_scale, device=device
        )
    if law == "sqg-xor-cheb-t12":
        values = (
            sqg_xor_cheb_t12_e4m3_lut(bits)
            .view(torch.float8_e4m3fn)
            .float()
            * compander_scale
        )
    else:
        values = procedural_state_values(law).float() * compander_scale
    levels = E2M1_LEVELS
    magnitude = levels[(values.abs()[:, None] - levels[None, :]).abs().argmin(1)]
    quantized = torch.where(values < 0, -magnitude, magnitude)
    quantized[quantized == 0] = 0
    return quantized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous().to(
        device=device
    )


def validate_e2m1_lut(codebook_e4m3: torch.Tensor) -> torch.Tensor:
    """Validate and canonicalize a state-indexed exact-E2M1 byte LUT."""

    if codebook_e4m3.dtype != torch.uint8 or codebook_e4m3.numel() != TRANSITIONS:
        raise ValueError("custom codebook must contain 65,536 uint8 E4M3 labels")
    lut = codebook_e4m3.detach().reshape(TRANSITIONS).cpu().contiguous()
    decoded = lut.view(torch.float8_e4m3fn).float()
    legal_values = torch.cat((-E2M1_LEVELS[1:].flip(0), E2M1_LEVELS))
    legal = torch.isfinite(decoded) & (
        decoded[:, None] == legal_values[None, :]
    ).any(1)
    if not bool(legal.all()):
        bad = int((~legal).sum().item())
        raise ValueError(f"custom codebook contains {bad} non-E2M1 labels")
    return lut


def pack_trellis_edges(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the low branch bits in native EXL3 word order."""

    _validate_bits(bits)
    if indices.ndim < 1 or indices.shape[-1] != TILE_VALUES:
        raise ValueError("indices must end in 256 states")
    if indices.dtype == torch.bool or indices.is_floating_point():
        raise TypeError("indices must use an integer dtype")
    values = indices.to(torch.int64) & ((1 << bits) - 1)
    spans = values.reshape(*values.shape[:-1], TILE_CHANNELS, TILE_CHANNELS)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=values.device)
    stream = ((spans[..., None] >> symbol_shifts) & 1).reshape(
        *values.shape[:-1], TILE_CHANNELS, bits * TILE_CHANNELS
    )
    word_shifts = torch.arange(15, -1, -1, device=values.device)
    words = (
        stream.reshape(*values.shape[:-1], TILE_CHANNELS, bits, 16)
        << word_shifts
    ).sum(-1)
    flat = words.reshape(*values.shape[:-1], TILE_CHANNELS * bits)
    return flat.reshape(*flat.shape[:-1], -1, 2).flip(-1).reshape(flat.shape).to(
        torch.int16
    ).contiguous()


def unpack_trellis_edges(packed: torch.Tensor, bits: int) -> torch.Tensor:
    _validate_bits(bits)
    expected = TILE_CHANNELS * bits
    if packed.dtype != torch.int16 or packed.shape[-1] != expected:
        raise ValueError(f"packed trellis must end in {expected} int16 words")
    words = packed.to(torch.int64) & 0xFFFF
    words = words.reshape(*words.shape[:-1], -1, 2).flip(-1).reshape(words.shape)
    words = words.reshape(*words.shape[:-1], TILE_CHANNELS, bits)
    word_shifts = torch.arange(15, -1, -1, device=words.device)
    stream = ((words[..., None] >> word_shifts) & 1).reshape(
        *words.shape[:-2], TILE_CHANNELS, bits * TILE_CHANNELS
    )
    symbol_bits = stream.reshape(
        *words.shape[:-2], TILE_CHANNELS, TILE_CHANNELS, bits
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=words.device)
    return (symbol_bits << symbol_shifts).sum(-1).reshape(
        *words.shape[:-2], TILE_VALUES
    ).to(torch.int16).contiguous()


def reconstruct_trellis_states(edges: torch.Tensor, bits: int) -> torch.Tensor:
    _validate_bits(bits)
    if edges.shape[-1] != TILE_VALUES:
        raise ValueError("edge symbols must end in 256 values")
    values = edges.to(torch.int64) & ((1 << bits) - 1)
    states = torch.zeros_like(values)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(torch.int16).contiguous()


@lru_cache(maxsize=1)
def _encoder_module():
    root = Path(os.environ.get("GLM53_KQUANT_ROOT", DEFAULT_KQUANT_ROOT))
    if not (root / "kquant" / "sqg_quantizer.py").is_file():
        raise FileNotFoundError(f"KQuant SQG encoder not found at {root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from kquant.sqg_quantizer import install_sqg_quantizer

    module = types.SimpleNamespace(
        quantize_tiles=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("unexpected fallback to the procedural EXL3 quantizer")
        )
    )
    install_sqg_quantizer(module)
    return module


def _values_to_codes(values: torch.Tensor) -> torch.Tensor:
    levels = E2M1_LEVELS.to(values.device)
    distances = (values.abs()[..., None] - levels).abs()
    magnitudes = distances.argmin(-1).to(torch.uint8)
    if not bool((distances.amin(-1) == 0).all()):
        raise RuntimeError("trellis decoder emitted a non-E2M1 value")
    signs = ((values < 0) & (magnitudes != 0)).to(torch.uint8) << 3
    return magnitudes | signs


def _prepare_tiles(
    weight: torch.Tensor,
    global_scale: torch.Tensor | float,
    search_grid: int,
    block_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    rows, width = weight.shape
    out_tiles, input_tiles = rows // 16, width // 16
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=weight.device)
    blocks = weight.float().reshape(rows, input_tiles, 16)
    if block_scales is None:
        block_scales = _best_scales(blocks, gs, search_grid)
    else:
        if block_scales.shape != blocks.shape[:2]:
            raise ValueError(
                f"block scales {tuple(block_scales.shape)} do not match "
                f"{tuple(blocks.shape[:2])}"
            )
        block_scales = block_scales.to(
            device=weight.device, dtype=torch.float8_e4m3fn
        ).contiguous()
    real_scales = (block_scales.float() * gs).clamp_min(1e-30)
    normalized = blocks / real_scales[..., None]
    tiles = normalized.reshape(out_tiles, 16, input_tiles, 16).permute(
        2, 0, 3, 1
    ).contiguous().reshape(-1, TILE_VALUES)
    tiles = tiles.index_select(1, tensor_core_permutation(weight.device))
    return tiles, block_scales, out_tiles, input_tiles


def _quantized_tiles_to_blocks(
    quantized: torch.Tensor,
    *,
    rows: int,
    width: int,
) -> torch.Tensor:
    """Undo encoder tile order into logical ``[row, K/16, 16]`` blocks."""

    out_tiles, input_tiles = rows // 16, width // 16
    inverse = torch.argsort(tensor_core_permutation(quantized.device))
    logical = quantized.index_select(1, inverse)
    return logical.reshape(input_tiles, out_tiles, 16, 16).permute(
        1, 3, 0, 2
    ).contiguous().reshape(rows, input_tiles, 16)


def _prepare_native_group_tiles(normalized_group: torch.Tensor) -> torch.Tensor:
    """Map one native 16-column group to the frozen Tensor-Core lane order."""

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


def _states_to_normalized_blocks(
    states: torch.Tensor,
    lut: torch.Tensor,
    *,
    rows: int,
    width: int,
) -> torch.Tensor:
    """Decode native-order trellis states to exact normalized E2M1 blocks."""

    indices = (states.to(torch.int64) & 0xFFFF).flatten()
    values = lut.view(torch.float8_e4m3fn).float()
    quantized = values.index_select(0, indices).reshape(-1, TILE_VALUES)
    return _quantized_tiles_to_blocks(
        quantized, rows=rows, width=width
    )


@torch.no_grad()
def _encode_nvfp4_with_gptq_feedback(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    lut: torch.Tensor,
    *,
    bits: int,
    tailbite_context: int,
    percdamp: float,
    column_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode native NVFP4 groups with GPTQ-style inter-group feedback.

    Viterbi jointly fixes each physical 16-column group in native order.  The
    full Hessian then propagates that frozen group's error to later groups.
    This is GPTQ error feedback; it does not use an LDLQ factorization or loss.
    """

    rows, width = weight.shape
    if hessian.shape != (width, width):
        raise ValueError("Hessian shape must match weight width")
    if block_scales.shape != (rows, width // 16):
        raise ValueError("block scales must have one E4M3 value per NVFP4 group")
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
    real_scales = block_scales.float() * global_scale

    for slab_start in range(0, width, column_block):
        slab_end = min(slab_start + column_block, width)
        slab = work[:, slab_start:slab_end]
        errors = torch.zeros_like(slab)
        inverse_slab = hinv[slab_start:slab_end, slab_start:slab_end]
        for local_start in range(0, slab_end - slab_start, 16):
            absolute_start = slab_start + local_start
            local_order = permutation[absolute_start : absolute_start + 16] - absolute_start
            inverse_local = torch.argsort(local_order)
            current_permuted = slab[:, local_start : local_start + 16]
            current_native = current_permuted[:, inverse_local]
            scale = real_scales[:, absolute_start // 16]
            normalized_native = current_native / scale[:, None].clamp_min(1e-30)
            tiles = _prepare_native_group_tiles(normalized_native)
            quantized_tiles, states = _encode_tiles(
                tiles, lut, bits=bits, tailbite_context=tailbite_context
            )
            quantized_native = _decode_native_group_tiles(
                quantized_tiles, rows=rows
            ) * scale[:, None]
            quantized_permuted = quantized_native[:, local_order]
            states_native[absolute_start // 16] = states.reshape(out_tiles, TILE_VALUES)

            for column in range(16):
                slab_column = local_start + column
                current = slab[:, slab_column]
                quantized = quantized_permuted[:, column]
                diagonal = inverse_slab[slab_column, slab_column]
                error = (current - quantized) / diagonal
                slab[:, slab_column:] -= (
                    error[:, None] * inverse_slab[slab_column, slab_column:][None, :]
                )
                slab[:, slab_column] = quantized
                errors[:, slab_column] = error
        if slab_end < width:
            work[:, slab_end:] -= errors @ hinv[slab_start:slab_end, slab_end:]
    return work[:, inverse].contiguous(), states_native


@lru_cache(maxsize=1)
def _positive_e4m3_levels_cpu() -> torch.Tensor:
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    values = raw.view(torch.float8_e4m3fn).float()
    return torch.unique(values[torch.isfinite(values) & (values > 0)]).sort().values


def _nearest_positive_e4m3(values: torch.Tensor) -> torch.Tensor:
    levels = _positive_e4m3_levels_cpu().to(values.device)
    targets = values.float().clamp(min=float(levels[0]), max=float(levels[-1]))
    upper_index = torch.searchsorted(levels, targets).clamp(max=levels.numel() - 1)
    lower_index = (upper_index - 1).clamp(min=0)
    lower = levels[lower_index]
    upper = levels[upper_index]
    selected = torch.where(
        (targets - lower).abs() <= (upper - targets).abs(), lower, upper
    )
    return selected.to(torch.float8_e4m3fn).contiguous()


def _refit_fixed_code_block_scales(
    weight: torch.Tensor,
    normalized_blocks: torch.Tensor,
    global_scale: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """Exactly minimize block SSE over positive E4M3 scales for fixed codes."""

    source = weight.float().reshape_as(normalized_blocks)
    numerator = (source * normalized_blocks).sum(-1)
    denominator = normalized_blocks.square().sum(-1)
    target = numerator / denominator.clamp_min(1e-30) / global_scale
    fitted = _nearest_positive_e4m3(target)
    return torch.where(denominator > 0, fitted, previous).contiguous()


def _refit_fixed_code_global_scale(
    weights: list[torch.Tensor],
    normalized_blocks: list[torch.Tensor],
    block_scales: list[torch.Tensor],
    previous: torch.Tensor,
) -> torch.Tensor:
    """Least-squares FP32 global scale shared by a tensor family."""

    numerator = torch.zeros((), dtype=torch.float64, device=previous.device)
    denominator = torch.zeros_like(numerator)
    for weight, normalized, scale in zip(
        weights, normalized_blocks, block_scales, strict=True
    ):
        basis = normalized.double() * scale.float().double()[..., None]
        source = weight.double().reshape_as(basis)
        numerator += (source * basis).sum()
        denominator += basis.square().sum()
    fitted = (numerator / denominator.clamp_min(1e-30)).float()
    if not bool(torch.isfinite(fitted)) or float(fitted) <= 0:
        return previous
    return fitted.reshape(())


def _fixed_code_reconstruction_mse(
    weights: list[torch.Tensor],
    normalized_blocks: list[torch.Tensor],
    block_scales: list[torch.Tensor],
    global_scale: torch.Tensor,
) -> float:
    error = torch.zeros((), dtype=torch.float64, device=global_scale.device)
    values = 0
    for weight, normalized, scale in zip(
        weights, normalized_blocks, block_scales, strict=True
    ):
        reconstructed = normalized * scale.float()[..., None] * global_scale
        source = weight.float().reshape_as(reconstructed)
        error += (source.double() - reconstructed.double()).square().sum()
        values += source.numel()
    return float((error / values).item())


def _nearest_e2m1_values(values: torch.Tensor) -> torch.Tensor:
    levels = E2M1_LEVELS.to(values.device)
    mids = (levels[1:] + levels[:-1]) / 2
    magnitude = levels[torch.bucketize(values.abs(), mids)]
    return torch.where(values < 0, -magnitude, magnitude)


@torch.no_grad()
def refine_rtn_nvfp4_group(
    weights: list[torch.Tensor],
    *,
    global_scale: torch.Tensor | float,
    search_grid: int = 12,
    scale_refinement_iterations: int = 2,
) -> list[PackedNVFP4]:
    """Matched scalar-E2M1 control with the same exact scale coordinate updates."""

    if not weights or scale_refinement_iterations < 0:
        raise ValueError("weights are required and refinement must be nonnegative")
    device = weights[0].device
    if any(weight.device != device for weight in weights):
        raise ValueError("matched RTN weights must share one device")
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=device).reshape(())
    scales = [
        _best_scales(
            weight.float().reshape(weight.shape[0], weight.shape[1] // 16, 16),
            gs,
            search_grid,
        )
        for weight in weights
    ]
    for _ in range(scale_refinement_iterations):
        normalized = [
            _nearest_e2m1_values(
                weight.float().reshape(weight.shape[0], weight.shape[1] // 16, 16)
                / (scale.float()[..., None] * gs).clamp_min(1e-30)
            )
            for weight, scale in zip(weights, scales, strict=True)
        ]
        scales = [
            _refit_fixed_code_block_scales(weight, values, gs, previous)
            for weight, values, previous in zip(weights, normalized, scales, strict=True)
        ]
        gs = _refit_fixed_code_global_scale(weights, normalized, scales, gs)
        scales = [
            _refit_fixed_code_block_scales(weight, values, gs, previous)
            for weight, values, previous in zip(weights, normalized, scales, strict=True)
        ]
    endpoints = []
    for weight, scale in zip(weights, scales, strict=True):
        normalized_source = (
            weight.float().reshape(weight.shape[0], weight.shape[1] // 16, 16)
            / (scale.float()[..., None] * gs).clamp_min(1e-30)
        )
        normalized = _nearest_e2m1_values(normalized_source)
        endpoints.append(
            PackedNVFP4(
                weight=pack_codes(
                    _values_to_codes(normalized).reshape_as(weight)
                ).cpu(),
                weight_scale=scale.cpu(),
                weight_scale_2=gs.cpu(),
            )
        )
    return endpoints


def _encode_tiles(
    tiles: torch.Tensor,
    lut: torch.Tensor,
    *,
    bits: int,
    tailbite_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _encoder_module().quantize_tiles(
        tiles,
        {
            "K": bits,
            "devices": [str(tiles.device)],
            "sqg_e4m3_lut": lut,
            "tailbite_context": tailbite_context,
        },
    )


@torch.no_grad()
def optimize_e2m1_codebook(
    weights_and_scales: list[tuple[torch.Tensor, torch.Tensor | float]],
    *,
    bits: int = 3,
    initial_law: str = "mcg",
    initial_compander_scale: float = 1.0,
    iterations: int = 4,
    search_grid: int = 12,
    tailbite_context: int = 128,
) -> tuple[torch.Tensor, list[CodebookFitStep]]:
    """Fit one global state->E2M1 table by discrete Viterbi/Lloyd steps.

    This is codebook learning only. It does not perform LDLQ, rotate weights,
    or learn per-tensor residuals. Each M-step chooses the legal E2M1 value
    minimizing scalar squared error for all samples assigned to a state.
    """

    _validate_bits(bits)
    if not weights_and_scales or iterations <= 0:
        raise ValueError("at least one weight and one positive iteration are required")
    devices = {str(weight.device) for weight, _scale in weights_and_scales}
    if len(devices) != 1 or next(iter(devices)).split(":")[0] != "cuda":
        raise ValueError("codebook fitting requires weights on one CUDA device")
    device = weights_and_scales[0][0].device
    prepared = [
        _prepare_tiles(weight, scale, search_grid)[0]
        for weight, scale in weights_and_scales
    ]
    lut = e2m1_state_lut(
        bits,
        law=initial_law,
        compander_scale=initial_compander_scale,
        device=device,
    )
    levels = torch.cat(
        (-E2M1_LEVELS[1:].flip(0), E2M1_LEVELS)
    ).to(device)
    history: list[CodebookFitStep] = []
    for iteration in range(iterations):
        sums = torch.zeros(TRANSITIONS, dtype=torch.float64, device=device)
        counts = torch.zeros(TRANSITIONS, dtype=torch.int64, device=device)
        squared_error = torch.zeros((), dtype=torch.float64, device=device)
        values_seen = 0
        for tiles in prepared:
            quantized, states = _encode_tiles(
                tiles,
                lut,
                bits=bits,
                tailbite_context=tailbite_context,
            )
            indices = (states.to(torch.int64) & 0xFFFF).flatten()
            source = tiles.flatten().double()
            sums.scatter_add_(0, indices, source)
            counts.scatter_add_(0, indices, torch.ones_like(indices))
            squared_error += (quantized.double() - tiles.double()).square().sum()
            values_seen += tiles.numel()
        visited = counts > 0
        means = torch.zeros(TRANSITIONS, dtype=torch.float32, device=device)
        means[visited] = (sums[visited] / counts[visited]).float()
        selected = (means[:, None] - levels[None, :]).square().argmin(1)
        updated_values = levels[selected]
        updated_values[~visited] = lut.view(torch.float8_e4m3fn).float()[~visited]
        updated_values[updated_values == 0] = 0
        updated = updated_values.to(torch.float8_e4m3fn).view(torch.uint8)
        changed = int((updated != lut).sum().item())
        history.append(
            CodebookFitStep(
                iteration=iteration,
                assigned_mse=float((squared_error / values_seen).item()),
                visited_states=int(visited.sum().item()),
                changed_labels=changed,
            )
        )
        lut = updated.contiguous()
    return lut.cpu(), history


@torch.no_grad()
def quantize_trellis_nvfp4_group(
    weights: list[torch.Tensor],
    *,
    global_scale: torch.Tensor | float,
    block_scales: list[torch.Tensor] | None = None,
    bits: int = 3,
    codebook_law: str = "mcg",
    codebook_e4m3: torch.Tensor | None = None,
    compander_scale: float = 1.0,
    search_grid: int = 12,
    tailbite_context: int = 128,
    scale_refinement_iterations: int = 2,
    refine_global_scale: bool = True,
) -> list[TrellisNVFP4]:
    """Encode tensors sharing one FP32 scale with codebook-specific refitting."""

    _validate_bits(bits)
    if not weights or scale_refinement_iterations < 0:
        raise ValueError("weights are required and scale refinement must be nonnegative")
    device = weights[0].device
    for weight in weights:
        if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
            raise ValueError("trellis NVFP4 requires 16x16-aligned 2D weights")
        if weight.device != device or weight.device.type != "cuda":
            raise ValueError("all weights must be on one CUDA device")
    if block_scales is not None and len(block_scales) != len(weights):
        raise ValueError("block scale list must match the weight list")
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=device).reshape(())
    if codebook_e4m3 is None:
        lut = e2m1_state_lut(
            bits,
            law=codebook_law,
            compander_scale=compander_scale,
            device=device,
        )
    else:
        lut = validate_e2m1_lut(codebook_e4m3).to(device=device)
        codebook_law = "learned-e2m1"

    scales: list[torch.Tensor] = []
    for index, weight in enumerate(weights):
        supplied = None if block_scales is None else block_scales[index]
        scales.append(_prepare_tiles(weight, gs, search_grid, supplied)[1])

    initial_mse: float | None = None
    for _ in range(scale_refinement_iterations):
        normalized: list[torch.Tensor] = []
        for weight, scale in zip(weights, scales, strict=True):
            tiles = _prepare_tiles(weight, gs, search_grid, scale)[0]
            quantized, _states = _encode_tiles(
                tiles, lut, bits=bits, tailbite_context=tailbite_context
            )
            normalized.append(
                _quantized_tiles_to_blocks(
                    quantized, rows=weight.shape[0], width=weight.shape[1]
                )
            )
        if initial_mse is None:
            initial_mse = _fixed_code_reconstruction_mse(
                weights, normalized, scales, gs
            )
        scales = [
            _refit_fixed_code_block_scales(weight, values, gs, previous)
            for weight, values, previous in zip(
                weights, normalized, scales, strict=True
            )
        ]
        if refine_global_scale:
            gs = _refit_fixed_code_global_scale(weights, normalized, scales, gs)
        scales = [
            _refit_fixed_code_block_scales(weight, values, gs, previous)
            for weight, values, previous in zip(
                weights, normalized, scales, strict=True
            )
        ]

    payloads: list[TrellisNVFP4] = []
    final_normalized: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for weight, scale in zip(weights, scales, strict=True):
        rows, width = weight.shape
        out_tiles, input_tiles = rows // 16, width // 16
        tiles = _prepare_tiles(weight, gs, search_grid, scale)[0]
        quantized, states = _encode_tiles(
            tiles, lut, bits=bits, tailbite_context=tailbite_context
        )
        normalized = _quantized_tiles_to_blocks(
            quantized, rows=rows, width=width
        )
        final_normalized.append(normalized)
        final_states.append(states.reshape(input_tiles, out_tiles, TILE_VALUES))
    final_mse = _fixed_code_reconstruction_mse(
        weights, final_normalized, scales, gs
    )
    if initial_mse is None:
        initial_mse = final_mse
    for weight, scale, normalized, states in zip(
        weights, scales, final_normalized, final_states, strict=True
    ):
        rows, width = weight.shape
        codes = _values_to_codes(normalized).reshape(rows, width)
        endpoint = PackedNVFP4(
            weight=pack_codes(codes).cpu(),
            weight_scale=scale.cpu(),
            weight_scale_2=gs.cpu(),
        )
        payloads.append(
            TrellisNVFP4(
                endpoint=endpoint,
                trellis=pack_trellis_edges(states, bits).cpu(),
                bits=bits,
                codebook_e4m3=lut.cpu(),
                codebook_law=codebook_law,
                scale_refinement_iterations=scale_refinement_iterations,
                initial_reconstruction_mse=initial_mse,
                final_reconstruction_mse=final_mse,
            )
        )
    return payloads


@torch.no_grad()
def quantize_trellis_nvfp4(
    weight: torch.Tensor,
    *,
    global_scale: torch.Tensor | float,
    block_scales: torch.Tensor | None = None,
    bits: int = 3,
    codebook_law: str = "mcg",
    codebook_e4m3: torch.Tensor | None = None,
    compander_scale: float = 1.0,
    search_grid: int = 12,
    tailbite_context: int = 128,
    scale_refinement_iterations: int = 2,
    refine_global_scale: bool = True,
) -> TrellisNVFP4:
    """Encode one matrix and materialize its exact native-NVFP4 endpoint."""

    return quantize_trellis_nvfp4_group(
        [weight],
        global_scale=global_scale,
        block_scales=None if block_scales is None else [block_scales],
        bits=bits,
        codebook_law=codebook_law,
        codebook_e4m3=codebook_e4m3,
        compander_scale=compander_scale,
        search_grid=search_grid,
        tailbite_context=tailbite_context,
        scale_refinement_iterations=scale_refinement_iterations,
        refine_global_scale=refine_global_scale,
    )[0]


@torch.no_grad()
def quantize_trellis_nvfp4_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    global_scale: torch.Tensor | float,
    block_scales: torch.Tensor | None = None,
    bits: int = 4,
    codebook_law: str = "mcg",
    codebook_e4m3: torch.Tensor | None = None,
    compander_scale: float = 1.0,
    search_grid: int = 12,
    tailbite_context: int = 128,
    scale_refinement_iterations: int = 2,
    refine_global_scale: bool = True,
    percdamp: float = 0.01,
    column_block: int = 128,
) -> TrellisNVFP4:
    """Output-aware K4/K3 Viterbi whose decoded endpoint is exact NVFP4.

    The encoder makes one joint Viterbi decision per physical 16x16 tile and
    carries its error through the full calibration Hessian using GPTQ-style
    feedback.  It iteratively refits the E4M3/16 and optional tensor-global
    scales.  The result decodes only to E2M1 nibbles and therefore targets the
    native ``mxf4nvf4`` P4 prologue.  No LDLQ algorithm is used.
    """

    _validate_bits(bits)
    if weight.ndim != 2 or weight.device.type != "cuda":
        raise ValueError("trellis NVFP4 requires a CUDA 2D tensor")
    if weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("trellis NVFP4 requires 16x16-aligned weights")
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Hessian shape must match weight width")
    if scale_refinement_iterations < 0:
        raise ValueError("scale refinement must be nonnegative")
    device = weight.device
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=device).reshape(())
    if codebook_e4m3 is None:
        lut = e2m1_state_lut(
            bits, law=codebook_law, compander_scale=compander_scale, device=device
        )
    else:
        lut = validate_e2m1_lut(codebook_e4m3).to(device=device)
        codebook_law = "learned-e2m1"
    scales = _prepare_tiles(weight, gs, search_grid, block_scales)[1]
    reconstruction = states = None
    initial_mse = None
    for iteration in range(scale_refinement_iterations + 1):
        reconstruction, states = _encode_nvfp4_with_gptq_feedback(
            weight,
            hessian.to(device),
            scales,
            gs,
            lut,
            bits=bits,
            tailbite_context=tailbite_context,
            percdamp=percdamp,
            column_block=column_block,
        )
        normalized = _states_to_normalized_blocks(
            states, lut, rows=weight.shape[0], width=weight.shape[1]
        )
        current_mse = _fixed_code_reconstruction_mse(
            [weight], [normalized], [scales], gs
        )
        if initial_mse is None:
            initial_mse = current_mse
        if iteration < scale_refinement_iterations:
            scales = _refit_fixed_code_block_scales(weight, normalized, gs, scales)
            if refine_global_scale:
                gs = _refit_fixed_code_global_scale([weight], [normalized], [scales], gs)
            scales = _refit_fixed_code_block_scales(weight, normalized, gs, scales)
    assert reconstruction is not None and states is not None and initial_mse is not None
    normalized = _states_to_normalized_blocks(
        states, lut, rows=weight.shape[0], width=weight.shape[1]
    )
    endpoint = PackedNVFP4(
        weight=pack_codes(_values_to_codes(normalized).reshape_as(weight)).cpu(),
        weight_scale=scales.cpu(),
        weight_scale_2=gs.cpu(),
    )
    payload = TrellisNVFP4(
        endpoint=endpoint,
        trellis=pack_trellis_edges(states, bits).cpu(),
        bits=bits,
        codebook_e4m3=lut.cpu(),
        codebook_law=codebook_law,
        scale_refinement_iterations=scale_refinement_iterations,
        initial_reconstruction_mse=initial_mse,
        final_reconstruction_mse=_fixed_code_reconstruction_mse(
            [weight], [normalized], [scales], gs
        ),
    )
    decoded = decode_trellis_endpoint(payload)
    if not torch.equal(decoded.weight, endpoint.weight):
        raise RuntimeError("P4 trellis stream does not bit-exactly decode to its endpoint")
    return payload


def pack_two_bit_selectors(selectors: torch.Tensor) -> torch.Tensor:
    if selectors.dtype == torch.bool or selectors.is_floating_point():
        raise TypeError("selectors must use an integer dtype")
    flat = selectors.to(torch.uint8).flatten()
    if bool((flat > 3).any()):
        raise ValueError("two-bit selectors must be in [0, 3]")
    padding = (-flat.numel()) % 4
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)))
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=flat.device)
    return (flat.reshape(-1, 4) << shifts).sum(-1).to(torch.uint8).contiguous()


def unpack_two_bit_selectors(
    packed: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed selectors must be a one-dimensional uint8 tensor")
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    values = ((packed[:, None] >> shifts) & 3).flatten()[: math.prod(shape)]
    return values.reshape(shape).contiguous()


@torch.no_grad()
def hybridize_trellis_tiles(
    weight: torch.Tensor,
    codecs: list[TrellisNVFP4],
    inputs: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    output_tile_chunk: int = 8,
    selector_sweeps: int = 2,
    selector_phases: int = 16,
    selector_isolated_regularization: float = 0.0,
) -> tuple[HybridTrellisNVFP4, dict[str, object]]:
    """Select SQG/MCG/MUL1 per 16x16 tile by routed output error.

    The selector is an offline calibration decision. The resulting payload
    still contains one K4 edge stream for every coefficient and decodes to a
    single exact E2M1 operand.
    """

    if not 2 <= len(codecs) <= 4:
        raise ValueError("the hybrid requires two through four candidate laws")
    if any(codec.bits != 4 for codec in codecs):
        raise ValueError("the native-NVFP4 hybrid is K4 only")
    if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("hybrid weights must be 16x16 aligned")
    if inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("hybrid calibration inputs do not match the weight")
    if route_weights.shape != (inputs.shape[0],):
        raise ValueError("route weights must contain one value per input row")
    if (
        output_tile_chunk <= 0
        or selector_sweeps < 0
        or selector_phases <= 0
        or selector_isolated_regularization < 0
    ):
        raise ValueError("hybrid chunk/phases must be positive and sweeps nonnegative")
    reference = codecs[0].endpoint
    if any(codec.endpoint.weight.shape != reference.weight.shape for codec in codecs):
        raise ValueError("candidate endpoint shapes differ")
    global_bytes = reference.weight_scale_2.numpy().tobytes()
    if any(codec.endpoint.weight_scale_2.numpy().tobytes() != global_bytes for codec in codecs[1:]):
        raise ValueError("tile-hybrid candidates must share the exact FP32 global scale")

    device = weight.device
    rows, width = weight.shape
    out_tiles, input_tiles = rows // 16, width // 16
    source_inputs = inputs.float().reshape(inputs.shape[0], input_tiles, 16)
    sample_weights = route_weights.float().square()
    sample_weights /= sample_weights.sum().clamp_min(1e-30)
    errors = torch.stack(
        [dequantize(codec.endpoint).to(device).float() - weight.float() for codec in codecs]
    ).reshape(len(codecs), out_tiles, 16, input_tiles, 16)
    selected_out_in = torch.empty(
        (out_tiles, input_tiles), dtype=torch.uint8, device="cpu"
    )
    initial_objective = 0.0
    final_objective = 0.0
    for start in range(0, out_tiles, output_tile_chunk):
        stop = min(start + output_tile_chunk, out_tiles)
        projected = torch.einsum(
            "sbi,corbi->scobr", source_inputs, errors[:, start:stop]
        )
        scores = (
            projected.double().square()
            * sample_weights.double()[:, None, None, None, None]
        ).sum(dim=(0, 4))
        selected = scores.argmin(0)
        gather_index = selected[None, None, :, :, None].expand(
            projected.shape[0], 1, projected.shape[2], projected.shape[3], projected.shape[4]
        )
        current = projected.gather(1, gather_index).squeeze(1)
        total = current.sum(2)
        chunk_initial = (
            total.double().square()
            * sample_weights.double()[:, None, None]
        ).sum()
        selected_isolated = scores.gather(0, selected[None]).squeeze(0)
        current_penalty = selected_isolated.sum()
        current_objective = (
            chunk_initial + selector_isolated_regularization * current_penalty
        )
        # Compare like with like: both the initial and final objectives include
        # the same isolated-error regularizer.
        initial_objective += float(current_objective.item())
        for _sweep in range(selector_sweeps):
            changed = False
            for phase in range(selector_phases):
                columns = torch.arange(
                    phase, input_tiles, selector_phases, device=device
                )
                if columns.numel() == 0:
                    continue
                old = current.index_select(2, columns)
                residual = total[:, :, None, :] - old
                options = residual[:, None] + projected.index_select(3, columns)
                option_scores = (
                    options.double().square()
                    * sample_weights.double()[:, None, None, None, None]
                ).sum(dim=(0, 4))
                isolated_options = scores.index_select(2, columns)
                choice = (
                    option_scores
                    + selector_isolated_regularization * isolated_options
                ).argmin(0)
                option_index = choice[None, None, :, :, None].expand(
                    options.shape[0], 1, options.shape[2], options.shape[3], options.shape[4]
                )
                replacement = options.gather(1, option_index).squeeze(1) - residual
                trial_total = total - old.sum(2) + replacement.sum(2)
                trial_joint_objective = (
                    trial_total.double().square()
                    * sample_weights.double()[:, None, None]
                ).sum()
                old_penalty = selected_isolated.index_select(1, columns).sum()
                new_penalty = isolated_options.gather(
                    0, choice[None]
                ).squeeze(0).sum()
                trial_penalty = current_penalty - old_penalty + new_penalty
                trial_objective = (
                    trial_joint_objective
                    + selector_isolated_regularization * trial_penalty
                )
                if bool(trial_objective <= current_objective):
                    changed = changed or bool(
                        (choice != selected.index_select(1, columns)).any()
                    )
                    selected[:, columns] = choice
                    current[:, :, columns] = replacement
                    selected_isolated[:, columns] = isolated_options.gather(
                        0, choice[None]
                    ).squeeze(0)
                    total = trial_total
                    current_penalty = trial_penalty
                    current_objective = trial_objective
            if not changed:
                break
        final_objective += float(current_objective.item())
        selected_out_in[start:stop] = selected.to(torch.uint8).cpu()
    selected_in_out = selected_out_in.transpose(0, 1).contiguous()

    packed_candidates = torch.stack(
        [codec.endpoint.weight for codec in codecs]
    ).reshape(len(codecs), out_tiles, 16, input_tiles, 8)
    scale_candidates = torch.stack(
        [codec.endpoint.weight_scale for codec in codecs]
    ).reshape(len(codecs), out_tiles, 16, input_tiles)
    trellis_candidates = torch.stack([codec.trellis for codec in codecs])
    packed_weight = torch.zeros_like(packed_candidates[0])
    block_scale = torch.zeros_like(scale_candidates[0])
    trellis = torch.zeros_like(trellis_candidates[0])
    for candidate in range(len(codecs)):
        mask = selected_out_in == candidate
        packed_weight = torch.where(
            mask[:, None, :, None], packed_candidates[candidate], packed_weight
        )
        block_scale = torch.where(
            mask[:, None, :], scale_candidates[candidate], block_scale
        )
        trellis = torch.where(
            selected_in_out[:, :, None] == candidate,
            trellis_candidates[candidate],
            trellis,
        )
    endpoint = PackedNVFP4(
        weight=packed_weight.reshape(rows, width // 2).contiguous(),
        weight_scale=block_scale.reshape(rows, input_tiles).contiguous(),
        weight_scale_2=reference.weight_scale_2.clone(),
    )
    payload = HybridTrellisNVFP4(
        endpoint=endpoint,
        trellis=trellis.contiguous(),
        selectors=pack_two_bit_selectors(selected_in_out).cpu(),
        selector_shape=(input_tiles, out_tiles),
        bits=4,
        codebooks_e4m3=torch.stack(
            [codec.codebook_e4m3 for codec in codecs]
        ).contiguous(),
        codebook_laws=tuple(codec.codebook_law for codec in codecs),
    )
    counts = torch.bincount(
        selected_out_in.flatten().long(), minlength=len(codecs)
    ).cpu()
    return payload, {
        "tile_counts": {
            law: int(count) for law, count in zip(payload.codebook_laws, counts, strict=True)
        },
        "selector_bpw": payload.selector_bpw,
        "stored_bpw_excluding_shared_luts_and_global_scalar": (
            payload.stored_bpw_excluding_shared_luts_and_global_scalar
        ),
        "selection_objective": "routed activation-weighted joint projection-output SSE with monotone phased block-coordinate tile-law selection",
        "selector_sweeps": selector_sweeps,
        "selector_phases": selector_phases,
        "selector_isolated_regularization": selector_isolated_regularization,
        "local_initial_objective": initial_objective,
        "joint_final_objective": final_objective,
        "joint_objective_reduction_percent": (
            (initial_objective - final_objective) / max(initial_objective, 1e-30) * 100
        ),
    }


@torch.no_grad()
def decode_hybrid_trellis_endpoint(payload: HybridTrellisNVFP4) -> PackedNVFP4:
    """Reference-decode the selected law per tile to one packed E2M1 endpoint."""

    input_tiles, out_tiles = payload.selector_shape
    selectors = unpack_two_bit_selectors(payload.selectors, payload.selector_shape)
    edges = unpack_trellis_edges(payload.trellis, payload.bits)
    states = reconstruct_trellis_states(edges, payload.bits)
    state_indices = (states.to(torch.int64) & 0xFFFF).long()
    codebooks = payload.codebooks_e4m3.view(torch.float8_e4m3fn).float()
    decoded = torch.zeros_like(states, dtype=torch.float32)
    for candidate in range(codebooks.shape[0]):
        values = codebooks[candidate].index_select(
            0, state_indices.flatten()
        ).reshape_as(states)
        decoded = torch.where(selectors[:, :, None] == candidate, values, decoded)
    inverse = torch.argsort(tensor_core_permutation())
    decoded = decoded.index_select(-1, inverse)
    blocks = decoded.reshape(input_tiles, out_tiles, 16, 16).permute(
        1, 3, 0, 2
    ).contiguous().reshape(out_tiles * 16, input_tiles, 16)
    return PackedNVFP4(
        weight=pack_codes(
            _values_to_codes(blocks).reshape(out_tiles * 16, input_tiles * 16)
        ),
        weight_scale=payload.endpoint.weight_scale.clone(),
        weight_scale_2=payload.endpoint.weight_scale_2.clone(),
    )


@torch.no_grad()
def decode_trellis_endpoint(
    payload: TrellisNVFP4,
) -> PackedNVFP4:
    """Reference-decode a trellis payload directly to packed E2M1 nibbles."""

    endpoint = payload.endpoint
    rows = endpoint.weight.shape[0]
    width = endpoint.weight.shape[1] * 2
    out_tiles, input_tiles = rows // 16, width // 16
    edges = unpack_trellis_edges(payload.trellis, payload.bits)
    states = reconstruct_trellis_states(edges, payload.bits)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = payload.codebook_e4m3.view(torch.float8_e4m3fn).float()
    decoded = values.index_select(0, indices.flatten()).reshape_as(states)
    inverse = torch.argsort(tensor_core_permutation())
    decoded = decoded.index_select(-1, inverse)
    blocks = decoded.reshape(input_tiles, out_tiles, 16, 16).permute(
        1, 3, 0, 2
    ).contiguous().reshape(rows, input_tiles, 16)
    return PackedNVFP4(
        weight=pack_codes(_values_to_codes(blocks).reshape(rows, width)),
        weight_scale=endpoint.weight_scale.clone(),
        weight_scale_2=endpoint.weight_scale_2.clone(),
    )
