"""CPU reference for the GLM-5.3 B12X FP8 NoPE KV-cache ABI.

This module deliberately contains no runtime dispatch.  It is a small,
executable contract for the record written and consumed by the B12X
``ModelType.GLM_NEXT`` path:

    [0, 512)   four consecutive groups of 128 E4M3 values
    [512, 528) four little-endian FP32 scales (group amax / 448)

GLM-5.3 has no decoupled RoPE cache lane (``qk_rope_head_dim == 0``), so a
656-byte DeepSeek-style record is not ABI-compatible.  Zero groups use a
scale of 1.0, matching the device writer and avoiding a zero reciprocal.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import torch


LATENT_DIM = 512
GROUP_SIZE = 128
NUM_GROUPS = LATENT_DIM // GROUP_SIZE
DATA_BYTES = LATENT_DIM
SCALE_OFFSET = DATA_BYTES
SCALE_BYTES = NUM_GROUPS * 4
RECORD_BYTES = DATA_BYTES + SCALE_BYTES
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN = float(torch.finfo(FP8_DTYPE).min)


@dataclass(frozen=True)
class Fp8NopeGeometry:
    """The semantic record geometry selected for GLM-5.3 NoPE MLA."""

    latent_dim: int = LATENT_DIM
    rope_dim: int = 0
    value_dim: int = LATENT_DIM
    group_size: int = GROUP_SIZE
    num_groups: int = NUM_GROUPS
    data_bytes: int = DATA_BYTES
    scale_offset: int = SCALE_OFFSET
    scale_bytes: int = SCALE_BYTES
    record_bytes: int = RECORD_BYTES
    scale_dtype: torch.dtype = torch.float32
    value_dtype: torch.dtype = FP8_DTYPE


GLM53_FP8_NOPE_GEOMETRY = Fp8NopeGeometry()


def select_fp8_nope_geometry(
    *,
    q_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    kv_cache_dtype: str,
) -> Fp8NopeGeometry:
    """Select the ABI only for the exact GLM-5.3 FP8-NoPE contract.

    The width alone is ambiguous with other MLA model families, so all four
    semantic inputs are required and every mismatch fails closed.
    """

    got = (
        int(q_head_dim),
        int(qk_rope_head_dim),
        int(kv_lora_rank),
        str(kv_cache_dtype),
    )
    expected = (LATENT_DIM, 0, LATENT_DIM, "fp8_ds_mla")
    if got != expected:
        raise ValueError(
            "GLM-5.3 FP8 NoPE requires "
            "q_head_dim=512, qk_rope_head_dim=0, kv_lora_rank=512, "
            f"kv_cache_dtype='fp8_ds_mla'; got {got!r}"
        )
    return GLM53_FP8_NOPE_GEOMETRY


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("the B12X FP8 NoPE record stores little-endian FP32 scales")


def _as_latent_rows(latent: torch.Tensor) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent must be a torch.Tensor")
    if latent.device.type != "cpu":
        raise ValueError("the ABI reference accepts CPU tensors only")
    if latent.ndim != 2 or int(latent.shape[1]) != LATENT_DIM:
        raise ValueError(f"latent must have shape (rows, {LATENT_DIM})")
    if latent.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("latent must have dtype torch.bfloat16 or torch.float32")
    latent_f32 = latent.contiguous().to(torch.float32)
    if not bool(torch.isfinite(latent_f32).all()):
        raise ValueError("latent must contain only finite values")
    return latent_f32


def encode_fp8_nope(latent: torch.Tensor) -> torch.Tensor:
    """Encode rows to bit-exact 528-byte GLM-5.3 FP8-NoPE records."""

    _require_little_endian()
    latent_f32 = _as_latent_rows(latent)
    rows = int(latent_f32.shape[0])
    groups = latent_f32.reshape(rows, NUM_GROUPS, GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    scales = amax / FP8_MAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = (groups / scales.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    values = normalized.to(FP8_DTYPE).reshape(rows, DATA_BYTES)
    value_bytes = values.view(torch.uint8)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(rows, SCALE_BYTES)
    return torch.cat((value_bytes, scale_bytes), dim=1).contiguous()


def _as_record_rows(records: torch.Tensor) -> torch.Tensor:
    if not isinstance(records, torch.Tensor):
        raise TypeError("records must be a torch.Tensor")
    if records.device.type != "cpu":
        raise ValueError("the ABI reference accepts CPU tensors only")
    if records.dtype != torch.uint8:
        raise TypeError("records must have dtype torch.uint8")
    if records.ndim != 2 or int(records.shape[1]) != RECORD_BYTES:
        raise ValueError(f"records must have shape (rows, {RECORD_BYTES})")
    return records.contiguous()


def decode_fp8_nope(records: torch.Tensor) -> torch.Tensor:
    """Decode 528-byte records to FP32 latent rows."""

    _require_little_endian()
    packed = _as_record_rows(records)
    rows = int(packed.shape[0])
    values = packed[:, :DATA_BYTES].contiguous().view(FP8_DTYPE)
    values = values.reshape(rows, NUM_GROUPS, GROUP_SIZE).to(torch.float32)
    scales = packed[:, SCALE_OFFSET:RECORD_BYTES].contiguous().view(torch.float32)
    scales = scales.reshape(rows, NUM_GROUPS)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("record scales must be finite and strictly positive")
    return (values * scales.unsqueeze(-1)).reshape(rows, LATENT_DIM).contiguous()


def cache_byte_offset(
    slot: int,
    *,
    page_size: int,
    page_stride_bytes: int,
    record_bytes: int = RECORD_BYTES,
) -> int:
    """Host mirror of the device writer's Int64 slot address calculation."""

    if slot < 0:
        raise ValueError("slot must be non-negative")
    if page_size <= 0 or page_stride_bytes <= 0 or record_bytes <= 0:
        raise ValueError("page_size and byte strides must be positive")
    semantic_page_bytes = int(page_size) * int(record_bytes)
    if int(page_stride_bytes) < semantic_page_bytes:
        raise ValueError("page_stride_bytes is smaller than one semantic page")
    page, row = divmod(int(slot), int(page_size))
    return page * int(page_stride_bytes) + row * int(record_bytes)
