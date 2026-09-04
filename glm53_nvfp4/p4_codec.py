"""CPU-only, table-free K4 MCG -> native NVFP4 operand reference.

This is a versioned matrix interchange, not a GPU runtime or a weight encoder.
See docs/P4_CODEC.md for the byte contract and compatibility limits. ExLlamaV3
owns the procedural constants and stream/lane layout; KQuant/QSRT and the
vendored w4a8_trellis port retain the notices in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


SCHEMA = "glm53-p4-mcg-matrix.v1"
CONTRACT = {
    "schema": SCHEMA,
    "role": "physical-codec",
    "bits": "4",
    "state_bits": "16",
    "tile_values": "256",
    "state_boundary": "cyclic-per-tile",
    "law": "procedural-mcg-alpha1-rne-e2m1",
    "mcg_arithmetic": "u32-wrap-mask-xor-add-rn-f16",
    "alphabet": "e2m1",
    "rounding": "nearest-even-satfinite",
    "signed_zero": "preserve",
    "byte_order": "little",
    "trellis_layout": "k16-n16-exl3-lane-pair-swapped-i16",
    "weight_layout": "row-major-n-k-low-nibble-first",
    "scale": "positive-e4m3fn-k16",
    "scale_layout": "row-major-n-k16",
    "global_scale": "positive-f32-scalar-multiply",
    "boundary": "identity",
    "state_lut_bytes": "0",
    "ldlq": "false",
}
SCALE_LAYOUT = CONTRACT["scale_layout"]
SWIZZLED_SCALE_LAYOUT = "modelopt-128x4"
# Only small numeric alphabets, never a 2**16 state-indexed lookup table.
E2M1_MAGNITUDES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_MIDPOINTS = (E2M1_MAGNITUDES[:-1] + E2M1_MAGNITUDES[1:]) / 2


@dataclass(frozen=True)
class P4Payload:
    rows: int
    width: int
    trellis: np.ndarray       # little-endian int16 [K/16, N/16, 64]
    scale_e4m3: np.ndarray    # raw positive E4M3 bytes [N, K/16]
    global_scale: np.ndarray  # little-endian float32 scalar, always charged


@dataclass(frozen=True)
class P4Operands:
    weight: np.ndarray        # uint8 [N, K/2], even K in low nibble
    scale_e4m3: np.ndarray    # raw physical bytes in logical [N, K/16] order
    global_scale: np.ndarray  # copied bit-exactly, not folded into scale bytes


def _shape(rows: int, width: int) -> None:
    if any(type(x) is not int or x <= 0 or x % 16 for x in (rows, width)):
        raise ValueError("shape must be positive integer [N, K], both divisible by 16")


def _array(value: np.ndarray, dtype: str, name: str, shape=None) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} requires {dtype} dtype and little endianness")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} requires a contiguous layout")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != {shape}")
    return value


def _edges(edges: np.ndarray) -> np.ndarray:
    _array(edges, "u1", "edges")
    if edges.ndim < 1 or edges.shape[-1] != 256 or edges.size == 0:
        raise ValueError("edges must end in 256 symbols with no empty dimensions")
    if np.any(edges > 15):
        raise ValueError("K4 edge symbols must be in [0, 15]")
    return edges


def pack_k4_edges(edges: np.ndarray) -> np.ndarray:
    """Pack branch symbols, without silently truncating purported full states."""
    _edges(edges)
    octets = edges.reshape(*edges.shape[:-1], 32, 8).astype(np.uint32)
    words = np.bitwise_or.reduce(octets << np.arange(28, -1, -4, dtype=np.uint32), axis=-1)
    return words.astype("<u4").view("<i2").reshape(*edges.shape[:-1], 64)


def unpack_k4_edges(trellis: np.ndarray) -> np.ndarray:
    _array(trellis, "<i2", "trellis")
    if trellis.ndim < 1 or trellis.shape[-1] != 64 or trellis.size == 0:
        raise ValueError("K4 trellis must end in 64 int16 words with no empty dimensions")
    words = trellis.view("<u4")
    return ((words[..., None] >> np.arange(28, -1, -4, dtype=np.uint32)) & 15).astype(
        np.uint8
    ).reshape(*trellis.shape[:-1], 256)


def reconstruct_k4_states(edges: np.ndarray) -> np.ndarray:
    """s[j] = e[j] | e[j-1]<<4 | e[j-2]<<8 | e[j-3]<<12, modulo 256."""
    values = _edges(edges).astype(np.uint16)
    states = values.copy()
    for lag in (1, 2, 3):
        states |= np.roll(values, lag, axis=-1) << (4 * lag)
    return states


def pack_k4_states(states: np.ndarray) -> np.ndarray:
    """Reject states that cannot be represented by this cyclic K4 stream."""
    _array(states, "<u2", "states")
    edges = (states & 15).astype(np.uint8)
    if not np.array_equal(reconstruct_k4_states(edges), states):
        raise ValueError("states violate the cyclic sliding-window K4 recurrence")
    return pack_k4_edges(edges)


def mcg_half_values(states: np.ndarray) -> np.ndarray:
    """EXL3's uint32 MCG and one correctly rounded binary16 addition; alpha=1."""
    _array(states, "<u2", "states")
    product = states.astype(np.uint32) * np.uint32(0xCBAC1FED)
    word = (product & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    low = (word & 0xFFFF).astype("<u2").view("<f2").astype(np.float32)
    high = (word >> 16).astype("<u2").view("<f2").astype(np.float32)
    return (low + high).astype("<f2")


def e2m1_codes(values: np.ndarray) -> np.ndarray:
    """Native finite RNE projection, saturating to +/-6 and preserving -0.

    Non-finite source values are rejected by this interchange policy, even
    though CUDA's saturating conversion defines a result for them.
    """
    if not isinstance(values, np.ndarray) or values.dtype.kind != "f":
        raise ValueError("E2M1 conversion requires floating-point values")
    if not np.isfinite(values).all():
        raise ValueError("non-finite E2M1 source")
    magnitude = np.abs(values.astype(np.float64))
    lower = np.searchsorted(_MIDPOINTS, magnitude, side="left")
    tie = (lower < 7) & (magnitude == _MIDPOINTS[np.minimum(lower, 6)])
    rounded = lower + (tie & ((lower & 1) != 0))
    return rounded.astype(np.uint8) | (np.signbit(values).astype(np.uint8) << 3)


def e2m1_values(codes: np.ndarray) -> np.ndarray:
    _array(codes, "u1", "E2M1 codes")
    if np.any(codes > 15):
        raise ValueError("E2M1 codes must be nibbles in [0, 15]")
    return np.copysign(E2M1_MAGNITUDES[codes & 7], np.where(codes & 8, -1, 1)).astype(np.float32)


def pack_e2m1(codes: np.ndarray) -> np.ndarray:
    _array(codes, "u1", "E2M1 codes")
    if codes.ndim < 1 or codes.shape[-1] % 2 or np.any(codes > 15):
        raise ValueError("E2M1 codes require an even final dimension and values in [0, 15]")
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(weight: np.ndarray) -> np.ndarray:
    _array(weight, "u1", "packed E2M1")
    if weight.ndim < 1:
        raise ValueError("packed E2M1 requires at least one dimension")
    return np.stack((weight & 15, weight >> 4), axis=-1).reshape(*weight.shape[:-1], -1)


def e4m3_scale_values(raw: np.ndarray) -> np.ndarray:
    """Decode strictly positive finite E4M3FN: 0x01..0x7e (2**-9..448)."""
    _array(raw, "u1", "scale_e4m3")
    if np.any((raw == 0) | (raw > 0x7E)):
        raise ValueError("logical scales must be positive finite E4M3 bytes 0x01..0x7e")
    exponent = (raw >> 3).astype(np.int32)
    mantissa = (raw & 7).astype(np.float64)
    return np.where(exponent == 0, mantissa / 512, np.ldexp(1 + mantissa / 8, exponent - 7))


def _global_scale(value: np.ndarray) -> float:
    _array(value, "<f4", "global_scale", ())
    if not np.isfinite(value) or value <= 0:
        raise ValueError("global_scale must be a positive finite FP32 scalar")
    return float(value)


def fit_e4m3_scales(weight: np.ndarray, codes: np.ndarray, global_scale: np.ndarray) -> np.ndarray:
    """Fit fixed E2M1 codes by float64 block SSE, then nearest positive E4M3.

    Ties choose the even E4M3 code. An all-zero code block uses 1.0 (0x38),
    since its scale cannot change the error. This does not select trellis paths.
    """
    _array(weight, "<f4", "weight")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [N, K]")
    _shape(*weight.shape)
    if not np.isfinite(weight).all():
        raise ValueError("weight must be finite")
    _array(codes, "u1", "E2M1 codes", weight.shape)
    gs = _global_scale(global_scale)
    basis = e2m1_values(codes).astype(np.float64).reshape(weight.shape[0], -1, 16) * gs
    numerator = (weight.astype(np.float64).reshape(basis.shape) * basis).sum(axis=-1)
    denominator = (basis * basis).sum(axis=-1)
    target = numerator / np.where(denominator == 0, 1, denominator)
    levels = e4m3_scale_values(np.arange(1, 127, dtype=np.uint8))  # 1008 bytes
    upper = np.searchsorted(levels, target).clip(0, 125)
    lower = (upper - 1).clip(0, 125)
    low_error = np.abs(target - levels[lower])
    high_error = np.abs(target - levels[upper])
    choose_upper = (high_error < low_error) | ((high_error == low_error) & (((upper + 1) & 1) == 0))
    result = (np.where(choose_upper, upper, lower) + 1).astype(np.uint8)
    return np.where(denominator == 0, np.uint8(0x38), result).astype(np.uint8)


def validate_payload(payload: P4Payload) -> None:
    _shape(payload.rows, payload.width)
    _array(payload.trellis, "<i2", "trellis", (payload.width // 16, payload.rows // 16, 64))
    _array(payload.scale_e4m3, "u1", "scale_e4m3", (payload.rows, payload.width // 16))
    e4m3_scale_values(payload.scale_e4m3)
    _global_scale(payload.global_scale)


def decode_p4(payload: P4Payload) -> P4Operands:
    """Emit native nibbles and actual scale bytes; no state lookup table."""
    validate_payload(payload)
    codes = np.empty((payload.rows, payload.width), dtype=np.uint8)
    positions = np.arange(256, dtype=np.uint16)
    lane, slot = positions // 8, positions % 8
    local_n = lane // 4 + (slot // 4) * 8
    local_k = (lane % 4) * 2 + (slot % 2) + ((slot // 2) % 2) * 8
    for kt in range(payload.width // 16):
        for nt in range(payload.rows // 16):
            states = reconstruct_k4_states(unpack_k4_edges(payload.trellis[kt, nt]))
            codes[nt * 16 + local_n, kt * 16 + local_k] = e2m1_codes(mcg_half_values(states))
    return P4Operands(pack_e2m1(codes), payload.scale_e4m3.copy(), payload.global_scale.copy())


def scale_byte_offset(row: int, column: int, *, rows: int, width: int, layout: str) -> int:
    """Byte address of the physical scale for weight [row, column]."""
    _shape(rows, width)
    if type(row) is not int or type(column) is not int or not (0 <= row < rows and 0 <= column < width):
        raise ValueError("scale address outside logical weight shape")
    group = column // 16
    if layout == SCALE_LAYOUT:
        return row * (width // 16) + group
    if layout == SWIZZLED_SCALE_LAYOUT:
        group_tiles = (width // 16 + 3) // 4
        return ((row // 128) * group_tiles + group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
    raise ValueError(f"unsupported scale layout: {layout}")


def swizzle_scales(raw: np.ndarray) -> np.ndarray:
    """ModelOpt 128x4 storage, with zero bytes only in padding positions."""
    if not isinstance(raw, np.ndarray) or raw.ndim != 2:
        raise ValueError("logical scale plane must be 2D")
    rows, groups = raw.shape
    _shape(rows, groups * 16)
    e4m3_scale_values(raw)
    rp, gp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    padded = np.zeros((rp, gp), dtype=np.uint8)
    padded[:rows, :groups] = raw
    return padded.reshape(rp // 128, 4, 32, gp // 4, 4).transpose(0, 3, 2, 1, 4).copy().reshape(rp, gp)


def unswizzle_scales(raw: np.ndarray, *, rows: int, width: int) -> np.ndarray:
    _shape(rows, width)
    groups = width // 16
    rp, gp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    _array(raw, "u1", "swizzled scale plane", (rp, gp))
    padded = raw.reshape(rp // 128, gp // 4, 32, 4, 4).transpose(0, 3, 2, 1, 4).copy().reshape(rp, gp)
    if padded[rows:, :].any() or padded[:rows, groups:].any():
        raise ValueError("nonzero scale padding")
    logical = padded[:rows, :groups].copy()
    e4m3_scale_values(logical)
    return logical


def _tensors(payload: P4Payload) -> dict[str, np.ndarray]:
    return {"global_scale": payload.global_scale, "scale_e4m3": payload.scale_e4m3, "trellis": payload.trellis}


def serialize_p4(payload: P4Payload) -> bytes:
    """Canonical safetensors subset: sorted JSON, 8-byte padding, no data holes."""
    validate_payload(payload)
    metadata = {**CONTRACT, "rows": str(payload.rows), "width": str(payload.width)}
    header = {"__metadata__": metadata}
    data = []
    offset = 0
    for name, tensor in _tensors(payload).items():
        raw = tensor.tobytes(order="C")
        metadata[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
        header[name] = {"dtype": {"global_scale": "F32", "scale_e4m3": "U8", "trellis": "I16"}[name],
                        "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        data.append(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(data)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def deserialize_p4(raw: bytes, *, expected_shape: tuple[int, int] | None = None,
                   expected_sha256: str | None = None) -> P4Payload:
    """Fail closed on ABI drift, malformed bytes, hashes, geometry, or padding."""
    if not isinstance(raw, bytes) or len(raw) < 8:
        raise ValueError("truncated P4 container")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("container SHA-256 mismatch")
    header_size = struct.unpack_from("<Q", raw)[0]
    if not 0 < header_size <= min(len(raw) - 8, 1 << 20) or header_size % 8:
        raise ValueError("invalid header size or endianness")
    try:
        header = json.loads(raw[8:8 + header_size], object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"invalid P4 header: {error}") from error
    if not isinstance(header, dict) or set(header) != {"__metadata__", "global_scale", "scale_e4m3", "trellis"}:
        raise ValueError("unexpected P4 tensor inventory")
    metadata = header["__metadata__"]
    metadata_keys = set(CONTRACT) | {"rows", "width", "global_scale_sha256", "scale_e4m3_sha256", "trellis_sha256"}
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys:
        raise ValueError("unexpected P4 metadata inventory")
    for key, value in CONTRACT.items():
        if metadata[key] != value:
            raise ValueError(f"P4 {key} mismatch: expected {value!r}, got {metadata[key]!r}")
    if any(type(metadata[key]) is not str for key in ("rows", "width")):
        raise ValueError("logical shape metadata must use decimal strings")
    try:
        rows, width = int(metadata["rows"]), int(metadata["width"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid logical shape") from error
    _shape(rows, width)
    if metadata["rows"] != str(rows) or metadata["width"] != str(width):
        raise ValueError("noncanonical logical shape")
    if expected_shape is not None:
        if not isinstance(expected_shape, tuple) or len(expected_shape) != 2:
            raise ValueError("expected_shape must be (N, K)")
        _shape(*expected_shape)
        if (rows, width) != expected_shape:
            raise ValueError("logical shape does not match consumer expectation")
    specs = {"global_scale": ("F32", "<f4", ()), "scale_e4m3": ("U8", "u1", (rows, width // 16)),
             "trellis": ("I16", "<i2", (width // 16, rows // 16, 64))}
    tensors = {}
    offset = 0
    for name, (tag, dtype, shape) in specs.items():
        size = math.prod(shape) * np.dtype(dtype).itemsize
        expected = {"dtype": tag, "shape": list(shape), "data_offsets": [offset, offset + size]}
        # Canonical serialization below also rejects JSON bool/float aliases of integers.
        if header[name] != expected:
            raise ValueError(f"{name} dtype, shape, or data offset mismatch")
        data = raw[8 + header_size + offset:8 + header_size + offset + size]
        if len(data) != size or hashlib.sha256(data).hexdigest() != metadata[name + "_sha256"]:
            raise ValueError(f"{name} byte count or SHA-256 mismatch")
        tensors[name] = np.frombuffer(data, dtype=dtype).copy().reshape(shape)
        offset += size
    if 8 + header_size + offset != len(raw):
        raise ValueError("trailing data in P4 container")
    payload = P4Payload(rows, width, **tensors)
    validate_payload(payload)
    if serialize_p4(payload) != raw:
        raise ValueError("noncanonical P4 container layout or padding")
    return payload


def read_p4(path: Path, **expectations) -> P4Payload:
    return deserialize_p4(Path(path).read_bytes(), **expectations)


def storage_accounting(payload: P4Payload) -> dict:
    """Exact integer byte totals and rational bpw, including the FP32 scalar."""
    serialized = serialize_p4(payload)
    weights = payload.rows * payload.width
    components = {name + "_bytes": tensor.nbytes for name, tensor in _tensors(payload).items()}
    payload_bytes = sum(components.values())
    header_bytes = struct.unpack_from("<Q", serialized)[0]
    result = {"logical_weights": weights, **components, "state_lut_bytes": 0,
              "payload_bytes": payload_bytes, "container_length_prefix_bytes": 8,
              "container_header_bytes_including_padding": header_bytes,
              "container_overhead_bytes": len(serialized) - payload_bytes,
              "file_bytes": len(serialized)}
    for name, count in (("stream_and_scales", payload_bytes - 4), ("payload", payload_bytes),
                        ("container_overhead", len(serialized) - payload_bytes), ("file", len(serialized))):
        ratio = Fraction(count * 8, weights)
        result[name + "_bpw"] = {"numerator": ratio.numerator, "denominator": ratio.denominator, "decimal": float(ratio)}
    return result


def verify_operands(payload: P4Payload, expected: P4Operands) -> dict[str, int]:
    """Compare every output byte, including signed-zero nibbles and FP32 scale."""
    actual = decode_p4(payload)
    mismatches = {}
    for name in ("weight", "scale_e4m3", "global_scale"):
        reference, observed = getattr(expected, name), getattr(actual, name)
        _array(reference, observed.dtype.str, "expected " + name, observed.shape)
        mismatches[name] = sum(a != b for a, b in zip(reference.tobytes(), observed.tobytes(), strict=True))
    if any(mismatches.values()):
        raise ValueError(f"P4 operand byte mismatches: {mismatches}")
    return mismatches
