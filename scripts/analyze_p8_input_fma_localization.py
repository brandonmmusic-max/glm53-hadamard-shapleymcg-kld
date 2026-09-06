#!/usr/bin/env python3
"""Reproduce the v6 P8 prequant FMA localization on CPU.

The script reads the preserved 512-byte device trace and frozen synthetic
sidecar, regenerates the seeded BF16 input, and prints a JSON receipt. It never
uses CUDA and never writes an artifact.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime_patch"))
from p8_coupled_scales import hadamard_blocks, quantize_e4m3_ue8m0_per32


SEED = 20260905128
BLOCKS = (40, 62)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)


def hex32(value: int) -> str:
    return f"0x{int(value):08x}"


def fma_inner_h128(outer: np.ndarray, suh: np.ndarray) -> np.ndarray:
    """Transcribe the observed contracted first butterfly, then strict H128."""

    libm = ctypes.CDLL("libm.so.6")
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float

    outer = np.ascontiguousarray(outer, dtype=np.float32).reshape(-1, 128)
    suh = np.ascontiguousarray(suh, dtype=np.float32).reshape(-1, 128)
    work = np.empty_like(outer)
    for row in range(outer.shape[0]):
        for index in range(0, 128, 2):
            left, right = outer[row, index : index + 2]
            left_scale, right_scale = suh[row, index : index + 2]
            rounded_right = np.float32(right * right_scale)
            work[row, index] = np.float32(
                fmaf(float(left), float(left_scale), float(rounded_right))
            )
            work[row, index + 1] = np.float32(
                fmaf(float(left), float(left_scale), -float(rounded_right))
            )
    stride = 2
    while stride < 128:
        stage = work.reshape(-1, 128 // (2 * stride), 2, stride)
        next_stage = np.empty_like(stage)
        left, right = stage[:, :, 0, :], stage[:, :, 1, :]
        next_stage[:, :, 0, :] = left + right
        next_stage[:, :, 1, :] = left - right
        work = next_stage.reshape(-1, 128)
        stride *= 2
    divisor = np.float32(np.sqrt(np.float32(128.0)))
    return np.ascontiguousarray(work / divisor, dtype=np.float32).reshape(-1)


def analyze(trace_path: Path, sidecar_path: Path) -> dict[str, object]:
    archive = np.load(trace_path)
    trace_bytes = np.ascontiguousarray(archive["input_prequant_trace"])
    if trace_bytes.dtype != np.uint8 or trace_bytes.shape != (512,):
        raise RuntimeError("input trace is not the frozen 512-byte carrier")
    trace = trace_bytes.view(np.float32)
    if not np.isfinite(trace).all():
        raise RuntimeError("input trace retains a nonfinite sentinel")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    source_input = (
        torch.randn(1, 4096, generator=generator) * 0.03125
    ).to(torch.bfloat16)
    with safe_open(sidecar_path, framework="pt", device="cpu") as handle:
        suh = handle.get_tensor("gate_up_suh_fp16").float()
    outer = hadamard_blocks(source_input.to(torch.float16).float(), 512)
    canonical = hadamard_blocks(outer * suh, 128).reshape(-1)
    _, scale_bytes, _ = quantize_e4m3_ue8m0_per32(canonical.reshape(1, -1))
    contracted = fma_inner_h128(outer.numpy(), suh.numpy())

    actual_raw = np.concatenate((trace[0:32], trace[64:96]))
    actual_normalized = np.concatenate((trace[32:64], trace[96:128]))
    indices = np.concatenate(
        tuple(np.arange(block * 32, block * 32 + 32) for block in BLOCKS)
    )
    canonical_raw = canonical.numpy()[indices]
    contracted_raw = contracted[indices]
    scales = np.concatenate(
        tuple(
            np.full(
                32,
                np.float32(2.0 ** (int(scale_bytes[0, block]) - 127)),
                dtype=np.float32,
            )
            for block in BLOCKS
        )
    )
    canonical_normalized = canonical_raw / scales
    contracted_normalized = contracted_raw / scales

    actual_bits = bits(actual_raw)
    canonical_bits = bits(canonical_raw)
    contracted_bits = bits(contracted_raw)
    actual_normalized_bits = bits(actual_normalized)
    canonical_normalized_bits = bits(canonical_normalized)
    contracted_normalized_bits = bits(contracted_normalized)
    rows = []
    for offset, logical in enumerate(indices):
        block = int(logical // 32)
        rows.append(
            {
                "block": block,
                "logical_channel": int(logical),
                "raw_device": float(actual_raw[offset]),
                "raw_device_bits": hex32(actual_bits[offset]),
                "raw_canonical": float(canonical_raw[offset]),
                "raw_canonical_bits": hex32(canonical_bits[offset]),
                "raw_contracted_fma": float(contracted_raw[offset]),
                "raw_contracted_fma_bits": hex32(contracted_bits[offset]),
                "normalized_device": float(actual_normalized[offset]),
                "normalized_device_bits": hex32(actual_normalized_bits[offset]),
                "normalized_canonical": float(canonical_normalized[offset]),
                "normalized_canonical_bits": hex32(
                    canonical_normalized_bits[offset]
                ),
                "normalized_contracted_fma": float(
                    contracted_normalized[offset]
                ),
                "normalized_contracted_fma_bits": hex32(
                    contracted_normalized_bits[offset]
                ),
                "ue8m0": int(scale_bytes[0, block]),
            }
        )

    return {
        "schema": "glm53.p8-input-fma-localization.v1",
        "evidence_level": "cpu-analysis-of-preserved-gpu-trace",
        "qualification_allowed": False,
        "trace_path": str(trace_path),
        "trace_archive_sha256": sha256_file(trace_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "seed": SEED,
        "blocks": list(BLOCKS),
        "raw_device_sha256": bytes_sha256(actual_raw),
        "raw_canonical_sha256": bytes_sha256(canonical_raw),
        "raw_contracted_fma_sha256": bytes_sha256(contracted_raw),
        "raw_device_vs_canonical_bit_mismatches": int(
            np.count_nonzero(actual_bits != canonical_bits)
        ),
        "raw_device_vs_contracted_fma_bit_mismatches": int(
            np.count_nonzero(actual_bits != contracted_bits)
        ),
        "normalized_device_vs_canonical_bit_mismatches": int(
            np.count_nonzero(actual_normalized_bits != canonical_normalized_bits)
        ),
        "normalized_device_vs_contracted_fma_bit_mismatches": int(
            np.count_nonzero(
                actual_normalized_bits != contracted_normalized_bits
            )
        ),
        "conclusion": (
            "captured prequant values equal the contracted-left-product FMA "
            "transcription and differ from the separately-rounded CPU contract"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.sidecar), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
