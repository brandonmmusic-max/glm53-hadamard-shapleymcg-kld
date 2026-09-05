#!/usr/bin/env python3
"""Preregistered raw-FC1 localization for the full-coupled P8 M1 path.

The opt-in kernel stops after the native W13 MMAs and captures the raw FP16
gate/up projections. It cannot produce a closure pass or a serving result.
Without ``--execute`` this script prints the frozen protocol and touches no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_p8_coupled_m1_device_closure as closure


SEED = closure.SEED
EXPERT_IDS = closure.EXPERT_IDS
ROWS_CAPACITY = 128
WORDS_PER_ROW = 128
PROTOCOL = {
    "schema": "glm53.p8-full-coupled-m1-raw-fc1-diagnostic.v1",
    "evidence_level": "gpu-localization-only",
    "product": "P8 K4 procedural MCG alpha2 to E4M3/UE8M0-K32",
    "purpose": "localize v4 middle-carrier divergence before or after raw W13 MMA",
    "geometry": {
        "layer": 3,
        "rank": 0,
        "tokens": 1,
        "experts": 288,
        "selected_experts": list(EXPERT_IDS),
        "hidden": 4096,
        "local_intermediate": 512,
        "topk": 8,
        "fc1_tile_n": 128,
        "physical_rows": ROWS_CAPACITY,
    },
    "capture": (
        "for each route slot and output tile, intermediate row slot*16+tile "
        "stores gate[128]|up[128] as raw FP16 immediately before joint H128"
    ),
    "input_prequant_trace": (
        "exactly 512 bytes in the MCG-dead trellis_lut ABI slot: raw K32 block40, "
        "normalized block40, raw block62, normalized block62 as FP32"
    ),
    "reference_arms": [
        "canonical CPU input carrier",
        "observed device E4M3/UE8M0 input carrier reconstructed in logical K order",
    ],
    "projection_contract": (
        "W13 trellis slot0=gate, slot1=up; physical W13 scale plane=up|gate"
    ),
    "numeric_gate": {
        "each_selected_expert_and_projection_cosine_strictly_greater_than": 0.995,
        "each_selected_expert_and_projection_relative_l2_strictly_less_than": 0.12,
    },
    "decision_rule": {
        "localized_after_raw_fc1": (
            "observed-carrier gate and up both satisfy the numeric gate; continue "
            "instrumentation after the first joint H128"
        ),
        "localized_at_or_before_raw_fc1": (
            "either observed-carrier projection fails; inspect stream decode, W13 "
            "scale/MMA geometry, and projection ordering"
        ),
    },
    "forbidden_claims": [
        "closure pass",
        "KLD qualification",
        "throughput qualification",
        "serving qualification",
        "all-layer or all-rank qualification",
    ],
    "failure_policy": (
        "fail closed on identity drift, non-SM120, non-M1/full-coupled dispatch, "
        "capture-row collision or bounds error, missing debug carrier, nonfinite "
        "FP16 capture, or malformed projection geometry"
    ),
    "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    "ldlq": False,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def capture_rows() -> list[list[int]]:
    rows = [[slot * 16 + tile for tile in range(4)] for slot in range(8)]
    flat = [row for route in rows for row in route]
    if len(flat) != len(set(flat)) or min(flat) < 0 or max(flat) >= ROWS_CAPACITY:
        raise RuntimeError("raw FC1 capture row map is not an in-bounds bijection")
    return rows


def unpack_raw_fc1(intermediate_u32):
    """Extract [8,512] FP16 gate and up from diagnostic capture rows."""
    import torch

    if intermediate_u32.dtype != torch.int32 or intermediate_u32.ndim != 1:
        raise ValueError("raw FC1 carrier must be flat int32")
    if intermediate_u32.numel() < ROWS_CAPACITY * WORDS_PER_ROW:
        raise ValueError("raw FC1 carrier is smaller than its frozen payload plane")
    words = intermediate_u32.detach().cpu().contiguous()
    gates, ups = [], []
    for route in capture_rows():
        gate_tiles, up_tiles = [], []
        for row in route:
            packed = words[
                row * WORDS_PER_ROW : (row + 1) * WORDS_PER_ROW
            ].view(torch.float16)
            if packed.numel() != 256:
                raise RuntimeError("raw FC1 row does not contain gate128|up128")
            gate_tiles.append(packed[:128])
            up_tiles.append(packed[128:])
        gates.append(torch.cat(gate_tiles))
        ups.append(torch.cat(up_tiles))
    return torch.stack(gates).contiguous(), torch.stack(ups).contiguous()


def audit_raw_capture_bounds(intermediate_u32) -> dict[str, object]:
    """Prove selected writes, actual expert routing, counts, and untouched space."""
    import torch

    expected_words = ROWS_CAPACITY * 132
    if intermediate_u32.dtype != torch.int32 or intermediate_u32.numel() != expected_words:
        raise ValueError("diagnostic intermediate carrier must be exactly 16896 int32")
    words = intermediate_u32.detach().cpu().contiguous()
    selected = {row for route in capture_rows() for row in route}
    payload = words[: ROWS_CAPACITY * WORDS_PER_ROW].reshape(
        ROWS_CAPACITY, WORDS_PER_ROW
    )
    for row in selected:
        if bool((payload[row] == -1).any()):
            raise RuntimeError(f"selected raw FC1 row {row} retained sentinel words")
    nonselected = [row for row in range(ROWS_CAPACITY) if row not in selected]
    if not bool((payload[nonselected] == -1).all()):
        raise RuntimeError("raw FC1 capture wrote outside selected route/tile rows")

    tail = words[ROWS_CAPACITY * WORDS_PER_ROW :]
    metadata = tail[:32]
    counts = tail[32:64]
    unused = tail[64:]
    decoded = []
    for slot, expert in enumerate(EXPERT_IDS):
        for tile in range(4):
            trace_slot = slot * 4 + tile
            value = int(metadata[trace_slot])
            observed = {
                "expert": value & 0x1FF,
                "route": (value >> 9) & 0x7,
                "tile": (value >> 12) & 0x3,
                "write_count": int(counts[trace_slot]),
            }
            expected = {
                "expert": int(expert), "route": slot, "tile": tile, "write_count": 1
            }
            if observed != expected:
                raise RuntimeError(
                    f"raw FC1 observed route/tile metadata mismatch at {trace_slot}: "
                    f"expected {expected}, got {observed}"
                )
            decoded.append(observed)
    if not bool((unused == -1).all()):
        raise RuntimeError("raw FC1 metadata trace exceeded its 64-word bound")
    return {
        "selected_rows": sorted(selected),
        "selected_rows_all_written": True,
        "nonselected_rows": len(nonselected),
        "nonselected_payload_sentinel_unchanged": True,
        "observed_route_tile_expert": decoded,
        "all_write_counts_one": True,
        "unused_tail_words": int(unused.numel()),
        "unused_tail_sentinel_unchanged": True,
    }


def reconstruct_input(payload, scale):
    """Undo the native K32 lane permutation and reconstruct logical FP32 A."""
    import torch

    if tuple(payload.shape) != (1, 4096) or tuple(scale.shape) != (1, 128):
        raise ValueError("observed input carrier has unexpected geometry")
    logical = logical_input_payload(payload)
    exponent = scale.to(torch.int16) - 127
    factor = torch.pow(torch.tensor(2.0), exponent.float()).unsqueeze(-1)
    return logical.view(torch.float8_e4m3fn).float().mul(factor).reshape(1, 4096)


def logical_input_payload(payload):
    """Undo only the native K32 lane permutation, retaining E4M3 bytes."""
    import torch

    if tuple(payload.shape) != (1, 4096):
        raise ValueError("observed input payload has unexpected geometry")
    perm = closure.K32_PERM
    inverse = [0] * 32
    for output_index, input_index in enumerate(perm):
        inverse[input_index] = output_index
    index = torch.tensor(inverse, dtype=torch.long)
    return payload.reshape(1, 128, 32).index_select(-1, index).contiguous()


def unpack_input_prequant_trace(carrier):
    """Validate the 512-byte sentinel-backed FP32 input trace."""
    import torch

    if carrier.dtype != torch.uint8 or carrier.numel() != 512:
        raise ValueError("input prequant trace must be exactly 512 uint8 bytes")
    values = carrier.detach().cpu().contiguous().view(torch.float32)
    if values.numel() != 128 or not bool(torch.isfinite(values).all()):
        raise RuntimeError("input prequant trace was not completely overwritten")
    return {
        "block40_raw": values[0:32],
        "block40_normalized": values[32:64],
        "block62_raw": values[64:96],
        "block62_normalized": values[96:128],
    }


def fp32_evidence(value) -> dict[str, object]:
    import torch

    value = value.detach().cpu().contiguous().to(torch.float32)
    return {
        "values": value.tolist(),
        "bits_hex": [f"0x{int(bits) & 0xFFFFFFFF:08x}" for bits in value.view(torch.int32)],
        "sha256": closure._tensor_bytes_sha256(value),
    }


def e4m3_midpoint_distance(normalized) -> list[float]:
    """Absolute distance to the nearest midpoint of finite E4M3 values."""
    import torch

    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).float()
    finite = torch.unique(values[torch.isfinite(values)]).sort().values
    midpoints = (finite[:-1] + finite[1:]) * 0.5
    result = []
    for item in normalized.detach().cpu().float():
        result.append(float((midpoints - item).abs().min()))
    return result


def nearest_e4m3_midpoint(value: float) -> tuple[float, float, float]:
    """Return nearest midpoint, signed value-minus-midpoint, and absolute distance."""
    import torch

    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).float()
    finite = torch.unique(values[torch.isfinite(values)]).sort().values
    midpoints = (finite[:-1] + finite[1:]) * 0.5
    item = torch.tensor(float(value), dtype=torch.float32)
    index = int((midpoints - item).abs().argmin())
    midpoint = float(midpoints[index])
    signed = float(item) - midpoint
    return midpoint, signed, abs(signed)


def float32_bits(value: float) -> str:
    import struct

    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def input_trace_rows(
    input_trace,
    canonical_trace,
    logical_observed_payload,
    expected_scale,
    observed_scale,
) -> list[dict[str, object]]:
    """Emit the preregistered nonaggregate 64-row input trace."""
    import torch

    rows = []
    for block in (40, 62):
        raw_name = f"block{block}_raw"
        normalized_name = f"block{block}_normalized"
        expected_codes = (
            canonical_trace[normalized_name]
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        wire_positions = logical_to_wire_positions(block)
        for channel in range(32):
            raw_device = float(input_trace[raw_name][channel])
            raw_reference = float(canonical_trace[raw_name][channel])
            normalized_device = float(input_trace[normalized_name][channel])
            normalized_reference = float(canonical_trace[normalized_name][channel])
            device_midpoint, device_signed, device_abs = nearest_e4m3_midpoint(
                normalized_device
            )
            reference_midpoint, reference_signed, reference_abs = (
                nearest_e4m3_midpoint(normalized_reference)
            )
            rows.append(
                {
                    "block": block,
                    "logical_channel": block * 32 + channel,
                    "logical_within_block": channel,
                    "wire_flat_index": wire_positions[channel],
                    "raw_device": raw_device,
                    "raw_device_bits": float32_bits(raw_device),
                    "raw_reference": raw_reference,
                    "raw_reference_bits": float32_bits(raw_reference),
                    "normalized_device": normalized_device,
                    "normalized_device_bits": float32_bits(normalized_device),
                    "normalized_reference": normalized_reference,
                    "normalized_reference_bits": float32_bits(normalized_reference),
                    "ue8m0_observed": int(observed_scale[0, block]),
                    "ue8m0_expected": int(expected_scale[0, block]),
                    "e4m3_observed": int(logical_observed_payload[0, block, channel]),
                    "e4m3_expected": int(expected_codes[channel]),
                    "device_nearest_midpoint": device_midpoint,
                    "device_signed_distance_to_midpoint": device_signed,
                    "device_abs_distance_to_midpoint": device_abs,
                    "reference_nearest_midpoint": reference_midpoint,
                    "reference_signed_distance_to_midpoint": reference_signed,
                    "reference_abs_distance_to_midpoint": reference_abs,
                }
            )
    return rows


def logical_to_wire_positions(block: int) -> list[int]:
    inverse = [0] * 32
    for wire_within, logical_within in enumerate(closure.K32_PERM):
        inverse[logical_within] = wire_within
    return [block * 32 + inverse[logical] for logical in range(32)]


def raw_reference(sidecar: Path, source) -> tuple[object, object, list[dict[str, object]]]:
    import torch

    gate_rows, up_rows, receipts = [], [], []
    for expert in EXPERT_IDS:
        gate, up, _down, receipt = closure._decode_expert(sidecar, expert)
        gate_rows.append((source @ gate.T).to(torch.float16).squeeze(0))
        up_rows.append((source @ up.T).to(torch.float16).squeeze(0))
        receipts.append(receipt)
    return torch.stack(gate_rows), torch.stack(up_rows), receipts


def metric(actual, expected) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    actual, expected = actual.float(), expected.float()
    return {
        "cosine": float(functional.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1))),
        "relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-9)),
        "max_abs": float((actual - expected).abs().max()),
    }


def metric_pass(value: dict[str, float]) -> bool:
    gate = PROTOCOL["numeric_gate"]
    return (
        math.isfinite(value["cosine"])
        and math.isfinite(value["relative_l2"])
        and value["cosine"] > gate[
            "each_selected_expert_and_projection_cosine_strictly_greater_than"
        ]
        and value["relative_l2"] < gate[
            "each_selected_expert_and_projection_relative_l2_strictly_less_than"
        ]
    )


def comparison(actual, expected) -> dict[str, object]:
    import torch

    if actual.shape != expected.shape or tuple(actual.shape) != (8, 512):
        raise ValueError("raw FC1 comparison requires selected [8,512] projections")
    rows = []
    for slot, expert in enumerate(EXPERT_IDS):
        value = metric(actual[slot], expected[slot])
        value.update(
            expert=int(expert),
            fp16_byte_mismatches=int(
                (
                    actual[slot].contiguous().view(torch.uint8)
                    != expected[slot].contiguous().view(torch.uint8)
                ).sum()
            ),
        )
        value["numeric_gate"] = metric_pass(value)
        rows.append(value)
    aggregate = metric(actual, expected)
    aggregate["fp16_byte_mismatches"] = int(
        (
            actual.contiguous().view(torch.uint8)
            != expected.contiguous().view(torch.uint8)
        ).sum()
    )
    return {"aggregate": aggregate, "per_expert": rows}


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from p8_coupled_scales import hadamard_blocks, quantize_e4m3_ue8m0_per32
    from p8_native_kernel import P8NativeTPMoE

    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_id):
        raise RuntimeError("probe requires immutable image ID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic container requires exactly one visible CUDA device")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("raw FC1 diagnostic requires SM120")
    identities = {
        "sidecar": closure.sha256_file(args.sidecar),
        "design": closure.sha256_file(args.design),
        "transform": closure.sha256_file(args.transform),
        "runtime_manifest": closure.sha256_file(args.runtime_manifest),
    }
    expected = {
        "sidecar": args.sidecar_sha256,
        "design": args.design_sha256,
        "transform": args.transform_sha256,
        "runtime_manifest": args.runtime_manifest_sha256,
    }
    if identities != expected:
        raise RuntimeError(f"input identity mismatch: expected {expected}, got {identities}")
    sources = closure._verify_runtime_sources(
        args.runtime_manifest, args.runtime_manifest_sha256
    )
    metadata, scales, scale_receipts = closure._load_sidecar_reference(
        args.sidecar, args.transform_sha256
    )
    if metadata.get("source_design_sha256") != args.design_sha256:
        raise RuntimeError("sidecar source design hash differs")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x_cpu = (torch.randn(1, 4096, generator=generator) * 0.03125).to(torch.bfloat16)
    logits = torch.randn(8, generator=generator)
    weights_cpu = torch.softmax(logits, 0).float()
    ids_cpu = torch.tensor(EXPERT_IDS, dtype=torch.int32).reshape(1, 8)
    canonical_source = hadamard_blocks(x_cpu.to(torch.float16).float(), 512)
    canonical_source = hadamard_blocks(
        canonical_source * scales.gate_up_suh.float(), 128
    )
    expected_payload, expected_scale, canonical_source = (
        quantize_e4m3_ue8m0_per32(canonical_source)
    )
    expected_payload = closure.packed_quantizer_payload(
        expected_payload, 1, 4096
    )
    canonical_gate, canonical_up, weight_receipts = raw_reference(
        args.sidecar, canonical_source
    )

    runtime = P8NativeTPMoE(
        args.sidecar,
        device=torch.device("cuda"),
        tp_rank=0,
        layer=3,
        expected_design_sha256=args.design_sha256,
        expected_transform_sha256=args.transform_sha256,
        topk=8,
        hidden=4096,
        intermediate=512,
        swiglu_limit=10.0,
        deterministic_output=True,
        small_m_scheduler=True,
        fc1_tile_n=128,
        debug_capture=True,
        diagnostic_raw_fc1=True,
        fuse_scratch_zero=False,
    )
    _unused = runtime(
        x_cpu.cuda(), weights_cpu.reshape(1, 8).cuda(), ids_cpu.cuda()
    )
    torch.cuda.synchronize()
    required_dispatch = {
        "small_m": True,
        "materialized": True,
        "fused_scratch_zero": False,
        "fc1_tile_n": 128,
        "tile_m": 16,
        "diagnostic_raw_fc1": True,
    }
    if runtime.debug_dispatch != required_dispatch:
        raise RuntimeError(f"wrong raw FC1 dispatch: {runtime.debug_dispatch}")
    expected_debug = closure.REQUIRED_DEBUG | {"input_prequant_trace"}
    if set(runtime.debug_tensors) != expected_debug:
        raise RuntimeError("raw FC1 diagnostic lacks required wrapper carriers")
    debug = runtime.debug_tensors
    observed_payload = debug["packed_a"].view(torch.uint8)[:4096].cpu().reshape(1, 4096)
    observed_scale = debug["scale_flat"].view(torch.uint8)[:128].cpu().reshape(1, 128)
    observed_source = reconstruct_input(observed_payload, observed_scale)
    input_trace = unpack_input_prequant_trace(debug["input_prequant_trace"])
    canonical_blocks = canonical_source.reshape(1, 128, 32)
    canonical_trace = {}
    for block in (40, 62):
        raw = canonical_blocks[0, block]
        exponent = expected_scale[0, block].to(torch.int16) - 127
        factor = torch.pow(torch.tensor(2.0), exponent.float())
        canonical_trace[f"block{block}_raw"] = raw
        canonical_trace[f"block{block}_normalized"] = raw / factor
    logical_observed_payload = logical_input_payload(observed_payload)
    input_trace_metrics = {}
    for block in (40, 62):
        raw_name = f"block{block}_raw"
        normalized_name = f"block{block}_normalized"
        normalized_bytes = (
            input_trace[normalized_name]
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        input_trace_metrics[f"block{block}"] = {
            "raw_vs_canonical": metric(input_trace[raw_name], canonical_trace[raw_name]),
            "normalized_vs_canonical": metric(
                input_trace[normalized_name], canonical_trace[normalized_name]
            ),
            "normalized_e4m3_byte_mismatches_vs_observed_wire": int(
                (normalized_bytes != logical_observed_payload[0, block]).sum()
            ),
            "raw_sha256": closure._tensor_bytes_sha256(input_trace[raw_name]),
            "normalized_sha256": closure._tensor_bytes_sha256(
                input_trace[normalized_name]
            ),
            "observed_raw_fp32": fp32_evidence(input_trace[raw_name]),
            "canonical_raw_fp32": fp32_evidence(canonical_trace[raw_name]),
            "observed_normalized_fp32": fp32_evidence(
                input_trace[normalized_name]
            ),
            "canonical_normalized_fp32": fp32_evidence(
                canonical_trace[normalized_name]
            ),
            "observed_normalized_distance_to_nearest_e4m3_midpoint": (
                e4m3_midpoint_distance(input_trace[normalized_name])
            ),
            "canonical_normalized_distance_to_nearest_e4m3_midpoint": (
                e4m3_midpoint_distance(canonical_trace[normalized_name])
            ),
            "ue8m0_scale_byte": {
                "observed": int(observed_scale[0, block]),
                "canonical": int(expected_scale[0, block]),
            },
            "logical_e4m3_bytes_observed": logical_observed_payload[0, block].tolist(),
            "logical_to_wire_flat_indices": logical_to_wire_positions(block),
        }
    input_rows = input_trace_rows(
        input_trace,
        canonical_trace,
        logical_observed_payload,
        expected_scale,
        observed_scale,
    )
    observed_gate, observed_up, _ = raw_reference(args.sidecar, observed_source)
    # Preserve raw evidence before any completeness/finite gate can fail.
    # These bounded diagnostic buffers contain no model or teacher tensors.
    import numpy as np
    capture_arrays = {
        "scratch_u32": debug["intermediate_u32"].detach().cpu().numpy(),
        "input_prequant_trace": debug["input_prequant_trace"].detach().cpu().numpy(),
    }
    if sum(value.nbytes for value in capture_arrays.values()) > 4 * 1024 * 1024:
        raise RuntimeError("raw diagnostic evidence exceeds frozen 4 MiB limit")
    capture_path = args.output.parent / "raw-capture.npz"
    with capture_path.open("xb") as stream:
        np.savez(stream, **capture_arrays)
    actual_gate, actual_up = unpack_raw_fc1(debug["intermediate_u32"])
    capture_audit = audit_raw_capture_bounds(debug["intermediate_u32"])
    if not bool(torch.isfinite(actual_gate).all() and torch.isfinite(actual_up).all()):
        raise RuntimeError("raw FC1 capture contains nonfinite FP16")

    comparisons = {
        "canonical_input": {
            "gate": comparison(actual_gate, canonical_gate),
            "up": comparison(actual_up, canonical_up),
        },
        "observed_device_input": {
            "gate": comparison(actual_gate, observed_gate),
            "up": comparison(actual_up, observed_up),
        },
        "projection_swap_diagnostic": {
            "actual_gate_vs_observed_up": comparison(actual_gate, observed_up),
            "actual_up_vs_observed_gate": comparison(actual_up, observed_gate),
        },
    }
    observed_pass = all(
        row["numeric_gate"]
        for projection in comparisons["observed_device_input"].values()
        for row in projection["per_expert"]
    )
    decision = (
        "localized_after_raw_fc1"
        if observed_pass
        else "localized_at_or_before_raw_fc1"
    )
    return {
        "schema": PROTOCOL["schema"],
        "decision": decision,
        "closure_pass": False,
        "qualification_allowed": False,
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image_id,
        "identities": identities,
        "executed_sources": sources,
        "capture_rows": capture_rows(),
        "dispatch": runtime.debug_dispatch,
        "input_carrier": {
            "payload_mismatches_vs_canonical": int((observed_payload != expected_payload).sum()),
            "scale_mismatches_vs_canonical": int((observed_scale != expected_scale).sum()),
            "payload_sha256": closure._tensor_bytes_sha256(observed_payload),
            "scale_sha256": closure._tensor_bytes_sha256(observed_scale),
        },
        "input_prequant_trace": {
            "bytes": 512,
            "all_values_overwrote_nan_sentinel": True,
            "metrics": input_trace_metrics,
            "rows": input_rows,
        },
        "raw_capture": {
            "gate_sha256": closure._tensor_bytes_sha256(actual_gate),
            "up_sha256": closure._tensor_bytes_sha256(actual_up),
            "shape": [8, 512],
            "dtype": "torch.float16",
            "bounds_and_dispatch_audit": capture_audit,
        },
        "comparisons": comparisons,
        "observed_device_input_all_experts_and_projections_pass": observed_pass,
        "scale_receipts": scale_receipts,
        "selected_weight_decode": weight_receipts,
        "gpu_used": True,
        "kld_tested": False,
        "throughput_tested": False,
    }


def outer_execute(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image):
        raise ValueError("--image must be an immutable sha256 image ID")
    if args.output != args.output.resolve() or args.output.exists():
        raise ValueError("fresh canonical output directory required")
    for path, expected in (
        (args.sidecar, args.sidecar_sha256),
        (args.design, args.design_sha256),
        (args.transform, args.transform_sha256),
    ):
        if not path.is_file() or closure.sha256_file(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    actual_image = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual_image != args.image:
        raise ValueError("local image identity differs")
    manifest_line = subprocess.check_output(
        ["docker", "run", "--rm", "--network=none", "--runtime=runc",
         "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum",
         args.image, "/opt/p8-coupled-runtime/image-manifest.json"], text=True
    ).strip()
    if manifest_line.split()[0] != args.runtime_manifest_sha256:
        raise ValueError("pinned installed runtime manifest differs")
    active = subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu_device, "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True
    ).strip()
    if active:
        raise RuntimeError("selected diagnostic GPU is not idle")
    temperature = int(subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu_device, "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"], text=True
    ).strip())
    if temperature > 75:
        raise RuntimeError(f"startup temperature {temperature} C exceeds 75 C")

    args.output.mkdir(mode=0o700, parents=True)
    repo = Path(__file__).resolve().parents[1]
    command = [
        "docker", "run", "--rm", "--gpus", f"device={args.gpu_device}",
        "--network=none", "--ipc=private", "--shm-size=1g",
        "-e", "PYTHONPATH=/opt/p8-coupled-runtime:/work", "-e", "OMP_NUM_THREADS=2",
        "-e", "GLM53_P8_NATIVE=", "-e", "GLM53_P4_NATIVE=",
        "-v", f"{repo}:/work:ro",
        "-v", f"{args.sidecar}:/inputs/sidecar.safetensors:ro",
        "-v", f"{args.design}:/inputs/design.json:ro",
        "-v", f"{args.transform}:/inputs/transform.json:ro",
        "-v", f"{args.output}:/out:rw",
        "--entrypoint", "/opt/venv/bin/python", args.image,
        "/work/scripts/run_p8_coupled_m1_raw_fc1_diagnostic.py", "--probe",
        "--image-id", args.image,
        "--sidecar", "/inputs/sidecar.safetensors", "--sidecar-sha256", args.sidecar_sha256,
        "--design", "/inputs/design.json", "--design-sha256", args.design_sha256,
        "--transform", "/inputs/transform.json", "--transform-sha256", args.transform_sha256,
        "--runtime-manifest", "/opt/p8-coupled-runtime/image-manifest.json",
        "--runtime-manifest-sha256", args.runtime_manifest_sha256,
        "--output", "/out/result.json",
    ]
    launch = {
        "schema": "glm53.p8-full-coupled-m1-raw-fc1-launch.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image,
        "gpu_device": args.gpu_device,
        "startup_temperature_c": temperature,
        "command": command,
        "services_modified": [],
        "closure_pass": False,
        "qualification_allowed": False,
        "started_unix_ns": time.time_ns(),
    }
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True)
    (args.output / "probe.stdout.log").write_text(completed.stdout)
    (args.output / "probe.stderr.log").write_text(completed.stderr)
    launch.update(completed_unix_ns=time.time_ns(), exit_code=completed.returncode,
                  result_present=(args.output / "result.json").is_file())
    (args.output / "execution.json").write_text(json.dumps(launch, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError("raw FC1 diagnostic failed; preserved logs and receipt")
    result = json.loads((args.output / "result.json").read_text())
    if result.get("decision") not in PROTOCOL["decision_rule"]:
        raise RuntimeError("raw FC1 probe emitted an invalid decision")
    if result.get("closure_pass") is not False or result.get("qualification_allowed") is not False:
        raise RuntimeError("localization diagnostic claimed closure or qualification")
    print(json.dumps({"decision": result["decision"], "result": str(args.output / "result.json")}))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image")
    parser.add_argument("--image-id")
    parser.add_argument("--gpu-device")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--sidecar-sha256")
    parser.add_argument("--design", type=Path)
    parser.add_argument("--design-sha256")
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--transform-sha256")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _require(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    for name in ("sidecar_sha256", "design_sha256", "transform_sha256", "runtime_manifest_sha256"):
        value = getattr(args, name)
        if value is not None and not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError(f"--{name.replace('_', '-')} must be lowercase SHA256")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.execute and args.probe:
        raise ValueError("--execute and --probe are mutually exclusive")
    if not args.execute and not args.probe:
        print(json.dumps({"protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256}, sort_keys=True))
        return
    common = ("sidecar", "sidecar_sha256", "design", "design_sha256", "transform",
              "transform_sha256", "runtime_manifest_sha256", "output")
    if args.probe:
        _require(args, (*common, "image_id", "runtime_manifest"))
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        try:
            result = run_probe(args)
        except BaseException as error:
            result = {
                "schema": PROTOCOL["schema"], "decision": "fail",
                "closure_pass": False, "qualification_allowed": False,
                "protocol": PROTOCOL,
                "protocol_sha256": PROTOCOL_SHA256,
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(), "gpu_used": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True))
            raise
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return
    _require(args, (*common, "image", "gpu_device"))
    outer_execute(args)


if __name__ == "__main__":
    main()
