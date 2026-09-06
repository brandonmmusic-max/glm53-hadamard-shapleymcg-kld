#!/usr/bin/env python3
"""Opt-in SM120 M1 closure for the full-coupled P8 native wrapper at K3, K4 or K5.

Mixed-rate (v11 lineage) sibling of ``run_p8_coupled_m1_device_closure``: the
same exact activation-carrier and numerical gates, with the stored trellis rate
taken from ``--bits`` and every source pin pointing at the mixed-rate kernels.

Without ``--execute`` this script only prints the frozen protocol. Device work
is performed only by an explicit outer invocation, in the exact image named by
an immutable ID. The same file is then invoked inside that container with the
private ``--probe`` switch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import traceback
from typing import Sequence


SEED = 20260905128
DEFAULT_LAYER = 3
RANK = 0
EXPERT_IDS = (0, 1, 17, 63, 127, 191, 255, 287)
K32_PERM = (
    0, 1, 8, 9, 4, 5, 12, 13,
    2, 3, 10, 11, 6, 7, 14, 15,
    20, 21, 28, 29, 16, 17, 24, 25,
    22, 23, 30, 31, 18, 19, 26, 27,
)
REQUIRED_DEBUG = {
    "packed_a",
    "scale_flat",
    "intermediate_u32",
    "route_output",
    "token_map",
    "row_counts",
    "expert_tile_base",
}
MODULE_SOURCES = {
    "p8_native_kernel": "runtime_patch/p8_mixed_rate_image/p8_native_kernel.py",
    "p8_coupled_scales": "runtime_patch/p8_coupled_scales.py",
    "p8_smallm_schedule": "runtime_patch/p8_smallm_schedule.py",
    "b12x.moe._shared.kernels.dynamic": (
        "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels/dynamic.py"
    ),
    "b12x.moe._shared.kernels.p8_h128_fc1": (
        "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels/p8_h128_fc1.py"
    ),
    "b12x.moe._shared.kernels.p8_small_m": (
        "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels/p8_small_m.py"
    ),
    "b12x.moe._shared.kernels.p8_coupled_topk": (
        "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels/p8_coupled_topk.py"
    ),
}
PROTOCOL = {
    "schema": "glm53.p8-mixed-rate-m1-device-closure.v1",
    "evidence_level": "gpu-smoke-numerical-closure",
    "product": "P8 K3/K4/K5 procedural MCG alpha2 to E4M3/UE8M0-K32 (rate from --bits)",
    "geometry": {
        "layer": "from --layer; every routed layer has identical geometry",
        "rank": RANK,
        "tokens": 1,
        "experts": 288,
        "hidden": 4096,
        "local_intermediate": 512,
        "topk": 8,
        "fc1_tile_n": 128,
        "small_m": True,
    },
    "seed": SEED,
    "expert_ids": list(EXPERT_IDS),
    "repetitions": 5,
    "exact_gates": [
        "all runtime source hashes equal the externally pinned image manifest",
        "sidecar/design/transform hashes and full-coupled schema agree",
        "selected-rate trellis and UE8M0 weight bytes decode as finite E4M3",
        "signed non-unit suh/svh are present in all five scale roles",
        "dispatch is M1 small_m materialized N128 and full_coupled is true",
        "input E4M3 payload and every UE8M0-K32 byte equal CPU reference",
        "each route down-input E4M3 payload and UE8M0-K32 byte equals reference",
        "five eager final outputs are bitwise identical",
    ],
    "numeric_gates": {
        "route_cosine_strictly_greater_than": 0.995,
        "route_relative_l2_strictly_less_than": 0.12,
        "final_cosine_strictly_greater_than": 0.995,
        "final_relative_l2_strictly_less_than": 0.12,
    },
    "failure_policy": (
        "any missing debug buffer, fallback/identity dispatch, byte mismatch, "
        "hash drift, nonfinite value, or failed numerical gate is a closure failure"
    ),
    "claim_boundary": (
        "one synthetic single-layer rank-0 M1 TP-local device closure at the requested rate on the "
        "layer named by --layer; not KLD, "
        "prefill, throughput, serving, all-rank, or full-model qualification"
    ),
    "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    "ldlq": False,
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes_sha256(value) -> str:
    import torch

    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def permute_k32_payload(payload):
    """Apply the exact B12X trellis-A permutation within every K32 block."""
    import torch

    if payload.dtype != torch.uint8 or payload.shape[-1] % 32:
        raise ValueError("E4M3 payload must be uint8 with a K32-aligned last axis")
    index = torch.tensor(K32_PERM, dtype=torch.long, device=payload.device)
    blocks = payload.reshape(*payload.shape[:-1], payload.shape[-1] // 32, 32)
    return blocks.index_select(-1, index).reshape_as(payload).contiguous()


def unpack_materialized_intermediate(intermediate_u32, physical_rows: Sequence[int]):
    """Read the wrapper's [payload rows][slice-major scale words] ABI."""
    import torch

    if intermediate_u32.dtype != torch.int32 or intermediate_u32.ndim != 1:
        raise ValueError("intermediate debug carrier must be flat int32")
    row_stride_words = 128 + 4  # 512 E4M3 bytes + sixteen UE8M0 bytes.
    if intermediate_u32.numel() % row_stride_words:
        raise ValueError("intermediate carrier size does not encode full rows")
    rows_capacity = intermediate_u32.numel() // row_stride_words
    if any(row < 0 or row >= rows_capacity for row in physical_rows):
        raise ValueError("physical row is outside intermediate carrier")
    cpu = intermediate_u32.detach().cpu().contiguous()
    payload_words = rows_capacity * 128
    payload = cpu[:payload_words].view(torch.uint8).reshape(rows_capacity, 512)
    scale_words = cpu[payload_words:]
    payload_out = payload.index_select(0, torch.tensor(physical_rows)).contiguous()
    scale_rows = []
    for row in physical_rows:
        words = torch.stack(
            [scale_words[slice_id * rows_capacity + row] for slice_id in range(4)]
        ).contiguous()
        scale_rows.append(words.view(torch.uint8))
    return payload_out, torch.stack(scale_rows).contiguous()


def _metric(actual, expected) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    actual = actual.float()
    expected = expected.float()
    return {
        "cosine": float(
            functional.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1))
        ),
        "relative_l2": float(
            (actual - expected).norm() / expected.norm().clamp_min(1e-9)
        ),
        "max_abs": float((actual - expected).abs().max()),
    }


def _numeric_pass(metric: dict[str, float], arm: str) -> bool:
    if arm not in {"route", "final"}:
        raise ValueError(f"unknown numerical arm: {arm}")
    gates = PROTOCOL["numeric_gates"]
    return (
        math.isfinite(metric["cosine"])
        and math.isfinite(metric["relative_l2"])
        and metric["cosine"] > gates[f"{arm}_cosine_strictly_greater_than"]
        and metric["relative_l2"] < gates[f"{arm}_relative_l2_strictly_less_than"]
    )


def _scale_evidence(name: str, value) -> dict[str, object]:
    import torch

    flat = value.float().reshape(-1)
    finite = bool(torch.isfinite(flat).all())
    nonzero = bool((flat != 0).all())
    positive = bool((flat > 0).any())
    negative = bool((flat < 0).any())
    nonunit = bool((flat != 1).any())
    result = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _tensor_bytes_sha256(value),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "finite": finite,
        "nonzero": nonzero,
        "has_positive": positive,
        "has_negative": negative,
        "has_nonunit": nonunit,
    }
    if not all((finite, nonzero, positive, negative, nonunit)):
        raise RuntimeError(f"scale role is not signed finite non-unit: {result}")
    return result


def _read_selected(handle, name: str, key):
    """Use safetensors slicing only; never fall back to loading a full weight plane."""
    try:
        return handle.get_slice(name)[key].contiguous()
    except BaseException as error:
        raise RuntimeError(f"slice-only read unavailable for {name}: {error}") from error


def _load_sidecar_reference(path: Path, transform_sha256: str, bits: int, layer: int):
    import torch
    from safetensors import safe_open
    from p8_coupled_scales import SCALE_NAMES, validate_coupled_component

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        required = {
            "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
            "layer": str(layer),
            "rank": str(RANK),
            "world_size": "4",
            "bits": str(bits),
            "alphabet": "e4m3",
            "scale": "ue8m0-k32",
            "law": "procedural-mcg-alpha2",
            "boundary": "coupled-h512-h128-suh-svh-v1",
            "full_coupled": "true",
            "activation": "silu-cap10",
            "ldlq": "false",
            "fc1_trellis_slot_order": "gate-up",
            "fc1_scale_plane_order": "up-gate",
        }
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"sidecar full-coupled contract mismatch: {mismatches}")
        if metadata.get("encoder_transform_sha256") != transform_sha256:
            raise RuntimeError("sidecar transform hash does not match pinned transform")
        scale_tensors = {name: handle.get_tensor(name) for name in SCALE_NAMES}
        scales = validate_coupled_component(
            metadata,
            scale_tensors,
            layer=layer,
            rank=RANK,
            experts=288,
            hidden=4096,
            intermediate=512,
            expected_transform_sha256=transform_sha256,
        )
        gate_svh, up_svh, down_suh = scales.split_intermediate()
        scale_receipts = [
            _scale_evidence("gate_up_suh_fp16", scales.gate_up_suh),
            _scale_evidence("gate_svh_fp16[selected]", gate_svh[list(EXPERT_IDS)]),
            _scale_evidence("up_svh_fp16[selected]", up_svh[list(EXPERT_IDS)]),
            _scale_evidence("down_suh_fp16[selected]", down_suh[list(EXPERT_IDS)]),
            _scale_evidence("down_svh_fp16", scales.down_svh),
        ]
    return metadata, scales, scale_receipts


def _decode_expert(path: Path, expert: int, bits: int):
    import torch
    from safetensors import safe_open
    from glm53_nvfp4.trellis_mxf import (
        decode_trellis_mxf,
        state_lut,
        unpack_ue8m0,
    )

    lut = state_lut(
        bits, alphabet="e4m3", law="mcg", compander_scale=2.0, device="cpu"
    )
    with safe_open(path, framework="pt", device="cpu") as handle:
        w13 = _read_selected(handle, "w13_trellis", (slice(None), expert))
        w2 = _read_selected(handle, "w2_trellis", expert)
        w13_scale = _read_selected(handle, "w13_scale_ue8m0", expert)
        w2_scale = _read_selected(handle, "w2_scale_ue8m0", expert)
    if tuple(w13.shape) != (2, 256, 32, 16 * bits) or tuple(w2.shape) != (32, 256, 16 * bits):
        raise RuntimeError(f"selected K{bits} trellis geometry differs for expert {expert}")
    if tuple(w13_scale.shape) != (1024, 128) or tuple(w2_scale.shape) != (4096, 16):
        raise RuntimeError(f"selected UE8M0 geometry differs for expert {expert}")
    if w13.dtype != torch.int16 or w2.dtype != torch.int16:
        raise RuntimeError(f"K{bits} trellis stream must use packed int16 words")
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise RuntimeError("weight scale planes must contain raw UE8M0 bytes")
    if bool((w13_scale == 255).any()) or bool((w2_scale == 255).any()):
        raise RuntimeError("UE8M0 contains reserved 255 code")
    if not bool(((w13_scale != 127).any()) and ((w2_scale != 127).any())):
        raise RuntimeError("selected expert does not exercise non-unit weight scales")

    # W13 trellis slots are gate/up; its physical SFB plane is up/gate.
    gate = decode_trellis_mxf(
        w13[0], lut, w13_scale[512:], bits=bits, block_size=32,
        rows=512, width=4096, device="cpu",
    )
    up = decode_trellis_mxf(
        w13[1], lut, w13_scale[:512], bits=bits, block_size=32,
        rows=512, width=4096, device="cpu",
    )
    down = decode_trellis_mxf(
        w2, lut, w2_scale, bits=bits, block_size=32,
        rows=4096, width=512, device="cpu",
    )
    if not all(bool(torch.isfinite(value).all()) for value in (gate, up, down)):
        raise RuntimeError("decoded selected weight contains nonfinite value")
    normalized_payload_hashes = {}
    for name, decoded, codes in (
        ("gate", gate, w13_scale[512:]),
        ("up", up, w13_scale[:512]),
        ("down", down, w2_scale),
    ):
        scale = unpack_ue8m0(codes).unsqueeze(-1)
        blocks = decoded.reshape(decoded.shape[0], -1, 32)
        normalized = blocks / scale
        payload = normalized.to(torch.float8_e4m3fn)
        reconstructed = payload.float() * scale
        if not torch.equal(reconstructed, blocks):
            raise RuntimeError(f"{name} decode is not exact E4M3 times UE8M0")
        normalized_payload_hashes[name] = _tensor_bytes_sha256(payload.view(torch.uint8))
    receipt = {
        "expert": expert,
        "w13_trellis_sha256": _tensor_bytes_sha256(w13),
        "w2_trellis_sha256": _tensor_bytes_sha256(w2),
        "w13_scale_ue8m0_sha256": _tensor_bytes_sha256(w13_scale),
        "w2_scale_ue8m0_sha256": _tensor_bytes_sha256(w2_scale),
        "decoded_gate_sha256": _tensor_bytes_sha256(gate),
        "decoded_up_sha256": _tensor_bytes_sha256(up),
        "decoded_down_sha256": _tensor_bytes_sha256(down),
        "normalized_e4m3_payload_sha256": normalized_payload_hashes,
        "w13_scale_min": int(w13_scale.min()),
        "w13_scale_max": int(w13_scale.max()),
        "w2_scale_min": int(w2_scale.min()),
        "w2_scale_max": int(w2_scale.max()),
    }
    return gate, up, down, receipt


def packed_quantizer_payload(payload, rows: int, width: int):
    """Convert only the quantizer's [rows, K/32, 32] layout to wire rows.

    No sorting, numeric conversion, or tolerance: retain every permuted byte.
    Reject other shapes so a routing/layout defect cannot pass by flattening.
    """
    if tuple(payload.shape) != (rows, width // 32, 32) or width % 32:
        raise ValueError("expected blocked quantizer payload [rows, K/32, 32]")
    return permute_k32_payload(payload).reshape(rows, width)


def _reference(path: Path, x, weights, scales, bits: int):
    import torch
    from p8_coupled_scales import hadamard_blocks, quantize_e4m3_ue8m0_per32

    gate_svh, up_svh, down_suh = scales.split_intermediate()
    pre_sign, post_sign = scales.split_signs()
    source = hadamard_blocks(x.to(torch.float16).float(), 512)
    source = hadamard_blocks(source * scales.gate_up_suh.float(), 128)
    input_payload, input_sf, source_q = quantize_e4m3_ue8m0_per32(source)
    raw_payload, raw_sf, _ = quantize_e4m3_ue8m0_per32(x.float())
    if torch.equal(input_payload, raw_payload) and torch.equal(input_sf, raw_sf):
        raise RuntimeError("coupled input transformed to identity carrier")

    routes = []
    middle_payloads = []
    middle_scales = []
    weight_receipts = []
    for slot, expert in enumerate(EXPERT_IDS):
        gate, up, down, receipt = _decode_expert(path, expert, bits)
        gp = (source_q @ gate.T).to(torch.float16)
        uph = (source_q @ up.T).to(torch.float16)
        raw = torch.stack(
            (gp.view(1, 16, 32), uph.view(1, 16, 32)), dim=2
        ).reshape(1, 1024)
        raw_scale = torch.stack(
            (
                gate_svh[expert].view(16, 32),
                up_svh[expert].view(16, 32),
            ),
            dim=1,
        ).reshape(1024)
        work = hadamard_blocks(raw.float(), 128)
        work = hadamard_blocks(work * raw_scale.float(), 128)
        work = work * pre_sign.float()
        g, u = work[:, 0::2], work[:, 1::2]
        activated = g.clamp(max=10.0) * torch.sigmoid(g.clamp(max=10.0))
        activated = activated * u.clamp(-10.0, 10.0)
        down_input = hadamard_blocks(activated * post_sign.float(), 128)
        down_input = hadamard_blocks(
            down_input * down_suh[expert].float(), 128
        )
        payload, sf, down_q = quantize_e4m3_ue8m0_per32(down_input)
        plain_payload, plain_sf, _ = quantize_e4m3_ue8m0_per32(activated)
        if torch.equal(payload, plain_payload) and torch.equal(sf, plain_sf):
            raise RuntimeError(f"expert {expert} down boundary collapsed to identity")
        physical = (down_q @ down.T).to(torch.float16)
        route = hadamard_blocks(physical.float(), 128) * scales.down_svh.float()
        routes.append(route.squeeze(0))
        middle_payloads.append(packed_quantizer_payload(payload, 1, 512).squeeze(0))
        middle_scales.append(sf.squeeze(0))
        weight_receipts.append(receipt)
        del gate, up, down
    route_tensor = torch.stack(routes)
    mixed = (route_tensor * weights.reshape(-1, 1)).sum(0, keepdim=True)
    final = hadamard_blocks(mixed, 512).to(torch.bfloat16)
    return {
        "input_payload": packed_quantizer_payload(input_payload, 1, 4096),
        "input_scale": input_sf,
        "middle_payload": torch.stack(middle_payloads),
        "middle_scale": torch.stack(middle_scales),
        "routes": route_tensor,
        "final": final,
        "weight_receipts": weight_receipts,
    }


def _verify_runtime_sources(manifest_path: Path, expected_sha256: str):
    if sha256_file(manifest_path) != expected_sha256:
        raise RuntimeError("installed runtime manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    expected_sources = manifest.get("source_sha256", {})
    receipts = {}
    for module_name, source_key in MODULE_SOURCES.items():
        expected = expected_sources.get(source_key)
        if not isinstance(expected, str):
            raise RuntimeError(f"runtime manifest lacks {source_key}")
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"executed source mismatch for {module_name}: expected {expected}, got {actual}"
            )
        receipts[module_name] = {
            "path": str(path), "manifest_key": source_key, "sha256": actual
        }
    return receipts


def _save_carrier_failure(directory: Path, pairs, extras) -> dict[str, object]:
    """Preserve bounded diagnostic data without changing any acceptance gate."""
    import numpy as np
    import torch

    arrays = {}
    summary = {}
    for name, (actual, expected) in pairs.items():
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        if actual.numel() != expected.numel():
            raise RuntimeError(f"diagnostic element-count mismatch for {name}")
        indices = torch.nonzero(actual.reshape(-1) != expected.reshape(-1)).flatten()
        sample = indices[:32]
        summary[name] = {
            "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
            "elements": actual.numel(), "mismatches": indices.numel(),
            "first_flat_indices": sample.tolist(),
            "actual_at_first": actual.reshape(-1)[sample].tolist(),
            "expected_at_first": expected.reshape(-1)[sample].tolist(),
        }
        arrays[name + "_actual"] = actual.numpy()
        arrays[name + "_expected"] = expected.numpy()
    for name, value in extras.items():
        arrays[name] = value.detach().float().cpu().numpy()
    if sum(value.nbytes for value in arrays.values()) > 4 * 1024 * 1024:
        raise RuntimeError("carrier diagnostic exceeds frozen 4 MiB limit")
    npz = directory / "carrier-failure.npz"
    with npz.open("xb") as stream:
        np.savez(stream, **arrays)
    record = {"schema": "glm53.p8-carrier-failure-diagnostic.v1",
              "protocol_sha256": PROTOCOL_SHA256, "gate_changed": False,
              "carriers": summary, "npz_sha256": sha256_file(npz)}
    with (directory / "carrier-failure.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from p8_native_kernel import P8NativeTPMoE

    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_id):
        raise RuntimeError("probe requires the outer launcher's immutable image ID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("closure container requires exactly one visible CUDA device")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("closure requires SM120")
    identities = {
        "sidecar": sha256_file(args.sidecar),
        "design": sha256_file(args.design),
        "transform": sha256_file(args.transform),
        "runtime_manifest": sha256_file(args.runtime_manifest),
    }
    expected = {
        "sidecar": args.sidecar_sha256,
        "design": args.design_sha256,
        "transform": args.transform_sha256,
        "runtime_manifest": args.runtime_manifest_sha256,
    }
    if identities != expected:
        raise RuntimeError(f"input identity mismatch: expected {expected}, got {identities}")
    sources = _verify_runtime_sources(
        args.runtime_manifest, args.runtime_manifest_sha256
    )
    metadata, scales, scale_receipts = _load_sidecar_reference(
        args.sidecar, args.transform_sha256, args.bits, args.layer
    )
    if metadata.get("source_design_sha256") != args.design_sha256:
        raise RuntimeError("sidecar source design hash differs")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x_cpu = (torch.randn(1, 4096, generator=generator) * 0.03125).to(torch.bfloat16)
    logits = torch.randn(8, generator=generator)
    weights_cpu = torch.softmax(logits, 0).float()
    ids_cpu = torch.tensor(EXPERT_IDS, dtype=torch.int32).reshape(1, 8)
    reference = _reference(args.sidecar, x_cpu, weights_cpu, scales, args.bits)

    runtime = P8NativeTPMoE(
        args.sidecar,
        device=torch.device("cuda"),
        tp_rank=RANK,
        layer=args.layer,
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
        fuse_scratch_zero=False,
    )
    if not runtime.full_coupled or runtime.scale_component is None:
        raise RuntimeError("actual wrapper did not resolve full-coupled mode")
    if runtime.trellis_bits != args.bits:
        raise RuntimeError(f"wrapper resolved K{runtime.trellis_bits}, expected K{args.bits}")

    x = x_cpu.cuda()
    weights = weights_cpu.reshape(1, 8).cuda()
    ids = ids_cpu.cuda()
    runs = []
    output_hashes = []
    exact_debug_runs = []
    for repeat in range(5):
        output = runtime(x, weights, ids)
        torch.cuda.synchronize()
        if runtime.debug_dispatch != {
            "small_m": True,
            "materialized": True,
            "fused_scratch_zero": False,
            "fc1_tile_n": 128,
            "tile_m": 16,
        }:
            raise RuntimeError(f"identity/fallback dispatch observed: {runtime.debug_dispatch}")
        if set(runtime.debug_tensors) != REQUIRED_DEBUG:
            raise RuntimeError(
                f"required device intermediates inaccessible: {set(runtime.debug_tensors)}"
            )
        debug = runtime.debug_tensors
        input_payload = debug["packed_a"].view(torch.uint8)[:4096].cpu().reshape(1, 4096)
        input_sf = debug["scale_flat"].view(torch.uint8)[:128].cpu().reshape(1, 128)
        physical_rows = [slot * 16 for slot in range(8)]
        middle_payload, middle_sf = unpack_materialized_intermediate(
            debug["intermediate_u32"], physical_rows
        )
        route_actual = debug["route_output"].float().cpu()
        final_actual = output.cpu()
        exact = {
            "input_payload": torch.equal(input_payload, reference["input_payload"]),
            "input_scale": torch.equal(input_sf, reference["input_scale"]),
            "middle_payload": torch.equal(middle_payload, reference["middle_payload"]),
            "middle_scale": torch.equal(middle_sf, reference["middle_scale"]),
        }
        if not all(exact.values()):
            _save_carrier_failure(args.output.parent, {
                "input_payload": (input_payload, reference["input_payload"]),
                "input_scale": (input_sf, reference["input_scale"]),
                "middle_payload": (middle_payload, reference["middle_payload"]),
                "middle_scale": (middle_sf, reference["middle_scale"]),
            }, {"input": x_cpu, "routes_actual": route_actual,
                "routes_expected": reference["routes"], "final_actual": final_actual,
                "final_expected": reference["final"],
                "token_map": debug["token_map"], "row_counts": debug["row_counts"],
                "expert_tile_base": debug["expert_tile_base"]})
            raise RuntimeError(f"observable activation carrier byte mismatch: {exact}")
        route_metric = _metric(route_actual, reference["routes"])
        final_metric = _metric(final_actual, reference["final"])
        if not _numeric_pass(route_metric, "route") or not _numeric_pass(
            final_metric, "final"
        ):
            raise RuntimeError(
                f"numerical output closure failed: route={route_metric}, final={final_metric}"
            )
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(route_actual).all()):
            raise RuntimeError("device output contains nonfinite value")
        output_hash = _tensor_bytes_sha256(final_actual)
        output_hashes.append(output_hash)
        exact_debug_runs.append(
            {
                "input_payload": _tensor_bytes_sha256(input_payload),
                "input_scale": _tensor_bytes_sha256(input_sf),
                "middle_payload": _tensor_bytes_sha256(middle_payload),
                "middle_scale": _tensor_bytes_sha256(middle_sf),
            }
        )
        runs.append(
            {
                "repeat": repeat,
                "output_sha256": output_hash,
                "route_metric": route_metric,
                "final_metric": final_metric,
                "exact_activation_carriers": exact,
            }
        )
    if len(set(output_hashes)) != 1 or any(
        item != exact_debug_runs[0] for item in exact_debug_runs
    ):
        raise RuntimeError("five-run bitwise determinism gate failed")

    return {
        "schema": PROTOCOL["schema"],
        "decision": "pass",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image_id,
        "bits": args.bits,
        "identities": identities,
        "executed_sources": sources,
        "scale_receipts": scale_receipts,
        "payload": {
            "x_sha256": _tensor_bytes_sha256(x_cpu),
            "topk_ids_sha256": _tensor_bytes_sha256(ids_cpu),
            "topk_weights_sha256": _tensor_bytes_sha256(weights_cpu),
        },
        "reference": {
            "input_payload_sha256": _tensor_bytes_sha256(reference["input_payload"]),
            "input_scale_sha256": _tensor_bytes_sha256(reference["input_scale"]),
            "middle_payload_sha256": _tensor_bytes_sha256(reference["middle_payload"]),
            "middle_scale_sha256": _tensor_bytes_sha256(reference["middle_scale"]),
            "route_output_sha256": _tensor_bytes_sha256(reference["routes"]),
            "final_output_sha256": _tensor_bytes_sha256(reference["final"]),
            "selected_weight_decode": reference["weight_receipts"],
        },
        "runs": runs,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_used": True,
        "kld_tested": False,
        "throughput_tested": False,
    }


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def probe_command(args, repo: Path) -> list[str]:
    """Argv for the in-container probe.

    Every probe-relevant option the outer parser accepts must be forwarded here; a flag
    added to the parser but not to this list silently runs the probe with its default.
    """
    return [
        "docker", "run", "--rm", "--gpus", f"device={args.gpu_device}",
        "--network=none", "--ipc=private", "--shm-size=1g",
        "-e", "PYTHONPATH=/opt/p8-coupled-runtime:/work",
        "-e", "OMP_NUM_THREADS=2",
        "-e", "GLM53_P8_NATIVE=", "-e", "GLM53_P4_NATIVE=",
        "-v", f"{repo}:/work:ro",
        "-v", f"{args.sidecar}:/inputs/sidecar.safetensors:ro",
        "-v", f"{args.design}:/inputs/design.json:ro",
        "-v", f"{args.transform}:/inputs/transform.json:ro",
        "-v", f"{args.output}:/out:rw",
        "--entrypoint", "/opt/venv/bin/python", args.image,
        "/work/scripts/run_p8_mixed_rate_m1_device_closure.py", "--probe",
        "--bits", str(args.bits),
        "--layer", str(args.layer),
        "--image-id", args.image,
        "--sidecar", "/inputs/sidecar.safetensors",
        "--sidecar-sha256", args.sidecar_sha256,
        "--design", "/inputs/design.json",
        "--design-sha256", args.design_sha256,
        "--transform", "/inputs/transform.json",
        "--transform-sha256", args.transform_sha256,
        "--runtime-manifest", "/opt/p8-coupled-runtime/image-manifest.json",
        "--runtime-manifest-sha256", args.runtime_manifest_sha256,
        "--output", "/out/result.json",
    ]


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
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    actual_image = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual_image != args.image:
        raise ValueError("local image identity differs")
    manifest_line = subprocess.check_output(
        [
            "docker", "run", "--rm", "--network=none", "--runtime=runc",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum",
            args.image, "/opt/p8-coupled-runtime/image-manifest.json",
        ],
        text=True,
    ).strip()
    if manifest_line.split()[0] != args.runtime_manifest_sha256:
        raise ValueError("pinned installed runtime manifest differs")
    active = subprocess.check_output(
        [
            "nvidia-smi", "-i", args.gpu_device, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if active:
        raise RuntimeError("selected closure GPU is not idle; no service action is authorized")
    temperature = int(
        subprocess.check_output(
            [
                "nvidia-smi", "-i", args.gpu_device, "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if temperature > 75:
        raise RuntimeError(f"startup temperature {temperature} C exceeds 75 C")

    args.output.mkdir(mode=0o700, parents=True)
    repo = Path(__file__).resolve().parents[1]
    command = probe_command(args, repo)
    launch = {
        "schema": "glm53.p8-mixed-rate-m1-device-launch.v1",
        "bits": args.bits,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image,
        "gpu_device": args.gpu_device,
        "startup_temperature_c": temperature,
        "command": command,
        "services_modified": [],
        "started_unix_ns": time.time_ns(),
    }
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    completed = _run(command, check=False)
    (args.output / "probe.stdout.log").write_text(completed.stdout)
    (args.output / "probe.stderr.log").write_text(completed.stderr)
    launch["completed_unix_ns"] = time.time_ns()
    launch["exit_code"] = completed.returncode
    launch["result_present"] = (args.output / "result.json").is_file()
    (args.output / "execution.json").write_text(json.dumps(launch, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError("coupled M1 closure failed; preserved logs and receipt")
    result = json.loads((args.output / "result.json").read_text())
    if result.get("decision") != "pass" or result.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("probe result did not satisfy the frozen protocol")
    print(json.dumps({"decision": "pass", "result": str(args.output / "result.json")}))


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
    parser.add_argument("--bits", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--layer", type=int, choices=range(3, 45), default=DEFAULT_LAYER,
                        help="routed layer whose rank-0 sidecar is closed; every routed layer has "
                             "identical geometry, so this selects which candidate sidecar to test")
    return parser.parse_args(argv)


def _require(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    for name in (
        "sidecar_sha256", "design_sha256", "transform_sha256",
        "runtime_manifest_sha256",
    ):
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
    common = (
        "sidecar", "sidecar_sha256", "design", "design_sha256", "transform",
        "transform_sha256", "runtime_manifest_sha256", "output",
    )
    if args.probe:
        _require(args, (*common, "image_id", "runtime_manifest"))
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        try:
            result = run_probe(args)
        except BaseException as error:
            result = {
                "schema": PROTOCOL["schema"],
                "decision": "fail",
                "protocol": PROTOCOL,
                "protocol_sha256": PROTOCOL_SHA256,
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
                "gpu_used": True,
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
