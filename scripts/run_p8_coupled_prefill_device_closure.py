#!/usr/bin/env python3
"""Opt-in SM120 device closure for coupled P8 M64/N128 prefill.

The default invocation prints only the frozen protocol. ``--execute`` is the
only public device entry point; the private ``--probe`` mode runs inside the
immutable, network-disabled image selected by the outer process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
M1_SCRIPT = Path(__file__).with_name("run_p8_coupled_m1_device_closure.py")


def _load_m1_helpers():
    spec = importlib.util.spec_from_file_location("p8_coupled_m1_helpers", M1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned M1 closure helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1 = _load_m1_helpers()

SEED = 20260905130
LAYER = 3
RANK = 0
EXPERT_IDS = M1.EXPERT_IDS
M_CASES = (2, 64, 65)
PROTOTYPE_COUNT = 2
REPETITIONS = 5
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_SCRATCH_UPPER_BOUND_BYTES = 130 * 1024 * 1024
REQUIRED_DEBUG = M1.REQUIRED_DEBUG
MODULE_SOURCES = {
    **M1.MODULE_SOURCES,
    "p8_coupled_prefill_plan": "runtime_patch/p8_coupled_prefill_plan.py",
    "b12x.moe._shared.kernels.p8_coupled_prefill_fc1": (
        "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/"
        "p8_coupled_prefill_fc1.py"
    ),
    "b12x.moe._shared.kernels.p8_coupled_prefill_fc2": (
        "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/"
        "p8_coupled_prefill_fc2.py"
    ),
}

PROTOCOL = {
    "schema": "glm53.p8-full-coupled-prefill-device-closure.v1",
    "evidence_level": "gpu-smoke-numerical-closure",
    "product": "P8 K4 procedural MCG alpha2 to E4M3/UE8M0-K32",
    "geometry": {
        "layer": LAYER,
        "rank": RANK,
        "m_cases": list(M_CASES),
        "experts": 288,
        "hidden": 4096,
        "local_intermediate": 512,
        "topk": 8,
        "tile_m": 64,
        "tile_n": 128,
        "input_prototypes": PROTOTYPE_COUNT,
    },
    "case_roles": {
        "M2": "smallest supported M>1; eight active partial expert tiles",
        "M64": "exact M64 boundary for each selected expert",
        "M65": "one complete M64 tile plus one-row tail for each selected expert",
    },
    "seed": SEED,
    "expert_ids": list(EXPERT_IDS),
    "repetitions_per_case": REPETITIONS,
    "routing": (
        "every token routes once to each of the same eight distinct experts; "
        "token inputs alternate between two frozen BF16 prototypes"
    ),
    "exact_gates": [
        "all imported runtime source hashes equal the installed image manifest",
        "sidecar/design/transform hashes and full-coupled schema agree",
        "actual P8NativeTPMoE call dispatches small_m=false materialized=true M64/N128",
        "row_counts, expert_tile_base and token_map bijectively map every route",
        "input E4M3 payload and every UE8M0-K32 byte equal CPU reference",
        "every routed down-input E4M3 payload and UE8M0-K32 byte equals reference",
        "five same-input runs preserve every carrier and output hash per M case",
    ],
    "numeric_gates": dict(M1.PROTOCOL["numeric_gates"]),
    "m1_protocol_sha256": M1.PROTOCOL_SHA256,
    "failure_policy": (
        "any hash drift, missing debug carrier, fallback/class-only evidence, "
        "schedule mismatch, byte mismatch, nonfinite value, or failed numeric "
        "gate is a closure failure; no gate may be weakened after execution"
    ),
    "storage": {
        "fixture": "reuse one existing streamed full-rank synthetic sidecar read-only",
        "new_fixture_bytes": 0,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
        "max_wrapper_scratch_upper_bound_bytes": MAX_SCRATCH_UPPER_BOUND_BYTES,
    },
    "claim_boundary": (
        "one synthetic layer-3/rank-0 TP-local M2/M64/M65 eager device closure; "
        "not KLD, throughput, CUDA-graph, serving, all-rank, or full-model qualification"
    ),
    "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    "ldlq": False,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def _expected_dispatch() -> dict[str, object]:
    return {
        "small_m": False,
        "materialized": True,
        "fused_scratch_zero": False,
        "fc1_tile_n": 128,
        "tile_m": 64,
    }


def _make_inputs():
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    prototypes = (
        torch.randn(PROTOTYPE_COUNT, 4096, generator=generator) * 0.03125
    ).to(torch.bfloat16)
    logits = torch.randn(max(M_CASES), 8, generator=generator)
    weights = torch.softmax(logits, dim=1).float()
    ids = torch.tensor(EXPERT_IDS, dtype=torch.int32).reshape(1, 8)
    return prototypes, weights, ids


def _case_inputs(m: int, prototypes, all_weights, ids):
    import torch

    if m not in M_CASES:
        raise ValueError(f"unregistered M case: {m}")
    prototype_index = torch.arange(m, dtype=torch.long) % PROTOTYPE_COUNT
    x = prototypes.index_select(0, prototype_index).contiguous()
    weights = all_weights[:m].contiguous()
    topk_ids = ids.expand(m, -1).contiguous()
    return x, weights, topk_ids, prototype_index


def map_route_order_physical_rows(
    row_counts,
    expert_tile_base,
    token_map,
    flat_ids,
    *,
    tile_m: int = 64,
) -> tuple[list[int], dict[str, object]]:
    """Validate grouped routing and return physical rows in route-output order."""
    import torch

    row_counts = row_counts.detach().cpu().to(torch.int64).reshape(-1)
    expert_tile_base = expert_tile_base.detach().cpu().to(torch.int64).reshape(-1)
    token_map = token_map.detach().cpu().to(torch.int64).reshape(-1)
    flat_ids = flat_ids.detach().cpu().to(torch.int64).reshape(-1)
    experts = row_counts.numel()
    if experts != 288 or expert_tile_base.numel() != experts + 1:
        raise RuntimeError("grouped schedule does not expose E288 prefix geometry")
    expected_counts = torch.bincount(flat_ids, minlength=experts)
    if not torch.equal(row_counts, expected_counts):
        raise RuntimeError("device row_counts differ from supplied route IDs")
    if int(expert_tile_base[0]) != 0 or bool(
        (expert_tile_base[1:] < expert_tile_base[:-1]).any()
    ):
        raise RuntimeError("expert tile prefix is not monotone from zero")

    route_to_physical: dict[int, int] = {}
    active_spans = {}
    for expert in range(experts):
        count = int(row_counts[expert])
        begin_tile = int(expert_tile_base[expert])
        end_tile = int(expert_tile_base[expert + 1])
        expected_tiles = (count + tile_m - 1) // tile_m
        if end_tile - begin_tile != expected_tiles:
            raise RuntimeError(
                f"expert {expert} tile span differs: {end_tile-begin_tile} != {expected_tiles}"
            )
        if count:
            active_spans[str(expert)] = {
                "rows": count,
                "tiles": expected_tiles,
                "partial_rows": count % tile_m,
            }
        for local_row in range(count):
            physical_row = begin_tile * tile_m + local_row
            if physical_row >= token_map.numel():
                raise RuntimeError("active physical row exceeds token_map")
            route = int(token_map[physical_row])
            if route < 0 or route >= flat_ids.numel():
                raise RuntimeError("token_map points outside supplied routes")
            if int(flat_ids[route]) != expert:
                raise RuntimeError("token_map route belongs to a different expert")
            if route in route_to_physical:
                raise RuntimeError("token_map is not injective over active rows")
            route_to_physical[route] = physical_row
    if set(route_to_physical) != set(range(flat_ids.numel())):
        raise RuntimeError("token_map is not surjective over supplied routes")
    physical = [route_to_physical[route] for route in range(flat_ids.numel())]
    return physical, {
        "routes": flat_ids.numel(),
        "active_experts": int((row_counts != 0).sum()),
        "physical_tiles_used": int(expert_tile_base[-1]),
        "active_spans": active_spans,
        "route_to_physical_sha256": hashlib.sha256(
            torch.tensor(physical, dtype=torch.int32).numpy().tobytes()
        ).hexdigest(),
    }


def _reference_prototypes(sidecar: Path, prototypes, scales):
    """Decode eight experts once and evaluate only two unique input rows."""
    import torch
    from p8_coupled_scales import hadamard_blocks, quantize_e4m3_ue8m0_per32

    gate_svh, up_svh, down_suh = scales.split_intermediate()
    pre_sign, post_sign = scales.split_signs()
    source = hadamard_blocks(prototypes.to(torch.float16).float(), 512)
    source = hadamard_blocks(source * scales.gate_up_suh.float(), 128)
    input_payload, input_sf, source_q = quantize_e4m3_ue8m0_per32(source)
    raw_payload, raw_sf, _ = quantize_e4m3_ue8m0_per32(prototypes.float())
    if torch.equal(input_payload, raw_payload) and torch.equal(input_sf, raw_sf):
        raise RuntimeError("coupled prefill input collapsed to identity carriers")

    route_cache = torch.empty(
        PROTOTYPE_COUNT, len(EXPERT_IDS), 4096, dtype=torch.float32
    )
    middle_payload_cache = torch.empty(
        PROTOTYPE_COUNT, len(EXPERT_IDS), 512, dtype=torch.uint8
    )
    middle_scale_cache = torch.empty(
        PROTOTYPE_COUNT, len(EXPERT_IDS), 16, dtype=torch.uint8
    )
    weight_receipts = []
    for slot, expert in enumerate(EXPERT_IDS):
        gate, up, down, receipt = M1._decode_expert(sidecar, expert)
        gp = (source_q @ gate.T).to(torch.float16)
        uph = (source_q @ up.T).to(torch.float16)
        raw = torch.stack(
            (gp.view(PROTOTYPE_COUNT, 16, 32), uph.view(PROTOTYPE_COUNT, 16, 32)),
            dim=2,
        ).reshape(PROTOTYPE_COUNT, 1024)
        raw_scale = torch.stack(
            (gate_svh[expert].view(16, 32), up_svh[expert].view(16, 32)),
            dim=1,
        ).reshape(1024)
        work = hadamard_blocks(raw.float(), 128)
        work = hadamard_blocks(work * raw_scale.float(), 128)
        work = work * pre_sign.float()
        gate_work, up_work = work[:, 0::2], work[:, 1::2]
        capped_gate = gate_work.clamp(max=10.0)
        activated = capped_gate * torch.sigmoid(capped_gate)
        activated = activated * up_work.clamp(-10.0, 10.0)
        down_input = hadamard_blocks(activated * post_sign.float(), 128)
        down_input = hadamard_blocks(down_input * down_suh[expert].float(), 128)
        payload, sf, down_q = quantize_e4m3_ue8m0_per32(down_input)
        plain_payload, plain_sf, _ = quantize_e4m3_ue8m0_per32(activated)
        if torch.equal(payload, plain_payload) and torch.equal(sf, plain_sf):
            raise RuntimeError(f"expert {expert} down boundary collapsed to identity")
        physical = (down_q @ down.T).to(torch.float16)
        route = hadamard_blocks(physical.float(), 128) * scales.down_svh.float()
        route_cache[:, slot] = route
        middle_payload_cache[:, slot] = M1.permute_k32_payload(payload)
        middle_scale_cache[:, slot] = sf
        weight_receipts.append(receipt)
        del gate, up, down, gp, uph, raw, work, down_q, physical, route
    return {
        "input_payload": M1.permute_k32_payload(input_payload),
        "input_scale": input_sf,
        "middle_payload": middle_payload_cache,
        "middle_scale": middle_scale_cache,
        "routes": route_cache,
        "weight_receipts": weight_receipts,
    }


def expand_reference_case(cache, prototype_index, weights):
    """Expand prototype results to route order and reduce numerically."""
    from p8_coupled_scales import hadamard_blocks

    input_payload = cache["input_payload"].index_select(0, prototype_index)
    input_scale = cache["input_scale"].index_select(0, prototype_index)
    middle_payload = cache["middle_payload"].index_select(0, prototype_index)
    middle_scale = cache["middle_scale"].index_select(0, prototype_index)
    routes = cache["routes"].index_select(0, prototype_index)
    mixed = (routes * weights.unsqueeze(-1)).sum(dim=1)
    final = hadamard_blocks(mixed, 512).to(__import__("torch").bfloat16)
    return {
        "input_payload": input_payload,
        "input_scale": input_scale,
        "middle_payload": middle_payload.reshape(-1, 512),
        "middle_scale": middle_scale.reshape(-1, 16),
        "routes": routes.reshape(-1, 4096),
        "final": final,
    }


def _verify_runtime_sources(manifest_path: Path, expected_sha256: str):
    if M1.sha256_file(manifest_path) != expected_sha256:
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
        actual = M1.sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"executed source mismatch for {module_name}: {actual} != {expected}"
            )
        receipts[module_name] = {
            "path": str(path), "manifest_key": source_key, "sha256": actual
        }
    return receipts


def _scratch_receipts() -> list[dict[str, int]]:
    from p8_coupled_prefill_plan import P8CoupledPrefillGeometry

    receipts = []
    for m in M_CASES:
        plan = P8CoupledPrefillGeometry(m)
        upper = plan.current_wrapper_scratch_upper_bound
        if upper > MAX_SCRATCH_UPPER_BOUND_BYTES:
            raise RuntimeError(f"M{m} scratch forecast exceeds frozen bound")
        receipts.append({
            "m": m,
            "route_rows": plan.route_rows,
            "rows_capacity": plan.rows_capacity,
            "upper_bound_bytes": upper,
        })
    return receipts


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
        "sidecar": M1.sha256_file(args.sidecar),
        "design": M1.sha256_file(args.design),
        "transform": M1.sha256_file(args.transform),
        "runtime_manifest": M1.sha256_file(args.runtime_manifest),
    }
    expected = {
        "sidecar": args.sidecar_sha256,
        "design": args.design_sha256,
        "transform": args.transform_sha256,
        "runtime_manifest": args.runtime_manifest_sha256,
    }
    if identities != expected:
        raise RuntimeError(f"input identity mismatch: expected {expected}, got {identities}")
    sources = _verify_runtime_sources(args.runtime_manifest, args.runtime_manifest_sha256)
    metadata, scales, scale_receipts = M1._load_sidecar_reference(
        args.sidecar, args.transform_sha256
    )
    if metadata.get("source_design_sha256") != args.design_sha256:
        raise RuntimeError("sidecar source design hash differs")
    scratch = _scratch_receipts()
    prototypes, all_weights, ids = _make_inputs()
    reference_cache = _reference_prototypes(args.sidecar, prototypes, scales)

    runtime = P8NativeTPMoE(
        args.sidecar,
        device=torch.device("cuda"),
        tp_rank=RANK,
        layer=LAYER,
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

    case_receipts = []
    for m in M_CASES:
        x_cpu, weights_cpu, ids_cpu, prototype_index = _case_inputs(
            m, prototypes, all_weights, ids
        )
        reference = expand_reference_case(reference_cache, prototype_index, weights_cpu)
        x = x_cpu.cuda()
        weights = weights_cpu.cuda()
        topk_ids = ids_cpu.cuda()
        run_receipts = []
        repeat_hashes = []
        for repeat in range(REPETITIONS):
            output = runtime(x, weights, topk_ids)
            torch.cuda.synchronize()
            if runtime.debug_dispatch != _expected_dispatch():
                raise RuntimeError(
                    f"fallback/class-only dispatch observed: {runtime.debug_dispatch}"
                )
            if set(runtime.debug_tensors) != REQUIRED_DEBUG:
                raise RuntimeError(
                    f"required device intermediates inaccessible: {set(runtime.debug_tensors)}"
                )
            debug = runtime.debug_tensors
            physical_rows, schedule = map_route_order_physical_rows(
                debug["row_counts"], debug["expert_tile_base"], debug["token_map"],
                ids_cpu.reshape(-1),
            )
            if schedule["active_experts"] != len(EXPERT_IDS):
                raise RuntimeError("grouped prefill did not exercise all selected experts")
            input_payload = debug["packed_a"].view(torch.uint8)[: m * 4096]
            input_payload = input_payload.cpu().reshape(m, 4096)
            input_sf = debug["scale_flat"].view(torch.uint8)[: m * 128]
            input_sf = input_sf.cpu().reshape(m, 128)
            middle_payload, middle_sf = M1.unpack_materialized_intermediate(
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
                raise RuntimeError(f"observable activation carrier byte mismatch: {exact}")
            route_metric = M1._metric(route_actual, reference["routes"])
            final_metric = M1._metric(final_actual, reference["final"])
            if not M1._numeric_pass(route_metric, "route") or not M1._numeric_pass(
                final_metric, "final"
            ):
                raise RuntimeError(
                    f"numerical output closure failed: route={route_metric}, final={final_metric}"
                )
            if not bool(torch.isfinite(route_actual).all()) or not bool(
                torch.isfinite(final_actual).all()
            ):
                raise RuntimeError("device output contains nonfinite value")
            hashes = {
                "input_payload": M1._tensor_bytes_sha256(input_payload),
                "input_scale": M1._tensor_bytes_sha256(input_sf),
                "middle_payload": M1._tensor_bytes_sha256(middle_payload),
                "middle_scale": M1._tensor_bytes_sha256(middle_sf),
                "route_output": M1._tensor_bytes_sha256(route_actual),
                "final_output": M1._tensor_bytes_sha256(final_actual),
            }
            repeat_hashes.append(hashes)
            run_receipts.append({
                "repeat": repeat,
                "exact_activation_carriers": exact,
                "route_metric": route_metric,
                "final_metric": final_metric,
                "route_to_physical_sha256": schedule["route_to_physical_sha256"],
                "hashes": hashes,
            })
        if any(item != repeat_hashes[0] for item in repeat_hashes[1:]):
            raise RuntimeError(f"M{m} five-run bitwise determinism gate failed")
        case_receipts.append({
            "m": m,
            "dispatch": runtime.debug_dispatch,
            "schedule": schedule,
            "reference": {
                key: M1._tensor_bytes_sha256(reference[key])
                for key in (
                    "input_payload", "input_scale", "middle_payload", "middle_scale",
                    "routes", "final",
                )
            },
            "runs": run_receipts,
        })

    return {
        "schema": PROTOCOL["schema"],
        "decision": "pass",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image_id,
        "identities": identities,
        "executed_sources": sources,
        "scratch_receipts": scratch,
        "scale_receipts": scale_receipts,
        "payload": {
            "prototypes_sha256": M1._tensor_bytes_sha256(prototypes),
            "topk_ids_sha256": M1._tensor_bytes_sha256(ids),
            "all_weights_sha256": M1._tensor_bytes_sha256(all_weights),
        },
        "selected_weight_decode": reference_cache["weight_receipts"],
        "cases": case_receipts,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_used": True,
        "kld_tested": False,
        "throughput_tested": False,
    }


def build_probe_command(args: argparse.Namespace, repo: Path) -> list[str]:
    return [
        "docker", "run", "--rm", "--gpus", f"device={args.gpu_device}",
        "--network=none", "--ipc=private", "--shm-size=1g",
        "-e", "PYTHONPATH=/opt/p8-coupled-runtime:/work",
        "-e", "OMP_NUM_THREADS=2", "-e", "GLM53_P8_NATIVE=",
        "-e", "GLM53_P4_NATIVE=", "-v", f"{repo}:/work:ro",
        "-v", f"{args.sidecar}:/inputs/sidecar.safetensors:ro",
        "-v", f"{args.design}:/inputs/design.json:ro",
        "-v", f"{args.transform}:/inputs/transform.json:ro",
        "-v", f"{args.output}:/out:rw", "--entrypoint", "/opt/venv/bin/python",
        args.image, "/work/scripts/run_p8_coupled_prefill_device_closure.py",
        "--probe", "--image-id", args.image,
        "--sidecar", "/inputs/sidecar.safetensors",
        "--sidecar-sha256", args.sidecar_sha256,
        "--design", "/inputs/design.json", "--design-sha256", args.design_sha256,
        "--transform", "/inputs/transform.json",
        "--transform-sha256", args.transform_sha256,
        "--runtime-manifest", "/opt/p8-coupled-runtime/image-manifest.json",
        "--runtime-manifest-sha256", args.runtime_manifest_sha256,
        "--output", "/out/result.json",
    ]


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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
        if not path.is_file() or M1.sha256_file(path) != expected:
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
    temperature = int(subprocess.check_output(
        [
            "nvidia-smi", "-i", args.gpu_device, "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip())
    if temperature > 75:
        raise RuntimeError(f"startup temperature {temperature} C exceeds 75 C")

    args.output.mkdir(mode=0o700, parents=True)
    command = build_probe_command(args, ROOT)
    launch = {
        "schema": "glm53.p8-full-coupled-prefill-device-launch.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image,
        "gpu_device": args.gpu_device,
        "startup_temperature_c": temperature,
        "command": command,
        "services_modified": [],
        "started_unix_ns": time.time_ns(),
    }
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (args.output / "probe.stdout.log").write_text(completed.stdout)
    (args.output / "probe.stderr.log").write_text(completed.stderr)
    launch["completed_unix_ns"] = time.time_ns()
    launch["exit_code"] = completed.returncode
    launch["result_present"] = (args.output / "result.json").is_file()
    launch["evidence_bytes"] = _tree_bytes(args.output)
    (args.output / "execution.json").write_text(json.dumps(launch, indent=2) + "\n")
    if _tree_bytes(args.output) > MAX_EVIDENCE_BYTES:
        raise RuntimeError("closure evidence exceeded the frozen 64 MiB bound")
    if completed.returncode != 0:
        raise RuntimeError("coupled prefill closure failed; preserved logs and receipt")
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
