#!/usr/bin/env python3
"""Opt-in CUDA-graph/eager parity closure for the exact v9 coupled P8 wrapper.

The default invocation prints the preregistered protocol and performs no device
work. ``--execute`` launches the private probe in the immutable v9 image.  This
file is bind-mounted read-only, so running it does not require an image rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
V9_IMAGE = "sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3"
V9_RUNTIME_MANIFEST_SHA256 = "9a57438b3cefd022772bc471980fb0ece8c19087d08f02c875a73da2b6e392d1"
V9_SIDECAR_SHA256 = "c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433"
V9_DESIGN_SHA256 = "deb72317a1f860e769bfc681190a851cf0d261a460761612fc0044c5295ce74a"
V9_TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
M1_RESULT_SHA256 = "2e3b1d09940259ee5792c4464f0187cc9a44e386e72686b046d9284dadd299de"
PREFILL_RESULT_SHA256 = "d8a0885d7fda130b959bacced2c0af13901ec96c0c6d8db844100b5c257bd262"
FROZEN_PAYLOAD_SHA256 = {
    "m1_x": "c432187fe0944c4f0dd398b5c1ee5e8d167358da489c18d71a9814bec9ebd0ff",
    "m1_topk_ids": "724a0e6f433e0b48fcc8a26cf7415f0a9ac47f82e3081f60dd868eb8bf9ab0db",
    "m1_topk_weights": "bf6c1df7b916eed2a8d5d8c251ec32390b658030059919d6f738fb077a9b3604",
    "prefill_prototypes": "ed8c573433420f351eb32ee15e100cf1d0573e0f92848249fd1899055a652459",
    "prefill_topk_ids": "724a0e6f433e0b48fcc8a26cf7415f0a9ac47f82e3081f60dd868eb8bf9ab0db",
    "prefill_all_weights": "4acb7027559ee77598fd97904e6200147df7f0342eaee53018c2902c79f0496a",
}
REPETITIONS = 5
M_CASES = (1, 2, 64, 65)
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


def _load(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1 = _load("p8_graph_m1_helpers", "run_p8_coupled_m1_device_closure.py")
PREFILL = _load("p8_graph_prefill_helpers", "run_p8_coupled_prefill_device_closure.py")

PROTOCOL = {
    "schema": "glm53.p8-full-coupled-cudagraph-closure.v1",
    "evidence_level": "gpu-smoke-cudagraph-eager-parity",
    "product": "P8 K4 procedural MCG alpha2 to E4M3/UE8M0-K32",
    "immutable_image": V9_IMAGE,
    "runtime_manifest_sha256": V9_RUNTIME_MANIFEST_SHA256,
    "source_eager_receipts": {
        "m1_result_sha256": M1_RESULT_SHA256,
        "prefill_result_sha256": PREFILL_RESULT_SHA256,
        "m1_protocol_sha256": M1.PROTOCOL_SHA256,
        "prefill_protocol_sha256": PREFILL.PROTOCOL_SHA256,
    },
    "geometry": {
        "layer": 3,
        "rank": 0,
        "m_cases": list(M_CASES),
        "experts": 288,
        "hidden": 4096,
        "local_intermediate": 512,
        "topk": 8,
        "fc1_tile_n": 128,
    },
    "frozen_inputs": {
        "M1": "exact seed, experts, BF16 input and routing recipe from the v9 M1 protocol",
        "M2_M64_M65": "exact seed, prototypes, experts and routing recipe from the v9 prefill protocol",
        "payload_sha256": FROZEN_PAYLOAD_SHA256,
    },
    "repetitions_per_case": REPETITIONS,
    "capture_order": [
        "instantiate the actual full-coupled P8NativeTPMoE wrapper",
        "warm the exact case on a side CUDA stream and synchronize before capture",
        "run and preserve one fresh eager observation",
        "capture one wrapper call using persistent inputs and torch.cuda.CUDAGraph",
        "replay the graph five times without changing the inputs",
    ],
    "exact_gates": [
        "installed sources and fixture/design/transform identities equal immutable v9 pins",
        "actual dispatch matches v9 full-coupled materialized M1 or M64/N128 path",
        "eager and every graph replay expose all required debug carriers",
        "input and routed down-input E4M3 payload plus UE8M0-K32 bytes equal the same CPU reference",
        "route and final output byte hashes from every graph replay equal the fresh eager observation",
        "all observable carrier hashes are identical between eager and every graph replay",
        "five graph replay receipt dictionaries are bitwise identical",
    ],
    "numeric_gates": dict(M1.PROTOCOL["numeric_gates"]),
    "failure_policy": (
        "capture failure, compilation during capture, missing/stale carrier, dispatch drift, "
        "byte mismatch, nonfinite output, or failed frozen numerical gate is a graph closure failure"
    ),
    "claim_boundary": (
        "synthetic layer-3/rank-0 component graph-versus-eager closure for M1/M2/M64/M65; "
        "not KLD, throughput, serving, all-rank, real-checkpoint, or full-model qualification"
    ),
    "image_rebuild_required": False,
    "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    "ldlq": False,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def _expected_dispatch(m: int) -> dict[str, object]:
    return {
        "small_m": m == 1,
        "materialized": True,
        "fused_scratch_zero": False,
        "fc1_tile_n": 128,
        "tile_m": 16 if m == 1 else 64,
    }


def _observe(runtime, output, *, m: int, ids_cpu, reference) -> dict[str, object]:
    """Copy and validate one completed eager call or graph replay."""
    import torch

    if runtime.debug_dispatch != _expected_dispatch(m):
        raise RuntimeError(f"fallback/identity dispatch observed: {runtime.debug_dispatch}")
    if set(runtime.debug_tensors) != M1.REQUIRED_DEBUG:
        raise RuntimeError(
            f"required graph-visible intermediates inaccessible: {set(runtime.debug_tensors)}"
        )
    debug = runtime.debug_tensors
    if m == 1:
        physical_rows = [slot * 16 for slot in range(8)]
        schedule = {"mode": "small_m", "physical_rows": physical_rows}
    else:
        physical_rows, schedule = PREFILL.map_route_order_physical_rows(
            debug["row_counts"], debug["expert_tile_base"], debug["token_map"],
            ids_cpu.reshape(-1),
        )
        if schedule["active_experts"] != len(M1.EXPERT_IDS):
            raise RuntimeError("grouped graph replay did not exercise all selected experts")

    input_payload = debug["packed_a"].view(torch.uint8)[: m * 4096]
    input_payload = input_payload.detach().cpu().reshape(m, 4096)
    input_scale = debug["scale_flat"].view(torch.uint8)[: m * 128]
    input_scale = input_scale.detach().cpu().reshape(m, 128)
    middle_payload, middle_scale = M1.unpack_materialized_intermediate(
        debug["intermediate_u32"], physical_rows
    )
    routes = debug["route_output"].detach().float().cpu()
    final = output.detach().cpu()
    exact = {
        "input_payload": torch.equal(input_payload, reference["input_payload"]),
        "input_scale": torch.equal(input_scale, reference["input_scale"]),
        "middle_payload": torch.equal(middle_payload, reference["middle_payload"]),
        "middle_scale": torch.equal(middle_scale, reference["middle_scale"]),
    }
    if not all(exact.values()):
        raise RuntimeError(f"observable activation carrier byte mismatch: {exact}")
    route_metric = M1._metric(routes, reference["routes"])
    final_metric = M1._metric(final, reference["final"])
    if not M1._numeric_pass(route_metric, "route") or not M1._numeric_pass(
        final_metric, "final"
    ):
        raise RuntimeError(
            f"frozen numerical closure failed: route={route_metric}, final={final_metric}"
        )
    if not bool(torch.isfinite(routes).all()) or not bool(torch.isfinite(final).all()):
        raise RuntimeError("graph/eager output contains nonfinite value")
    tensors = {
        "input_payload": input_payload,
        "input_scale": input_scale,
        "middle_payload": middle_payload,
        "middle_scale": middle_scale,
        "route_output": routes,
        "final_output": final,
    }
    return {
        "exact_activation_carriers": exact,
        "route_metric": route_metric,
        "final_metric": final_metric,
        "hashes": {name: M1._tensor_bytes_sha256(value) for name, value in tensors.items()},
        "schedule": schedule,
    }


def _case_material(m: int, *, m1_scales, prefill_cache, prototypes, all_weights, ids):
    """Return the exact previously frozen CPU inputs and reference for one M."""
    import torch

    if m == 1:
        generator = torch.Generator(device="cpu").manual_seed(M1.SEED)
        x = (torch.randn(1, 4096, generator=generator) * 0.03125).to(torch.bfloat16)
        logits = torch.randn(8, generator=generator)
        weights = torch.softmax(logits, 0).float().reshape(1, 8)
        topk_ids = torch.tensor(M1.EXPERT_IDS, dtype=torch.int32).reshape(1, 8)
        reference = M1._reference(m1_scales["sidecar"], x, weights.reshape(-1), m1_scales["scales"])
        return x, weights, topk_ids, reference
    x, weights, topk_ids, prototype_index = PREFILL._case_inputs(
        m, prototypes, all_weights, ids
    )
    reference = PREFILL.expand_reference_case(prefill_cache, prototype_index, weights)
    return x, weights, topk_ids, reference


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from p8_native_kernel import P8NativeTPMoE

    if args.image_id != V9_IMAGE:
        raise RuntimeError("graph closure is pinned to the immutable v9 image")
    if M1.sha256_file(Path(__file__).resolve()) != args.harness_sha256:
        raise RuntimeError("bind-mounted graph harness differs from outer launch receipt")
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
        "sidecar": V9_SIDECAR_SHA256,
        "design": V9_DESIGN_SHA256,
        "transform": V9_TRANSFORM_SHA256,
        "runtime_manifest": V9_RUNTIME_MANIFEST_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"v9 input identity mismatch: expected {expected}, got {identities}")
    sources = PREFILL._verify_runtime_sources(
        args.runtime_manifest, V9_RUNTIME_MANIFEST_SHA256
    )
    metadata, scales, scale_receipts = M1._load_sidecar_reference(
        args.sidecar, V9_TRANSFORM_SHA256
    )
    if metadata.get("source_design_sha256") != V9_DESIGN_SHA256:
        raise RuntimeError("sidecar source design hash differs")
    prototypes, all_weights, ids = PREFILL._make_inputs()
    prefill_payload = {
        "prefill_prototypes": M1._tensor_bytes_sha256(prototypes),
        "prefill_topk_ids": M1._tensor_bytes_sha256(ids),
        "prefill_all_weights": M1._tensor_bytes_sha256(all_weights),
    }
    for name, actual in prefill_payload.items():
        if actual != FROZEN_PAYLOAD_SHA256[name]:
            raise RuntimeError(f"frozen {name} payload differs")
    prefill_cache = PREFILL._reference_prototypes(args.sidecar, prototypes, scales)

    runtime = P8NativeTPMoE(
        args.sidecar,
        device=torch.device("cuda"), tp_rank=0, layer=3,
        expected_design_sha256=V9_DESIGN_SHA256,
        expected_transform_sha256=V9_TRANSFORM_SHA256,
        topk=8, hidden=4096, intermediate=512, swiglu_limit=10.0,
        deterministic_output=True, small_m_scheduler=True, fc1_tile_n=128,
        debug_capture=True, fuse_scratch_zero=False,
    )
    if not runtime.full_coupled or runtime.scale_component is None:
        raise RuntimeError("actual wrapper did not resolve full-coupled mode")

    cases = []
    observed_payload = dict(prefill_payload)
    for m in M_CASES:
        x_cpu, weights_cpu, ids_cpu, reference = _case_material(
            m,
            m1_scales={"sidecar": args.sidecar, "scales": scales},
            prefill_cache=prefill_cache,
            prototypes=prototypes, all_weights=all_weights, ids=ids,
        )
        x = x_cpu.cuda()
        weights = weights_cpu.cuda()
        topk_ids = ids_cpu.cuda()
        if m == 1:
            observed_payload.update({
                "m1_x": M1._tensor_bytes_sha256(x_cpu),
                "m1_topk_ids": M1._tensor_bytes_sha256(ids_cpu),
                "m1_topk_weights": M1._tensor_bytes_sha256(weights_cpu.reshape(-1)),
            })
            if observed_payload != FROZEN_PAYLOAD_SHA256:
                raise RuntimeError("frozen M1 payload differs from the v9 eager receipt")

        # Compile and initialize every branch outside capture, on a side stream.
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            runtime(x, weights, topk_ids)
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()

        eager_output = runtime(x, weights, topk_ids)
        torch.cuda.synchronize()
        eager = _observe(
            runtime, eager_output, m=m, ids_cpu=ids_cpu, reference=reference
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = runtime(x, weights, topk_ids)
        graph_runs = []
        for repeat in range(REPETITIONS):
            graph.replay()
            torch.cuda.synchronize()
            observed = _observe(
                runtime, graph_output, m=m, ids_cpu=ids_cpu, reference=reference
            )
            observed["repeat"] = repeat
            if observed["hashes"] != eager["hashes"]:
                raise RuntimeError(
                    f"M{m} graph replay {repeat} is not byte-exact to eager"
                )
            graph_runs.append(observed)
        comparable = [
            {key: value for key, value in run.items() if key != "repeat"}
            for run in graph_runs
        ]
        if any(run != comparable[0] for run in comparable[1:]):
            raise RuntimeError(f"M{m} five graph replays are not bitwise deterministic")
        cases.append({"m": m, "eager": eager, "graph_replays": graph_runs})
        del graph_output, graph, eager_output, x, weights, topk_ids

    return {
        "schema": PROTOCOL["schema"],
        "decision": "pass",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image_id,
        "harness_sha256": args.harness_sha256,
        "identities": identities,
        "executed_sources": sources,
        "payload": observed_payload,
        "scale_receipts": scale_receipts,
        "cases": cases,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_used": True,
        "cuda_graph_tested": True,
        "kld_tested": False,
        "throughput_tested": False,
    }


def build_probe_command(args: argparse.Namespace, repo: Path) -> list[str]:
    harness_sha256 = M1.sha256_file(Path(__file__).resolve())
    return [
        "docker", "run", "--rm", "--gpus", f"device={args.gpu_device}",
        "--network=none", "--ipc=private", "--shm-size=1g",
        "-e", "PYTHONPATH=/opt/p8-coupled-runtime:/work", "-e", "OMP_NUM_THREADS=2",
        "-e", "GLM53_P8_NATIVE=", "-e", "GLM53_P4_NATIVE=",
        "-v", f"{repo}:/work:ro",
        "-v", f"{args.sidecar}:/inputs/sidecar.safetensors:ro",
        "-v", f"{args.design}:/inputs/design.json:ro",
        "-v", f"{args.transform}:/inputs/transform.json:ro",
        "-v", f"{args.output}:/out:rw",
        "--entrypoint", "/opt/venv/bin/python", V9_IMAGE,
        "/work/scripts/run_p8_coupled_graph_device_closure.py", "--probe",
        "--image-id", V9_IMAGE,
        "--harness-sha256", harness_sha256,
        "--sidecar", "/inputs/sidecar.safetensors",
        "--design", "/inputs/design.json",
        "--transform", "/inputs/transform.json",
        "--runtime-manifest", "/opt/p8-coupled-runtime/image-manifest.json",
        "--output", "/out/result.json",
    ]


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def outer_execute(args: argparse.Namespace) -> None:
    if args.image != V9_IMAGE:
        raise ValueError(f"--image must be the frozen v9 ID {V9_IMAGE}")
    if args.output != args.output.resolve() or args.output.exists():
        raise ValueError("fresh canonical output directory required")
    for path, expected in (
        (args.sidecar, V9_SIDECAR_SHA256),
        (args.design, V9_DESIGN_SHA256),
        (args.transform, V9_TRANSFORM_SHA256),
    ):
        if not path.is_file() or M1.sha256_file(path) != expected:
            raise ValueError(f"pinned v9 input differs: {path}")
    actual_image = subprocess.check_output(
        ["docker", "image", "inspect", V9_IMAGE, "--format", "{{.Id}}"], text=True
    ).strip()
    if actual_image != V9_IMAGE:
        raise ValueError("local image identity differs")
    manifest_line = subprocess.check_output(
        ["docker", "run", "--rm", "--network=none", "--runtime=runc", "-e",
         "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", V9_IMAGE,
         "/opt/p8-coupled-runtime/image-manifest.json"], text=True
    ).strip()
    if manifest_line.split()[0] != V9_RUNTIME_MANIFEST_SHA256:
        raise ValueError("installed v9 runtime manifest differs")
    active = subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu_device, "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True
    ).strip()
    if active:
        raise RuntimeError("selected GPU is not idle; no service action is authorized")
    temperature = int(subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu_device, "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"], text=True
    ).strip())
    if temperature > 75:
        raise RuntimeError(f"startup temperature {temperature} C exceeds 75 C")

    args.output.mkdir(mode=0o700, parents=True)
    command = build_probe_command(args, ROOT)
    harness_sha256 = M1.sha256_file(Path(__file__).resolve())
    receipt = {
        "schema": "glm53.p8-full-coupled-cudagraph-launch.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": V9_IMAGE,
        "harness_sha256": harness_sha256,
        "gpu_device": args.gpu_device,
        "startup_temperature_c": temperature,
        "command": command,
        "services_modified": [],
        "started_unix_ns": time.time_ns(),
    }
    (args.output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (args.output / "probe.stdout.log").write_text(completed.stdout)
    (args.output / "probe.stderr.log").write_text(completed.stderr)
    receipt.update({
        "completed_unix_ns": time.time_ns(),
        "exit_code": completed.returncode,
        "result_present": (args.output / "result.json").is_file(),
        "evidence_bytes": _tree_bytes(args.output),
    })
    (args.output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if _tree_bytes(args.output) > MAX_EVIDENCE_BYTES:
        raise RuntimeError("graph closure evidence exceeded the frozen 64 MiB bound")
    if completed.returncode:
        raise RuntimeError("coupled graph closure failed; preserved logs and receipt")
    result = json.loads((args.output / "result.json").read_text())
    if result.get("decision") != "pass" or result.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("probe result did not satisfy the frozen graph protocol")
    print(json.dumps({"decision": "pass", "result": str(args.output / "result.json")}))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image")
    parser.add_argument("--image-id")
    parser.add_argument("--harness-sha256")
    parser.add_argument("--gpu-device")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _require(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.execute and args.probe:
        raise ValueError("--execute and --probe are mutually exclusive")
    if not args.execute and not args.probe:
        print(json.dumps({"protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256}, sort_keys=True))
        return
    if args.probe:
        _require(args, (
            "sidecar", "design", "transform", "runtime_manifest", "output",
            "image_id", "harness_sha256",
        ))
        if not re.fullmatch(r"[a-f0-9]{64}", args.harness_sha256):
            raise ValueError("--harness-sha256 must be lowercase SHA256")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        try:
            result = run_probe(args)
        except BaseException as error:
            result = {
                "schema": PROTOCOL["schema"], "decision": "fail",
                "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(), "gpu_used": True,
                "cuda_graph_tested": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True))
            raise
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return
    _require(args, (
        "sidecar", "design", "transform", "output", "image", "gpu_device",
    ))
    outer_execute(args)


if __name__ == "__main__":
    main()
