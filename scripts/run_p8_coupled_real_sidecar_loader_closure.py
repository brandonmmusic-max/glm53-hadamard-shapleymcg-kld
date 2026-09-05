#!/usr/bin/env python3
"""Close the real-layer TP4 sidecar loader ABI in the immutable P8 v9 image.

The default invocation prints the frozen protocol and performs no device work.
``--execute`` launches one private probe which constructs the actual v9
``P8NativeTPMoE`` once per rank, sequentially.  It does not call the wrapper's
MoE/MMA path and cannot provide numerical, serving, throughput, or KLD evidence.
The constructor's existing scale-repack path may itself launch device work.
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
V3_DESIGN_SHA256 = "4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695"
TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
LAYER = 3
WORLD_SIZE = 4
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
TENSOR_NAMES = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "gate_up_suh_fp16",
    "intermediate_scales_fp16",
    "down_svh_fp16",
    "coupled_sign_draw_u8",
)
EXPECTED_SHAPES = {
    "w13_trellis": [2, 288, 256, 32, 64],
    "w2_trellis": [288, 32, 256, 64],
    "w13_scale_ue8m0": [288, 1024, 128],
    "w2_scale_ue8m0": [288, 4096, 16],
    "gate_up_suh_fp16": [4096],
    "intermediate_scales_fp16": [288, 1536],
    "down_svh_fp16": [4096],
    "coupled_sign_draw_u8": [288],
}
EXPECTED_DTYPES = {
    "w13_trellis": "int16",
    "w2_trellis": "int16",
    "w13_scale_ue8m0": "uint8",
    "w2_scale_ue8m0": "uint8",
    "gate_up_suh_fp16": "float16",
    "intermediate_scales_fp16": "float16",
    "down_svh_fp16": "float16",
    "coupled_sign_draw_u8": "uint8",
}


def _load_helper():
    path = Path(__file__).with_name("run_p8_coupled_m1_device_closure.py")
    spec = importlib.util.spec_from_file_location("p8_real_loader_m1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1 = _load_helper()

PROTOCOL = {
    "schema": "glm53.p8-full-coupled-real-sidecar-loader-closure.v1",
    "evidence_level": "gpu-loader-abi-closure",
    "product": "P8 K4 procedural MCG alpha2 to E4M3/UE8M0-K32",
    "immutable_image": V9_IMAGE,
    "runtime_manifest_sha256": V9_RUNTIME_MANIFEST_SHA256,
    "geometry": {
        "layer": LAYER,
        "ranks": list(range(WORLD_SIZE)),
        "world_size": WORLD_SIZE,
        "experts": 288,
        "hidden": 4096,
        "local_intermediate": 512,
        "topk": 8,
        "fc1_tile_n": 128,
    },
    "external_pins": {
        "v3_design_sha256": V3_DESIGN_SHA256,
        "transform_sha256": TRANSFORM_SHA256,
    },
    "execution_order": "ranks 0,1,2,3 sequentially on one visible GPU",
    "exact_gates": [
        "installed runtime sources equal the immutable v9 manifest",
        "postwrite receipt is source-exact PASS and still retirement-false",
        "four sidecar file hashes, metadata, shapes, dtypes and tensor hashes equal the postwrite receipt",
        "actual v9 P8NativeTPMoE constructs for the matching real layer and rank",
        "all seven runtime payload/scale tensor byte streams equal the postwrite tensor hashes",
        "stored draw0 bytes and runtime-regenerated sign bytes equal their pinned hashes",
        "wrapper resolves full_coupled K4 MCG, N128, deterministic TP4 with descriptor aliases and no identity fallback",
    ],
    "failure_policy": (
        "any image/source/input/hash/schema/shape/dtype/rank/layer drift, partial coupling, "
        "identity mode, alternate descriptor storage, missing runtime tensor, or CUDA allocation "
        "failure is a loader closure failure"
    ),
    "claim_boundary": (
        "real layer-3 all-four-rank sidecar load/repack/ABI closure only; no MoE/MMA call, "
        "numerical closure, CUDA graph, KLD, throughput, serving, or full-model qualification; "
        "constructor scale repacking may use device kernels"
    ),
    "moe_mma_executed": False,
    "kld_tested": False,
    "throughput_tested": False,
    "ldlq": False,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _load_postwrite(path: Path, sidecars: Sequence[Path]) -> dict[str, object]:
    """Validate the closure receipt and bind its ordered ranks to four files."""

    if len(sidecars) != WORLD_SIZE or len({item.resolve() for item in sidecars}) != WORLD_SIZE:
        raise RuntimeError("exactly four distinct sidecars are required in rank order")
    receipt = json.loads(path.read_text())
    required = {
        "schema": "glm53-p8-coupled-tp4-postwrite-closure.v1",
        "status": "pass",
        "evidence_level": "postwrite-source-exact-structural-closure",
        "layer": LAYER,
        "world_size": WORLD_SIZE,
        "runtime_loader_closure": "not tested",
        "retirement_authorized": False,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in required.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"postwrite receipt contract mismatch: {mismatches}")
    if receipt.get("source_design_sha256") != V3_DESIGN_SHA256:
        raise RuntimeError("postwrite receipt does not bind the external V3 design")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or [row.get("rank") for row in ranks] != list(range(4)):
        raise RuntimeError("postwrite receipt must contain ranks 0,1,2,3 exactly once")
    for rank, (path_item, row) in enumerate(zip(sidecars, ranks, strict=True)):
        if not path_item.is_file():
            raise FileNotFoundError(path_item)
        if (
            row.get("rank") != rank
            or row.get("bytes") != path_item.stat().st_size
            or row.get("sha256") != M1.sha256_file(path_item)
            or row.get("tensor_count") != len(TENSOR_NAMES)
            or row.get("source_exact") is not True
            or row.get("shapes") != EXPECTED_SHAPES
            or row.get("dtypes") != EXPECTED_DTYPES
            or set(row.get("tensor_sha256", {})) != set(TENSOR_NAMES)
            or not all(_is_sha256(value) for value in row.get("tensor_sha256", {}).values())
        ):
            raise RuntimeError(f"rank {rank} sidecar/postwrite identity differs")
    return receipt


def _tensor_sha256_chunked(tensor, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Hash device or CPU tensor bytes with bounded host-copy memory."""

    import torch

    if not tensor.is_contiguous():
        raise RuntimeError("runtime tensor is not contiguous")
    raw = tensor.detach().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for begin in range(0, raw.numel(), chunk_bytes):
        part = raw[begin : begin + chunk_bytes].cpu().numpy().tobytes()
        digest.update(part)
    return digest.hexdigest()


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def _view_runtime_tensor(tensor, *, name: str):
    expected_dtype = _torch_dtype(EXPECTED_DTYPES[name])
    expected_shape = tuple(EXPECTED_SHAPES[name])
    expected_bytes = 1
    for dim in expected_shape:
        expected_bytes *= dim
    expected_bytes *= _torch_dtype(EXPECTED_DTYPES[name]).itemsize
    if tensor.numel() * tensor.element_size() != expected_bytes:
        raise RuntimeError(f"runtime {name} byte extent differs")
    return tensor.view(expected_dtype).reshape(expected_shape)


def _inspect_sidecar(path: Path, row: dict[str, object], *, rank: int) -> dict[str, object]:
    """Reopen every stored tensor and require its postwrite hash/ABI."""

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        required_metadata = {
            "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
            "layer": str(LAYER),
            "rank": str(rank),
            "world_size": "4",
            "bits": "4",
            "alphabet": "e4m3",
            "scale": "ue8m0-k32",
            "law": "procedural-mcg-alpha2",
            "boundary": "coupled-h512-h128-suh-svh-v1",
            "full_coupled": "true",
            "activation": "silu-cap10",
            "ldlq": "false",
            "source_design_sha256": V3_DESIGN_SHA256,
            "encoder_transform_sha256": TRANSFORM_SHA256,
            "sign_draw": "0",
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in required_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"rank {rank} sidecar metadata mismatch: {mismatches}")
        if set(handle.keys()) != set(TENSOR_NAMES):
            raise RuntimeError(f"rank {rank} sidecar tensor inventory differs")
        hashes = {}
        for name in TENSOR_NAMES:
            tensor = handle.get_tensor(name)
            dtype = str(tensor.dtype).removeprefix("torch.")
            if list(tensor.shape) != EXPECTED_SHAPES[name] or dtype != EXPECTED_DTYPES[name]:
                raise RuntimeError(f"rank {rank} stored {name} ABI differs")
            digest = _tensor_sha256_chunked(tensor)
            if digest != row["tensor_sha256"][name] or metadata.get(f"sha256_{name}") != digest:
                raise RuntimeError(f"rank {rank} stored {name} hash differs")
            hashes[name] = digest
        if metadata.get("sha256_coupled_signs_fp16") is None:
            raise RuntimeError(f"rank {rank} lacks regenerated sign hash")
    return {"metadata": metadata, "tensor_sha256": hashes}


def _inspect_runtime_rank(runtime_class, sidecar: Path, row: dict[str, object], *, rank: int, device):
    """Construct and byte-audit one real P8NativeTPMoE rank."""

    import torch

    runtime = runtime_class(
        sidecar,
        device=device,
        tp_rank=rank,
        layer=LAYER,
        expected_design_sha256=V3_DESIGN_SHA256,
        expected_transform_sha256=TRANSFORM_SHA256,
        topk=8,
        hidden=4096,
        intermediate=512,
        swiglu_limit=10.0,
        deterministic_output=True,
        small_m_scheduler=True,
        fc1_tile_n=128,
        debug_capture=False,
        fuse_scratch_zero=False,
    )
    if (
        type(runtime).__name__ != "P8NativeTPMoE"
        or runtime.tp_rank != rank
        or runtime.layer != LAYER
        or runtime.experts != 288
        or runtime.hidden != 4096
        or runtime.intermediate != 512
        or runtime.topk != 8
        or runtime.trellis_bits != 4
        or runtime.source_design_sha256 != V3_DESIGN_SHA256
        or runtime.full_coupled is not True
        or runtime.scale_component is None
        or runtime.scale_component.full_coupled is not True
        or runtime.scale_component.transform_sha256 != TRANSFORM_SHA256
        or runtime.small_m_scheduler is not True
        or runtime.fc1_tile_n != 128
        or runtime.deterministic_output is not True
        or runtime._compiled != {}
    ):
        raise RuntimeError(f"rank {rank} actual wrapper did not resolve the frozen full-coupled ABI")
    if (
        runtime.w13_dummy.data_ptr() != runtime.w13_stream.data_ptr()
        or runtime.w2_dummy.data_ptr() != runtime.w2_stream.data_ptr()
    ):
        raise RuntimeError(f"rank {rank} used alternate/fallback descriptor weight storage")

    loaded = {
        "w13_trellis": _view_runtime_tensor(runtime.w13_stream, name="w13_trellis"),
        "w2_trellis": _view_runtime_tensor(runtime.w2_stream, name="w2_trellis"),
        "w13_scale_ue8m0": _view_runtime_tensor(
            runtime.w13_scale_mx, name="w13_scale_ue8m0"
        ),
        "w2_scale_ue8m0": _view_runtime_tensor(runtime.w2_scale_mx, name="w2_scale_ue8m0"),
    }
    packed = runtime.scale_component_packed
    offset = 0
    for name in (
        "gate_up_suh_fp16",
        "intermediate_scales_fp16",
        "down_svh_fp16",
    ):
        count = 1
        for dim in EXPECTED_SHAPES[name]:
            count *= dim
        loaded[name] = packed[offset : offset + count].reshape(EXPECTED_SHAPES[name])
        offset += count
    runtime_signs = packed[offset:]
    if runtime_signs.numel() != 1536 or runtime_signs.dtype != torch.float16:
        raise RuntimeError(f"rank {rank} regenerated sign carrier ABI differs")

    loaded_hashes = {}
    for name, tensor in loaded.items():
        expected_dtype = _torch_dtype(EXPECTED_DTYPES[name])
        if tensor.dtype != expected_dtype or list(tensor.shape) != EXPECTED_SHAPES[name]:
            raise RuntimeError(f"rank {rank} runtime-loaded {name} ABI differs")
        digest = _tensor_sha256_chunked(tensor)
        if digest != row["tensor_sha256"][name]:
            raise RuntimeError(f"rank {rank} runtime-loaded {name} hash differs")
        loaded_hashes[name] = digest
    signs_sha256 = _tensor_sha256_chunked(runtime_signs)
    return runtime, {
        "rank": rank,
        "layer": LAYER,
        "mode": "full-coupled",
        "identity_fallback": False,
        "moe_kernel_compiled": False,
        "moe_mma_executed": False,
        "runtime_loaded_tensor_sha256": loaded_hashes,
        "runtime_regenerated_signs_fp16_sha256": signs_sha256,
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from p8_native_kernel import P8NativeTPMoE

    if args.image_id != V9_IMAGE:
        raise RuntimeError("probe requires the immutable v9 image ID")
    if M1.sha256_file(Path(__file__).resolve()) != args.harness_sha256:
        raise RuntimeError("executed harness differs from the outer pin")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("loader closure requires exactly one visible CUDA device")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("loader closure requires SM120")
    identities = {
        "design": M1.sha256_file(args.design),
        "transform": M1.sha256_file(args.transform),
        "postwrite": M1.sha256_file(args.postwrite),
        "runtime_manifest": M1.sha256_file(args.runtime_manifest),
    }
    if identities != {
        "design": V3_DESIGN_SHA256,
        "transform": TRANSFORM_SHA256,
        "postwrite": args.postwrite_sha256,
        "runtime_manifest": V9_RUNTIME_MANIFEST_SHA256,
    }:
        raise RuntimeError("probe input identity differs")
    sources = M1._verify_runtime_sources(
        args.runtime_manifest, V9_RUNTIME_MANIFEST_SHA256
    )
    receipt = _load_postwrite(args.postwrite, args.sidecar)
    ranks = []
    for rank, (sidecar, row) in enumerate(zip(args.sidecar, receipt["ranks"], strict=True)):
        stored = _inspect_sidecar(sidecar, row, rank=rank)
        runtime, loaded = _inspect_runtime_rank(
            P8NativeTPMoE, sidecar, row, rank=rank, device=torch.device("cuda")
        )
        if loaded["runtime_regenerated_signs_fp16_sha256"] != stored["metadata"].get(
            "sha256_coupled_signs_fp16"
        ):
            raise RuntimeError(f"rank {rank} runtime regenerated sign hash differs")
        if row["tensor_sha256"]["coupled_sign_draw_u8"] != stored["tensor_sha256"][
            "coupled_sign_draw_u8"
        ]:
            raise RuntimeError(f"rank {rank} stored draw tensor differs")
        ranks.append({
            **loaded,
            "sidecar_sha256": row["sha256"],
            "stored_tensor_sha256": stored["tensor_sha256"],
            "stored_draw0_validated": True,
        })
        del runtime
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return {
        "schema": PROTOCOL["schema"],
        "decision": "pass",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": args.image_id,
        "harness_sha256": args.harness_sha256,
        "identities": identities,
        "executed_sources": sources,
        "ranks": ranks,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_used": True,
        "moe_mma_executed": False,
        "numerical_tested": False,
        "kld_tested": False,
        "throughput_tested": False,
        "loader_abi_gate_closed": True,
        "retirement_authorized": False,
    }


def build_probe_command(args: argparse.Namespace, repo: Path) -> list[str]:
    command = [
        "docker", "run", "--rm", "--gpus", f"device={args.gpu_device}",
        "--network=none", "--ipc=private", "--shm-size=1g",
        "-e", "PYTHONPATH=/opt/p8-coupled-runtime:/work",
        "-e", "OMP_NUM_THREADS=2", "-e", "GLM53_P8_NATIVE=", "-e", "GLM53_P4_NATIVE=",
        "-v", f"{repo}:/work:ro",
        "-v", f"{args.design}:/inputs/design.json:ro",
        "-v", f"{args.transform}:/inputs/transform.json:ro",
        "-v", f"{args.postwrite}:/inputs/postwrite.json:ro",
        "-v", f"{args.output}:/out:rw",
    ]
    for rank, sidecar in enumerate(args.sidecar):
        command.extend(("-v", f"{sidecar}:/inputs/rank-{rank}.safetensors:ro"))
    command.extend((
        "--entrypoint", "/opt/venv/bin/python", V9_IMAGE,
        "/work/scripts/run_p8_coupled_real_sidecar_loader_closure.py", "--probe",
        "--image-id", V9_IMAGE,
        "--harness-sha256", M1.sha256_file(Path(__file__).resolve()),
        "--design", "/inputs/design.json",
        "--transform", "/inputs/transform.json",
        "--postwrite", "/inputs/postwrite.json",
        "--postwrite-sha256", M1.sha256_file(args.postwrite),
        "--runtime-manifest", "/opt/p8-coupled-runtime/image-manifest.json",
        "--output", "/out/result.json",
    ))
    for rank in range(WORLD_SIZE):
        command.extend(("--sidecar", f"/inputs/rank-{rank}.safetensors"))
    return command


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def outer_execute(args: argparse.Namespace) -> None:
    if args.image != V9_IMAGE:
        raise ValueError(f"--image must be the frozen v9 ID {V9_IMAGE}")
    if args.output != args.output.resolve() or args.output.exists():
        raise ValueError("fresh canonical output directory required")
    if M1.sha256_file(args.design) != V3_DESIGN_SHA256:
        raise ValueError("external V3 design identity differs")
    if M1.sha256_file(args.transform) != TRANSFORM_SHA256:
        raise ValueError("external transform identity differs")
    postwrite = _load_postwrite(args.postwrite, args.sidecar)
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
    if temperature > 89:
        raise RuntimeError(f"startup temperature {temperature} C exceeds user-approved 89 C")

    args.output.mkdir(mode=0o700, parents=True)
    command = build_probe_command(args, ROOT)
    launch = {
        "schema": "glm53.p8-full-coupled-real-sidecar-loader-launch.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "image_id": V9_IMAGE,
        "harness_sha256": M1.sha256_file(Path(__file__).resolve()),
        "postwrite_sha256": M1.sha256_file(args.postwrite),
        "sidecar_sha256": [row["sha256"] for row in postwrite["ranks"]],
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
    launch.update({
        "completed_unix_ns": time.time_ns(),
        "exit_code": completed.returncode,
        "result_present": (args.output / "result.json").is_file(),
        "evidence_bytes": _tree_bytes(args.output),
    })
    (args.output / "execution.json").write_text(json.dumps(launch, indent=2) + "\n")
    if _tree_bytes(args.output) > MAX_EVIDENCE_BYTES:
        raise RuntimeError("loader closure evidence exceeded the frozen 16 MiB bound")
    if completed.returncode:
        raise RuntimeError("real-sidecar loader closure failed; logs and receipt preserved")
    result = json.loads((args.output / "result.json").read_text())
    if result.get("decision") != "pass" or result.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("probe result did not satisfy the frozen loader protocol")
    print(json.dumps({"decision": "pass", "result": str(args.output / "result.json")}))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image")
    parser.add_argument("--image-id")
    parser.add_argument("--harness-sha256")
    parser.add_argument("--gpu-device")
    parser.add_argument("--sidecar", type=Path, action="append")
    parser.add_argument("--design", type=Path)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--postwrite", type=Path)
    parser.add_argument("--postwrite-sha256")
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
    common = ("sidecar", "design", "transform", "postwrite", "output")
    if args.probe:
        _require(args, (*common, "image_id", "harness_sha256", "postwrite_sha256", "runtime_manifest"))
        if len(args.sidecar) != WORLD_SIZE:
            raise ValueError("probe requires four --sidecar arguments in rank order")
        if not _is_sha256(args.harness_sha256) or not _is_sha256(args.postwrite_sha256):
            raise ValueError("probe hashes must be lowercase SHA256")
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
                "moe_mma_executed": False, "kld_tested": False,
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
    if len(args.sidecar) != WORLD_SIZE:
        raise ValueError("execute requires four --sidecar arguments in rank order")
    outer_execute(args)


if __name__ == "__main__":
    main()
