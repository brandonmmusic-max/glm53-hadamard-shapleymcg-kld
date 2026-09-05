"""Prepare, but never execute, a receipt-gated dedicated P8 Nsight launch.

The exclusive launch spec contains the cloned environment: keep it private.
Trace inclusion, rank coverage and actual decode iteration counts still require
verification after an independently authorized launch. No shell is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

from .audit_p8_speed_receipts import audit, audit_slot, config_fingerprint, sha

IMAGE = "sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8"
MODEL = "glm53-p8-decode-profile-v1"
NSYS = "/usr/local/cuda/bin/nsys"
PROFILER = {"profiler": "cuda", "delay_iterations": 33, "max_iterations": 16,
            "warmup_iterations": 0, "ignore_frontend": True}
PREFIX = ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "/model"]
REQUIRED = {
    "--served-model-name": "glm53-p8-uniform-all42-native", "--host": "0.0.0.0",
    "--port": "8017", "--tensor-parallel-size": "4", "--decode-context-parallel-size": "1",
    "--max-num-batched-tokens": "2048", "--max-num-seqs": "1", "--max-model-len": "131072",
    "--attention-backend": "B12X_MLA_SPARSE", "--kv-cache-dtype": "nvfp4_ds_mla",
    "--dtype": "bfloat16", "--quantization": "modelopt", "--load-format": "instanttensor",
    "--dcp-comm-backend": "a2a", "--generation-config": "/model", "--reasoning-parser": "glm45",
}
BOOLEAN_FLAGS = {"--language-model-only", "--enable-chunked-prefill",
                 "--no-enable-prefix-caching", "--disable-custom-all-reduce"}
VALUE_FLAGS = set(REQUIRED) | {"--gpu-memory-utilization"}
DEVICE_REQUEST = {"Driver": "", "Count": -1, "DeviceIDs": None,
                  "Capabilities": [["gpu"]], "Options": {}}


def serving_argv(container: dict) -> list[str]:
    config = container["Config"]
    cmd = config.get("Cmd")
    if (config.get("Entrypoint") != ["/bin/bash"] or not isinstance(cmd, list)
            or len(cmd) != 2 or cmd[0] != "-lc" or not isinstance(cmd[1], str)):
        raise ValueError("unexpected source shell command structure")
    # The known source has no expansions, redirections, pipelines or statements.
    # Reject them even though the returned argv will never enter a shell.
    if any(char in cmd[1] for char in "$`\n\r;|&<>()"):
        raise ValueError("source shell contains unsupported syntax")
    tokens = shlex.split(cmd[1], posix=True)
    if tokens[:6] != ["exec", *PREFIX]:
        raise ValueError("source shell must exec the pinned vLLM Python command")
    result = tokens[1:]
    parsed = {}
    positions = {}
    index = len(PREFIX)
    while index < len(result):
        option = result[index]
        if option in parsed or option not in VALUE_FLAGS | BOOLEAN_FLAGS:
            raise ValueError("unexpected or duplicate serving option")
        positions[option] = index
        if option in BOOLEAN_FLAGS:
            parsed[option] = True
            index += 1
        else:
            if index + 1 == len(result) or result[index + 1].startswith("--"):
                raise ValueError("missing serving option value")
            parsed[option] = result[index + 1]
            index += 2
    if any(parsed.get(key) != value for key, value in REQUIRED.items()):
        raise ValueError("required P8 serving options differ")
    if not BOOLEAN_FLAGS <= parsed.keys() or "--gpu-memory-utilization" not in parsed:
        raise ValueError("required serving options are missing")
    result[positions["--port"] + 1] = "8018"
    result[positions["--served-model-name"] + 1] = MODEL
    result.extend(["--profiler-config", json.dumps(PROFILER, separators=(",", ":"))])
    return result


def validate_destination(out: Path, root: Path) -> Path:
    if (not out.is_absolute() or out != out.resolve() or out.exists() or out.is_symlink()
            or not out.parent.is_dir() or any(char in str(out) for char in ":\n\r\x00")):
        raise ValueError("diagnostic destination must be a fresh absolute path with an existing real parent")
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("diagnostic destination overlaps the source root")
    return out


def build_argv(root: Path, plan: Path, out: Path) -> list[str]:
    """Return an argv only; audit completion first and never create or launch anything."""
    root, plan, out = Path(root).resolve(), Path(plan).resolve(), Path(out)
    validate_destination(out, root)
    analysis_path = root / "analysis.json"
    if not analysis_path.is_file():
        raise ValueError("parent speed analysis is missing")
    analysis = json.loads(analysis_path.read_text())
    if (analysis.get("schema") != "glm53-p8-uniform-all42-vs-exl3-speed-analysis.v1"
            or analysis.get("plan", {}).get("sha256") != sha(plan)):
        raise ValueError("parent speed analysis identity differs")
    plan_data = json.loads(plan.read_text())
    candidate = plan_data.get("candidate", {})
    if (candidate.get("image_id") != IMAGE
            or candidate.get("topology") != {"tp": 4, "ep": False, "dcp": 1}
            or plan_data.get("benchmark", {}).get("cold_process_runs_per_arm") != 5):
        raise ValueError("expected pinned P8 five-pair plan differs")
    completed = audit(root, plan, allow_incomplete=False)
    if completed.get("status") != "pass" or completed.get("missing") or len(completed.get("runs", [])) != 10:
        raise ValueError("complete receipt audit is required")
    slot = root / "round-01-p8"
    verified = audit_slot(slot, plan, "p8", 1)
    source = slot / "container.json"
    raw = source.read_bytes()
    receipt = json.loads((slot / "receipt.json").read_text())
    identity = receipt["files"]["container.json"]
    if identity.get("sha256") != hashlib.sha256(raw).hexdigest() or identity.get("bytes") != len(raw):
        raise ValueError("source container changed after receipt audit")
    containers = json.loads(raw)
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("source container inventory differs")
    container = containers[0]
    if container.get("Image") != IMAGE or verified.get("image_id") != IMAGE:
        raise ValueError("source image differs from pinned P8 image")
    if config_fingerprint(container) != verified.get("config_sha256"):
        raise ValueError("source configuration differs from audited identity")
    return _clone_container_argv(container, out)


def _clone_container_argv(container: dict, out: Path) -> list[str]:
    """Validate and clone configuration; internal helper has no completion gate."""
    command = serving_argv(container)
    config, host = container["Config"], container["HostConfig"]
    if (host.get("NetworkMode") != "host" or host.get("IpcMode") != "host"
            or host.get("ShmSize") != 68719476736 or host.get("Runtime") != "runc"
            or host.get("DeviceRequests") != [DEVICE_REQUEST]
            or host.get("Privileged") is not False or host.get("CapAdd") or host.get("CapDrop")
            or host.get("SecurityOpt") != ["label=disable"]
            or host.get("PortBindings") or host.get("ReadonlyRootfs")
            or host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}):
        raise ValueError("source host or security configuration differs")
    if any(host.get(key) for key in ("Devices", "DeviceCgroupRules", "Mounts", "ExtraHosts", "Ulimits",
                                     "PidsLimit", "Memory", "CpusetCpus", "PidMode", "UsernsMode")):
        raise ValueError("unsupported additional source host configuration")
    if config.get("User") or config.get("WorkingDir") != "/":
        raise ValueError("source user or working directory differs")
    env, binds = config.get("Env"), host.get("Binds")
    if (not isinstance(env, list) or not isinstance(binds, list)
            or any(not isinstance(value, str) or "=" not in value or "\x00" in value for value in env)):
        raise ValueError("invalid source environment or mounts")
    for value in binds:
        parts = value.split(":")
        if (len(parts) != 3 or not parts[0].startswith("/") or not parts[1].startswith("/")
                or parts[2] not in {"ro", "rw"} or parts[1] == "/profile"
                or parts[1].startswith("/profile/") or any(char in value for char in "\n\r\x00")):
            raise ValueError("invalid or conflicting mount destination")
    argv = ["docker", "run", "--name", MODEL, "--cidfile", str(out / 'container.cid'),
            "--detach", "--network", "host", "--ipc", "host",
            "--shm-size", str(host["ShmSize"]), "--gpus", "all", "--runtime", "runc",
            "--security-opt", "label=disable", "--workdir", "/", "--entrypoint", NSYS]
    for value in env:
        argv.extend(["--env", value])
    for value in binds:
        argv.extend(["--volume", value])
    argv.extend(["--volume", f"{out}:/profile:rw", IMAGE, "profile", "--trace=cuda,nvtx,osrt",
                 "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
                 "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                 "--output=/profile/native-p8-c1", *command])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    argv = build_argv(args.root, args.plan, args.output_dir)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    spec = {"schema": "p8-nsys-launch-spec.v1", "status": "prepared-not-launched", "argv": argv,
            "source_container_sha256": sha(args.root / "round-01-p8" / "container.json"),
            "source_receipt_sha256": sha(args.root / "round-01-p8" / "receipt.json"),
            "plan_sha256": sha(args.plan), "analysis_sha256": sha(args.root / "analysis.json"),
            "trace_verified": False, "expected_decode_iterations": 16,
            "limitations": ["trace inclusion and all-rank coverage require verification",
                            "delay33/max16 assumes 16 prefill steps, no cache hits, no speculation, idle server",
                            "source bind contents are not rehashed by this launcher",
                            "Nsight/CUPTI support, free port, device availability and launch authority require a separate preflight"]}
    path = args.output_dir / "launch-spec.json"
    body = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(body)
    print(json.dumps({"status": spec["status"], "spec_path": str(path),
                      "spec_sha256": hashlib.sha256(body).hexdigest()}))


if __name__ == "__main__":
    main()
