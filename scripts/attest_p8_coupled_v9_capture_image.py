#!/usr/bin/env python3
"""Attest that immutable coupled-v9 already contains the CF32 capture stack.

This is deliberately not an image builder.  It performs only CPU/runc,
read-only inspection of an existing image and emits a failure-preserving
receipt compatible with the three-layer runtime validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "runtime_patch/p8_decode_capture"
IMAGE = "sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3"
RUNTIME_MANIFEST = "/opt/p8-coupled-runtime/image-manifest.json"
RUNTIME_MANIFEST_SHA256 = "9a57438b3cefd022772bc471980fb0ece8c19087d08f02c875a73da2b6e392d1"
TAIL_V2 = "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py"
TAIL_V2_SHA256 = "494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251"
CAPTURE_SOURCES = {
    "__init__.py": "267c4f5a565687a18b224d04c1f5c702149ea8ac866986dc5922c084077fcb41",
    "processor.py": "4886091710d11c00641af8074332cdd19116fa41a811be1134e7dfb713b6f287",
    "v2_hook.py": "62f35d78931a7251cd3be594d3228b0f13868f39b3e25adb5c0ad8a96c87c683",
}
CAPTURE_PACKAGE = "/opt/venv/lib/python3.12/site-packages/p8_decode_capture"
SAMPLERS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/sample/sampler.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py",
)
SAMPLER_SHA256 = "717bdd2203c8977205e16ca63385027f7e7ec8acd54afd962ada3e611cf7b958"
WARMUPS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/warmup.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py",
)
WARMUP_SHA256 = "de321498f305e2f61d5cfe4701d19a066ebfec83147f527b2ec82bfb653ea8c7"
EXPECTED_LABELS = {
    "org.klc.experiment": "glm53-p8-coupled-h512-h128-suh-svh-v9",
    "org.klc.qualification": "research-only-not-device-qualified",
    "org.klc.parent.digest": "sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745",
    "org.klc.tail-v2.sha256": TAIL_V2_SHA256,
    "org.klc.source.commit": "1a52229c17d6e2bf12e5e13e322e94b17980e815",
    "org.klc.source.tree": "6498194160a3174e320434e70b9bba8810976f10",
}
MAX_RECEIPT_BYTES = 1024 * 1024


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True)


def _runc(entrypoint: str, *args: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--network=none", "--runtime=runc",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1m",
        "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", entrypoint,
        IMAGE, *args,
    ]


def _parse_sha256sum(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if not match or match.group(2) in result:
            raise ValueError("malformed or duplicate sha256sum output")
        result[match.group(2)] = match.group(1)
    return result


def _expected_installs(manifest: dict) -> dict[str, str]:
    source_sha = manifest.get("source_sha256")
    installs = manifest.get("install")
    if not isinstance(source_sha, dict) or not isinstance(installs, dict):
        raise ValueError("runtime manifest lacks source/install maps")
    expected: dict[str, str] = {}
    for source, destinations in installs.items():
        digest = source_sha.get(source)
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"runtime manifest lacks source hash: {source}")
        if not isinstance(destinations, list) or not destinations:
            raise ValueError(f"runtime manifest lacks install destinations: {source}")
        for destination in destinations:
            if not isinstance(destination, str) or not destination.startswith("/"):
                raise ValueError(f"invalid runtime install destination: {source}")
            if destination in expected and expected[destination] != digest:
                raise ValueError(f"conflicting runtime install hash: {destination}")
            expected[destination] = digest
    return expected


def _artifact(path: Path) -> dict[str, int | str]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def attest(output: Path) -> dict:
    output = Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError("output must be an absolute, fresh directory")
    local_capture = {
        name: sha(CONTEXT / name) for name in CAPTURE_SOURCES
    }
    if local_capture != CAPTURE_SOURCES:
        raise ValueError("local capture package source identity differs")

    inspect_command = ["docker", "image", "inspect", IMAGE]
    manifest_command = _runc("cat", RUNTIME_MANIFEST)
    output.mkdir(parents=True)
    plan = {
        "schema": "glm53.p8-coupled-cf32-capture-attestation-plan.v1",
        "status": "started",
        "started_unix_ns": time.time_ns(),
        "image_id": IMAGE,
        "parent_image_id": IMAGE,
        "reuse_existing_image": True,
        "image_built": False,
        "added_image_bytes": 0,
        "attester_sha256": sha(Path(__file__)),
        "local_capture_source_sha256": local_capture,
        "commands": {
            "image_inspect": inspect_command,
            "runtime_manifest": manifest_command,
        },
        "gpu_used": False,
        "speed_measurement_valid": False,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    receipt = dict(plan)
    try:
        inspect_raw = _run(inspect_command)
        (output / "image-inspect.json").write_text(inspect_raw)
        inspected = json.loads(inspect_raw)
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise ValueError("image inspect did not return one image")
        image = inspected[0]
        if image.get("Id") != IMAGE:
            raise ValueError("immutable v9 image identity differs")
        labels = image.get("Config", {}).get("Labels", {})
        if any(labels.get(key) != value for key, value in EXPECTED_LABELS.items()):
            raise ValueError("immutable v9 image labels differ")

        manifest_raw = _run(manifest_command)
        (output / "image-manifest.json").write_text(manifest_raw)
        if sha(output / "image-manifest.json") != RUNTIME_MANIFEST_SHA256:
            raise ValueError("runtime manifest hash differs")
        manifest = json.loads(manifest_raw)
        if manifest.get("schema") != "glm53.p8-coupled-image-sources.v9":
            raise ValueError("runtime manifest schema differs")
        if manifest.get("tail_v2", {}).get("sha256") != TAIL_V2_SHA256:
            raise ValueError("runtime manifest Tail-V2 identity differs")

        expected = _expected_installs(manifest)
        expected[RUNTIME_MANIFEST] = RUNTIME_MANIFEST_SHA256
        expected[TAIL_V2] = TAIL_V2_SHA256
        for name, digest in CAPTURE_SOURCES.items():
            expected[f"{CAPTURE_PACKAGE}/{name}"] = digest
        expected.update({path: SAMPLER_SHA256 for path in SAMPLERS})
        expected.update({path: WARMUP_SHA256 for path in WARMUPS})
        hash_command = _runc("sha256sum", *sorted(expected))
        plan["commands"]["installed_sha256"] = hash_command
        (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        hashes_raw = _run(hash_command)
        (output / "installed-sha256.txt").write_text(hashes_raw)
        observed = _parse_sha256sum(hashes_raw)
        if observed != {path: expected[path] for path in sorted(expected)}:
            raise ValueError("installed v9 runtime/capture source hash differs")

        installed = {
            **{f"p8_decode_capture/{name}": observed[f"{CAPTURE_PACKAGE}/{name}"]
               for name in CAPTURE_SOURCES},
            "sampler.py": observed[SAMPLERS[0]],
            "warmup.py": observed[WARMUPS[0]],
        }
        receipt.update(
            schema="glm53.p8-coupled-cf32-capture-image.v1",
            status="complete",
            runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
            tail_v2_sha256=TAIL_V2_SHA256,
            installed_sha256=installed,
            full_installed_sha256=observed,
            image_size_bytes=int(image["Size"]),
            image_layer_count=len(image.get("RootFS", {}).get("Layers", [])),
            classification={
                "distribution_role": "custom",
                "qualification_status": "research-only",
                "maintenance_status": "ephemeral",
            },
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["completed_unix_ns"] = time.time_ns()
        receipt["artifacts"] = {
            path.name: _artifact(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "receipt.json"
        }
        serialized = json.dumps(receipt, indent=2) + "\n"
        if len(serialized.encode()) > MAX_RECEIPT_BYTES:
            raise RuntimeError("capture attestation receipt exceeds 1 MiB")
        with (output / "receipt.json").open("x") as stream:
            stream.write(serialized)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = attest(args.output)
    print(json.dumps({
        "status": receipt["status"],
        "image_id": receipt["image_id"],
        "reuse_existing_image": receipt["reuse_existing_image"],
        "receipt": str(args.output / "receipt.json"),
    }))


if __name__ == "__main__":
    main()
