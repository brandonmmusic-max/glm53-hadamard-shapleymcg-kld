#!/usr/bin/env python3
"""Build one CPU-only forced-decode control image from a pinned product image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "runtime_patch/p8_decode_capture"
PARENTS = {
    "stock": "sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe",
    "exl3": "sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf",
}
DOCKERFILES = {arm: CONTEXT / f"Dockerfile.{arm}-control" for arm in PARENTS}
TAGS = {arm: f"klc/glm53-{arm}-decode-control:v1" for arm in PARENTS}
SOURCE_NAMES = (
    "__init__.py", "processor.py", "v2_hook.py",
    "sampler-v2-hook.patch", "warmup-v2-hook.patch",
)
SAMPLERS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/sample/sampler.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py",
)
WARMUPS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/warmup.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py",
)
SAMPLER_PATCHED_SHA = "717bdd2203c8977205e16ca63385027f7e7ec8acd54afd962ada3e611cf7b958"
WARMUP_PATCHED_SHA = "de321498f305e2f61d5cfe4701d19a066ebfec83147f527b2ec82bfb653ea8c7"
PACKAGE = "/opt/venv/lib/python3.12/site-packages/p8_decode_capture/"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def image_id(ref: str) -> str:
    return command(["docker", "image", "inspect", ref, "--format", "{{.Id}}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(PARENTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm, output = args.arm, args.output
    parent, tag, dockerfile = PARENTS[arm], TAGS[arm], DOCKERFILES[arm]
    if output != output.resolve() or output.exists():
        raise ValueError("output must be a fresh absolute directory")
    if image_id(parent) != parent:
        raise ValueError("pinned parent image is unavailable")
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        raise ValueError("control image tag already exists; preserve it")
    sources = {name: sha(CONTEXT / name) for name in SOURCE_NAMES}
    sources[dockerfile.name] = sha(dockerfile)
    output.mkdir(parents=True)
    receipt = {
        "schema": "glm53.decode-path-control-image.v1", "status": "started",
        "arm": arm, "parent_image_id": parent, "tag": tag,
        "source_sha256": sources, "started_unix_ns": time.time_ns(),
        "gpu_used": False, "speed_measurement_valid": False,
    }
    build = ["docker", "build", "--pull=false", "--network=none", "-f", str(dockerfile), "-t", tag, str(CONTEXT)]
    receipt["command"] = build
    (output / "plan.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    try:
        with (output / "build.log").open("x") as log:
            subprocess.run(build, env={**os.environ, "DOCKER_BUILDKIT": "0"}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        built = image_id(tag)
        if image_id(parent) != parent or sources != {
            **{name: sha(CONTEXT / name) for name in SOURCE_NAMES}, dockerfile.name: sha(dockerfile)
        }:
            raise ValueError("parent or build source changed")
        paths = [*SAMPLERS, *WARMUPS, *(PACKAGE + n for n in SOURCE_NAMES if n.endswith(".py"))]
        raw = command(["docker", "run", "--rm", "--network=none", "--runtime=runc",
                       "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", built, *paths])
        hashes = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in raw.splitlines()}
        if any(hashes[p] != SAMPLER_PATCHED_SHA for p in SAMPLERS):
            raise ValueError("sampler patch identity differs")
        if any(hashes[p] != WARMUP_PATCHED_SHA for p in WARMUPS):
            raise ValueError("warmup patch identity differs")
        if any(hashes[PACKAGE + n] != sources[n] for n in SOURCE_NAMES if n.endswith(".py")):
            raise ValueError("capture package identity differs")
        (output / "image-source-sha256.txt").write_text(raw + "\n")
        receipt.update(status="complete", image_id=built, image_source_sha256=hashes)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt["completed_unix_ns"] = time.time_ns()
        receipt["build_log_sha256"] = sha(output / "build.log") if (output / "build.log").exists() else None
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm": arm, "image_id": receipt["image_id"], "receipt": str(output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
