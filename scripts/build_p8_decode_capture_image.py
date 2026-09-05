#!/usr/bin/env python3
"""Build a separate, CPU-only V2 capture image with immutable build receipts."""
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
PARENT_TAG = "klc/glm53-p8-fc1tiles:cropped-v1"
PARENT_ID = "sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141"
SOURCE_NAMES = ("Dockerfile", "__init__.py", "processor.py", "v2_hook.py", "sampler-v2-hook.patch", "warmup-v2-hook.patch")
SAMPLERS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/sample/sampler.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py",
)
SAMPLER_PATCHED_SHA = "717bdd2203c8977205e16ca63385027f7e7ec8acd54afd962ada3e611cf7b958"
WARMUPS = (
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/warmup.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py",
)
WARMUP_ORIGINAL_SHA = "696cdd462f58908f3511e9983fe18aa3811acecdd6da1ff580bbab99aed08dcc"
WARMUP_PATCHED_SHA = "de321498f305e2f61d5cfe4701d19a066ebfec83147f527b2ec82bfb653ea8c7"
PACKAGE = "/opt/venv/lib/python3.12/site-packages/p8_decode_capture/"
INHERITED_FC2 = "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py"
INHERITED_FC2_SHA = "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def image_id(tag: str) -> str:
    return command(["docker", "image", "inspect", tag, "--format", "{{.Id}}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="klc/glm53-p8-capture:v2-v2")
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("output must be an absolute, fresh directory")
    if args.tag == PARENT_TAG or not args.tag.startswith("klc/glm53-p8-capture:"):
        raise ValueError("tag must belong to the separate klc/glm53-p8-capture namespace")
    existing = subprocess.run(["docker", "image", "inspect", args.tag], capture_output=True)
    if existing.returncode == 0:
        raise ValueError("capture tag already exists; preserve it and choose a new version")
    if image_id(PARENT_TAG) != PARENT_ID:
        raise ValueError("local parent tag identity mismatch")
    sources = {name: sha(CONTEXT / name) for name in SOURCE_NAMES}
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "glm53-p8.decode-capture-image.v2",
        "status": "started", "started_unix_ns": time.time_ns(),
        "parent_tag": PARENT_TAG, "parent_image_id": PARENT_ID,
        "tag": args.tag, "source_sha256": sources,
        "builder_sha256": sha(Path(__file__)),
        "gpu_used": False, "speed_measurement_valid": False,
    }
    build = ["docker", "build", "--pull=false", "--network=none", "--tag", args.tag, str(CONTEXT)]
    receipt["command"] = build
    (args.output / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        # Disable remote BuildKit resolution of the authenticated local base.
        env = dict(os.environ, DOCKER_BUILDKIT="0")
        with (args.output / "build.log").open("x") as log:
            subprocess.run(build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        built = image_id(args.tag)
        if image_id(PARENT_TAG) != PARENT_ID:
            raise ValueError("parent tag changed during build")
        if sources != {name: sha(CONTEXT / name) for name in SOURCE_NAMES}:
            raise ValueError("capture sources changed during build")
        paths = [*SAMPLERS, *WARMUPS, *(PACKAGE + name for name in SOURCE_NAMES if name.endswith(".py")), INHERITED_FC2]
        # No GPU device allocation and no serving entrypoint. Network is disabled.
        output = command(["docker", "run", "--rm", "--network=none", "--runtime=runc",
                          "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", built, *paths])
        (args.output / "image-source-sha256.txt").write_text(output + "\n")
        hashes = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in output.splitlines()}
        if any(hashes[path] != SAMPLER_PATCHED_SHA for path in SAMPLERS):
            raise ValueError("exact sampler patch was not installed in both copies")
        if any(hashes[path] != WARMUP_PATCHED_SHA for path in WARMUPS):
            raise ValueError("warmup patch was not installed in both copies")
        for name in SOURCE_NAMES:
            if name.endswith(".py") and hashes[PACKAGE + name] != sources[name]:
                raise ValueError(f"package source mismatch: {name}")
        if hashes[INHERITED_FC2] != INHERITED_FC2_SHA:
            raise ValueError("inherited FC2 changed")
        receipt.update(status="complete", image_id=built, image_source_sha256=hashes)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["completed_unix_ns"] = time.time_ns()
        receipt["build_log_sha256"] = sha(args.output / "build.log") if (args.output / "build.log").exists() else None
        with (args.output / "receipt.json").open("x") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"status": receipt["status"], "image_id": receipt["image_id"], "receipt": str(args.output / "receipt.json")}))


if __name__ == "__main__":
    main()
