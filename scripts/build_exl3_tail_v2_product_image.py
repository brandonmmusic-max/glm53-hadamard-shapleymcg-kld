#!/usr/bin/env python3
"""Build the no-GPU Tail-V2 production-EXL3 capture/speed image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
CONTEXT = REPO / "runtime_patch/p8_tail_repair_v2"
DOCKERFILE = CONTEXT / "Dockerfile.exl3-product"
PATCH = CONTEXT / "kpool-tail-after-valid-history-v2.patch"
PARENT = "sha256:c0f334320c5616392c115279e8a03ac977590ca318cb6f7801fddc0b9c955049"
ORIGINAL = "01bfba91f667214760e1fe8af8c4151498ae9b0407493a4670e255945ae01a58"
TARGETS = (
    "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py",
)
TAG = "klc/glm53-exl3-tail-v2-product:v1a"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def image_id(ref: str) -> str:
    return command(["docker", "image", "inspect", ref, "--format", "{{.Id}}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output != output.resolve() or output.exists():
        raise ValueError("fresh canonical receipt directory required")
    if image_id(PARENT) != PARENT:
        raise ValueError("immutable production-EXL3 capture parent missing")
    if subprocess.run(["docker", "image", "inspect", TAG], capture_output=True).returncode == 0:
        raise ValueError("dedicated image tag already exists")
    sources = {DOCKERFILE.name: sha(DOCKERFILE), PATCH.name: sha(PATCH), Path(__file__).name: sha(Path(__file__))}
    output.mkdir(parents=True, mode=0o700)
    receipt = {"schema": "glm53.exl3-tail-v2-product-image.v1", "status": "started",
               "parent_image_id": PARENT, "tag": TAG, "original_kpool_sha256": ORIGINAL,
               "source_sha256": sources, "started_unix_ns": time.time_ns(),
               "gpu_used": False, "speed_measurement_valid": False}
    build = ["docker", "build", "--pull=false", "--network=none", "-f", str(DOCKERFILE), "-t", TAG, str(CONTEXT)]
    receipt["command"] = build
    (output / "plan.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    try:
        with (output / "build.log").open("x") as log:
            subprocess.run(build, env={**os.environ, "DOCKER_BUILDKIT": "0"}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        built = image_id(TAG)
        raw = command(["docker", "run", "--rm", "--network=none", "--runtime=runc",
                       "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", built, *TARGETS])
        hashes = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in raw.splitlines()}
        if len(set(hashes.values())) != 1 or next(iter(hashes.values())) == ORIGINAL:
            raise ValueError("Tail-V2 source not identically installed in source and site-packages")
        receipt.update(status="complete", image_id=built, patched_kpool_sha256=next(iter(hashes.values())),
                       image_source_sha256=hashes)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt["completed_unix_ns"] = time.time_ns()
        receipt["build_log_sha256"] = sha(output / "build.log") if (output / "build.log").exists() else None
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "image_id": receipt["image_id"],
                      "receipt": str(output / "receipt.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
