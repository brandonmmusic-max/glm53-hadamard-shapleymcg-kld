#!/usr/bin/env python3
"""Build the CPU-only capture image for P8 tail repair V2."""
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
PARENT = "sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9"
ORIGINAL = "01bfba91f667214760e1fe8af8c4151498ae9b0407493a4670e255945ae01a58"
TARGET = "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py"
SOURCES = ("Dockerfile", "kpool-tail-after-valid-history-v2.patch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def image_id(name: str) -> str:
    return output(["docker", "image", "inspect", name, "--format", "{{.Id}}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="klc/glm53-p8-tail-repair:v2")
    args = parser.parse_args()
    if args.output != args.output.resolve() or args.output.exists():
        raise ValueError("fresh canonical output directory required")
    if args.tag != "klc/glm53-p8-tail-repair:v2":
        raise ValueError("exact dedicated V2 tag required")
    if subprocess.run(["docker", "image", "inspect", args.tag], capture_output=True).returncode == 0:
        raise ValueError("tail-repair V2 image tag already exists")
    if image_id(PARENT) != PARENT:
        raise ValueError("capture-capable P8 parent image is unavailable")
    sources = {name: sha(CONTEXT / name) for name in SOURCES}
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "glm53.p8-tail-repair-image.v2", "status": "started",
        "started_unix_ns": time.time_ns(), "parent_image_id": PARENT,
        "original_kpool_sha256": ORIGINAL, "tag": args.tag,
        "source_sha256": sources, "builder_sha256": sha(Path(__file__)),
        "gpu_used": False, "speed_measurement_valid": False,
    }
    build = ["docker", "build", "--pull=false", "--network=none", "--tag", args.tag, str(CONTEXT)]
    receipt["command"] = build
    (args.output / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (args.output / "build.log").open("x") as log:
            subprocess.run(build, env={**os.environ, "DOCKER_BUILDKIT": "0"}, stdout=log, stderr=subprocess.STDOUT, check=True)
        built = image_id(args.tag)
        line = output(["docker", "run", "--rm", "--network=none", "--runtime=runc", "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", built, TARGET])
        patched_sha, installed = line.split(maxsplit=1)
        if installed != TARGET or patched_sha == ORIGINAL or image_id(PARENT) != PARENT:
            raise ValueError("tail-repair V2 installation identity differs")
        marker = output(["docker", "run", "--rm", "--network=none", "--runtime=runc", "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "grep", built, "-c", "P8 tail-after-valid-history-v2", TARGET])
        if marker != "1":
            raise ValueError("tail-repair V2 source marker differs")
        receipt.update(status="complete", image_id=built, patched_kpool_sha256=patched_sha)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["completed_unix_ns"] = time.time_ns()
        receipt["build_log_sha256"] = sha(args.output / "build.log") if (args.output / "build.log").exists() else None
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"status": receipt["status"], "image_id": receipt["image_id"], "receipt": str(args.output / "receipt.json")}))


if __name__ == "__main__":
    main()
