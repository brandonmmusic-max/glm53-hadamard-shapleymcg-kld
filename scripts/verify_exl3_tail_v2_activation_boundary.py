#!/usr/bin/env python3
"""Append-only CPU verification of the v1a EXL3 BF16 kernel boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


IMAGE = "sha256:af4e6a9ac0feb29ae2399ea45fad7db189594f0ff564c5bff3a691156c60b6b7"
SOURCE = "/opt/infernal-invocation/vllm/vllm/model_executor/layers/quantization/exl3.py"
SOURCE_SHA256 = "ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != args.output.resolve() or args.output.exists():
        raise ValueError("fresh canonical output required")
    image_receipt = json.loads(args.image_receipt.read_text())
    if (image_receipt.get("status") != "complete" or image_receipt.get("image_id") != IMAGE
            or image_receipt.get("gpu_used") is not False):
        raise ValueError("v1a image receipt differs")
    actual_image = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if actual_image != IMAGE:
        raise ValueError("immutable v1a image unavailable")
    raw = subprocess.check_output([
        "docker", "run", "--rm", "--network=none", "--runtime=runc",
        "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", IMAGE, SOURCE,
    ], text=True).strip()
    source_sha, source_path = raw.split(maxsplit=1)
    if source_path != SOURCE or source_sha != SOURCE_SHA256:
        raise ValueError("EXL3 source identity differs")
    payload = {
        "schema": "glm53.exl3-tail-v2-activation-boundary.v1",
        "status": "pass",
        "image_id": IMAGE,
        "image_receipt": str(args.image_receipt),
        "image_receipt_sha256": sha(args.image_receipt),
        "source": SOURCE,
        "source_sha256": source_sha,
        "source_contract": [
            "create_weights rejects activation parameter dtypes outside BF16/FP16",
            "runtime dispatch selects rotation_input_dtype=bf16 exactly when x.dtype is torch.bfloat16",
        ],
        "runtime_gate": "server log must also state dtype=torch.bfloat16, quantization=exl3, and moe_backend=b12x",
        "gpu_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "source_sha256": source_sha,
                      "receipt_sha256": sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
