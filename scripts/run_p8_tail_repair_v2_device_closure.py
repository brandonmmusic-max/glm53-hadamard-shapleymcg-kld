#!/usr/bin/env python3
"""Run and preserve the V2 tail selector's GPU per-row closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts/verify_p8_tail_repair_v2_device.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output != args.output.resolve() or args.output.exists() or args.image_receipt != args.image_receipt.resolve():
        raise ValueError("canonical fresh device-closure paths required")
    image_receipt = json.loads(args.image_receipt.read_text())
    image = image_receipt.get("image_id", "")
    if image_receipt.get("status") != "complete" or not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("complete immutable V2 image receipt required")
    command = [
        "docker", "run", "--rm", "--network=none", "--gpus", "device=0",
        "--volume", f"{VERIFY}:/opt/p8-tail-v2-verify.py:ro",
        "--entrypoint", "/opt/venv/bin/python", image, "/opt/p8-tail-v2-verify.py",
    ]
    started = time.time_ns()
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload.get("status") != "pass" or payload.get("differing_rows") != 1536 or payload.get("unchanged_rows") != 511 or payload.get("diff_count_by_L_mod_4") != {"0": 0, "1": 512, "2": 512, "3": 512}:
        raise ValueError("device row-selectivity closure differs")
    receipt = {
        **payload, "image_id": image, "image_receipt": str(args.image_receipt),
        "image_receipt_sha256": sha(args.image_receipt), "verifier": str(VERIFY),
        "verifier_sha256": sha(VERIFY), "runner_sha256": sha(Path(__file__)),
        "started_unix_ns": started, "completed_unix_ns": time.time_ns(),
        "stderr": completed.stderr, "gpu_used": True, "kld_run": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "pass", "receipt": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
