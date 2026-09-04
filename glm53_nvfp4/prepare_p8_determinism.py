"""Freeze the five-cold-run P8 split determinism gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260944)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    payload = {
        "schema": "glm53-p8-mcg-split-determinism-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-runs",
        "decision_question": "Are all four K3/K4 x M33/M64 output tensors bitwise identical across five cold container runs?",
        "runs": 5,
        "seed": args.seed,
        "image": {"tag": args.image, "id": image_id},
        "probe_sha256": sha256_file(repo / "glm53_nvfp4/probe_p8_mcg_moe.py"),
        "decision_rule": "pass only if every run passes arithmetic closure and each cell's output SHA-256 is identical across all five runs",
        "role": "developmental device determinism; no teacher logits or protected roles",
        "ldlq": False,
        "strict_kld_gate_relaxed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
