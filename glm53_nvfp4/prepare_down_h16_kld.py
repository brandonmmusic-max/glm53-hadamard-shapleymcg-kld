"""Freeze the protected V4 layer-3 down-only H16 KLD comparison."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--stock-index", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    validation = json.loads(args.validation.read_text())
    if validation["decision"] != "pass-build-layer3-down-h16-kld-candidate":
        raise ValueError("down-only validation did not pass")
    roles = json.loads(args.roles.read_text())
    if roles["counts"]["selection"] != 32 or roles["counts"]["confirmation"] != 28:
        raise ValueError("unexpected V4 protected-role counts")
    payload = {
        "schema": "glm53-down-h16-v4-kld-freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "selection",
        "selection_wave": 1,
        "windows": 32,
        "arms": {
            "stock": {"rotation": "identity", "layers": "3", "scope": "gate-up"},
            "candidate": {"rotation": "had16", "layers": "3", "scope": "mid-only"},
        },
        "runtime": {
            "image_id": args.image_id,
            "load_format": "instanttensor",
            "moe_backend": "humming",
            "activation_dtype": "bf16",
            "tp": 4,
            "ep": 4,
            "dcp": 4,
            "mtp": 0,
        },
        "estimand": "paired per-window candidate minus stock teacher KLD and relative geometric-mean KLD improvement",
        "bootstrap": {"unit": "window", "method": "BCa", "replicates": 20000, "seed": 2026090409},
        "decision_rule": "qualify down-only H16 as beating stock NVFP4 only if relative geometric-mean KLD improvement is at least 3 percent, the paired BCa upper bound for candidate-minus-stock mean KLD is below zero, and at least three of four domain mean deltas are negative",
        "confirmation_boundary": "do not open the 28 confirmation logits for this layer-only rotation screen",
        "run_ids": {"stock": "v4-wave1-downh16-stock", "candidate": "v4-wave1-downh16-candidate"},
        "inputs": {
            "validation": _file(args.validation),
            "roles": _file(args.roles),
            "candidate_index": _file(args.candidate_index),
            "stock_index": _file(args.stock_index),
            "runtime_manifest": _file(args.runtime_manifest),
            "runner": _file(args.runner),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
