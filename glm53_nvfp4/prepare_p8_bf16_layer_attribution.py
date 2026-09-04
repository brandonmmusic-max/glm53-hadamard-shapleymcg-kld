"""Freeze BF16-matched per-layer P8 KLD attribution on the adaptive role."""
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
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--corrected-composite-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("expected exact n=32 conditional-fit role")
    if any(roles["roles"][r] for r in ("selection", "confirmation", "final")):
        raise RuntimeError("protected role exposed")
    candidates = {}
    for path in args.candidate:
        receipt = json.loads((path / "BF16_WEIGHT_OVERLAY_RECEIPT.json").read_text())
        if len(receipt["layers"]) != 1 or not receipt.get("config_unchanged"):
            raise RuntimeError(f"candidate is not one BF16-matched layer: {path}")
        layer = int(receipt["layers"][0])
        candidates[str(layer)] = {
            "model": str(path),
            "receipt": _file(path / "BF16_WEIGHT_OVERLAY_RECEIPT.json"),
        }
    if set(candidates) != {"3", "19", "20"}:
        raise RuntimeError(f"expected layers 3,19,20; got {sorted(candidates)}")
    payload = {
        "schema": "glm53-p8-bf16-matched-layer-attribution-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit adaptive attribution",
        "windows": 32,
        "baseline": {
            "model": str(args.baseline),
            "receipt": _file(args.baseline / "BF16_LAYER_RECEIPT.json"),
        },
        "candidates": candidates,
        "run_order": ["3", "19", "20"],
        "estimand": "per-layer candidate minus matched decoded-GPTQ BF16-overlay KLD",
        "decision_rule": "a layer is supported only if mean delta is at most -0.0004 nats and its Bonferroni-adjusted 98.333 percent BCa interval upper bound is below zero",
        "multiplicity": "three comparisons; family alpha 0.05, individual two-sided alpha 1/60",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090425},
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "no selection, confirmation, or final data are read",
        "inputs": {
            "roles": _file(args.roles),
            "corrected_composite_analysis": _file(args.corrected_composite_analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
