"""Freeze the two-arm layer attribution KLD diagnostic for P8."""
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
    parser.add_argument("--candidate19", type=Path, required=True)
    parser.add_argument("--candidate20", type=Path, required=True)
    parser.add_argument("--composite-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("expected n=32 conditional-fit role")
    if any(roles["roles"][r] for r in ("selection", "confirmation", "final")):
        raise RuntimeError("protected roles exposed")
    payload = {
        "schema": "glm53-p8-layer19-20-ablation-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit adaptive diagnostic",
        "windows": 32,
        "arms": {
            "baseline": _file(args.baseline / "model.safetensors.index.json"),
            "layer19": _file(args.candidate19 / "BF16_LAYER_RECEIPT.json"),
            "layer20": _file(args.candidate20 / "BF16_LAYER_RECEIPT.json"),
        },
        "run_order": ["layer19", "layer20"],
        "estimands": ["layer19 minus baseline KLD", "layer20 minus baseline KLD"],
        "decision_rule": "an individual layer advances only if mean delta is at most -0.0014 nats and its multiplicity-adjusted 97.5 percent BCa interval upper bound is below zero",
        "multiplicity": "two comparisons; Bonferroni family alpha 0.05, individual two-sided alpha 0.025",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090424},
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "no selection, confirmation, or final data are read; the 28 reserved confirmation logits remain unopened",
        "inputs": {
            "roles": _file(args.roles),
            "failed_composite_analysis": _file(args.composite_analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
