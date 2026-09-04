"""Analyze and rank the frozen P8 block-Hessian layer sweep."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    results = []
    for layer in plan["layers"]:
        path = args.root / f"layer-{layer:03d}" / "raw.json"
        raw = json.loads(path.read_text())
        ratios = np.asarray([float(row["log_ratio_gmean"]) for row in raw["rows"]])
        indices = rng.integers(0, len(ratios), size=(plan["bootstrap"]["replicates"], len(ratios)))
        interval = bca_mean_interval(ratios, ratios[indices].mean(1))
        geometric_ratio = math.exp(float(ratios.mean()))
        improvement = 1.0 - geometric_ratio
        wins = int((ratios < 0).sum())
        passed = improvement >= 0.10 and interval[1] < 0 and wins >= 12
        results.append({
            "layer": layer,
            "geometric_mean_relative_improvement": improvement,
            "expert_wins": wins,
            "mean_log_ratio_ci95_bca": interval,
            "decision": "pass-screen-positive" if passed else "fail-screen-negative",
            "raw_sha256": sha256_file(path),
        })
    ranking = sorted(results, key=lambda row: row["geometric_mean_relative_improvement"], reverse=True)
    positive = [row["layer"] for row in ranking if row["decision"].startswith("pass")]
    payload = {
        "schema": "glm53-p8-blockhessian-layer-sweep-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "layers": len(results),
        "screen_positive_layers_ranked": positive,
        "ranking": ranking,
        "decision": "pass-design-prior" if positive else "fail-no-screen-positive-layer",
        "interpretation_boundary": plan["interpretation_boundary"],
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"screen_positive_layers_ranked": positive, "top10": ranking[:10]}, indent=2))


if __name__ == "__main__":
    main()
