"""Aggregate the preregistered exhaustive causal P8 layer screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    rows = []
    for layer in plan["layers"]:
        path = args.root / f"layer-{layer:03d}" / "analysis.json"
        analysis = json.loads(path.read_text())
        aggregate = analysis["aggregate"]
        rows.append({
            "layer": layer,
            "geometric_mean_relative_improvement": aggregate["geometric_mean_relative_improvement"],
            "expert_wins": aggregate["expert_wins"],
            "mean_log_ratio_ci95_bca": aggregate["mean_log_ratio_ci95_bca"],
            "decision": analysis["decision"],
            "analysis_sha256": sha256_file(path),
        })
    ranking = sorted(rows, key=lambda row: row["geometric_mean_relative_improvement"], reverse=True)
    eligible = [row["layer"] for row in ranking if row["decision"].startswith("pass-")]
    payload = {
        "schema": "glm53-p8-causal-all-layer-sweep-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "layers": len(rows),
        "eligible_layers_ranked": eligible,
        "ranking": ranking,
        "decision": "pass-layer-prior-for-subset-design" if eligible else "fail-no-causal-layer-eligible",
        "interpretation_boundary": "causal local NMSE is a design prior, not end-to-end KLD or Shapley attribution",
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"eligible_layers_ranked": eligible, "top10": ranking[:10]}, indent=2))


if __name__ == "__main__":
    main()
