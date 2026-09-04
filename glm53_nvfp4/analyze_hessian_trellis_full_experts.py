"""Apply the frozen paired-expert decision to causal full-expert NMSE."""
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    raw = json.loads(args.input.read_text())
    values = {
        (int(row["expert"]), row["variant"]): float(row["full_expert_evaluation_nmse"])
        for row in raw["rows"]
    }
    expert_rows = []
    for expert in plan["experts"]:
        candidate = values[(expert, "hessian-mcg-k4")]
        control = values[(expert, "gptq-nvfp4")]
        expert_rows.append({
            "expert": expert,
            "candidate_nmse": candidate,
            "control_nmse": control,
            "log_ratio": math.log(candidate / control),
            "relative_improvement": 1.0 - candidate / control,
        })
    log_ratio = np.asarray([row["log_ratio"] for row in expert_rows])
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    indices = rng.integers(0, len(log_ratio), size=(plan["bootstrap"]["replicates"], len(log_ratio)))
    bootstrap = log_ratio[indices].mean(axis=1)
    interval = bca_mean_interval(log_ratio, bootstrap)
    geometric_ratio = math.exp(float(log_ratio.mean()))
    wins = sum(row["relative_improvement"] > 0 for row in expert_rows)
    passed = 1.0 - geometric_ratio >= 0.10 and interval[1] < 0 and wins >= 12
    payload = {
        "schema": "glm53-hessian-trellis-full-16expert-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "experts": expert_rows,
        "aggregate": {
            "expert_count": len(expert_rows),
            "expert_wins": wins,
            "geometric_mean_ratio": geometric_ratio,
            "geometric_mean_relative_improvement": 1.0 - geometric_ratio,
            "mean_log_ratio_ci95_bca": interval,
        },
        "bootstrap": plan["bootstrap"],
        "decision": (
            f"pass-build-layer{plan['layer']}-kld-candidate"
            if passed
            else f"fail-do-not-build-layer{plan['layer']}"
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], sort_keys=True))
    print(payload["decision"])


if __name__ == "__main__":
    main()
