"""Paired expert-bootstrap decision for the learned T12 family law."""
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
    rows = []
    learned_mcg = []
    learned_gptq = []
    for expert in plan["experts"]:
        learned = values[(expert, "learned-t12-k4")]
        mcg = values[(expert, "hessian-mcg-k4")]
        gptq = values[(expert, "gptq-nvfp4")]
        learned_mcg.append(math.log(learned / mcg))
        learned_gptq.append(math.log(learned / gptq))
        rows.append({
            "expert": expert,
            "learned_nmse": learned,
            "mcg_nmse": mcg,
            "gptq_nmse": gptq,
            "improvement_vs_mcg": 1.0 - learned / mcg,
            "improvement_vs_gptq": 1.0 - learned / gptq,
        })
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    replicates = plan["bootstrap"]["replicates"]
    results = {}
    for label, raw_ratios in (("mcg", learned_mcg), ("gptq", learned_gptq)):
        ratios = np.asarray(raw_ratios)
        indices = rng.integers(0, len(ratios), size=(replicates, len(ratios)))
        bootstrap = ratios[indices].mean(axis=1)
        mean = float(ratios.mean())
        results[label] = {
            "geometric_mean_ratio": math.exp(mean),
            "geometric_mean_relative_improvement": 1.0 - math.exp(mean),
            "mean_log_ratio_ci95_bca": bca_mean_interval(ratios, bootstrap),
        }
    wins = sum(row["improvement_vs_mcg"] > 0 for row in rows)
    passed = (
        results["mcg"]["geometric_mean_relative_improvement"] >= 0.15
        and results["gptq"]["geometric_mean_relative_improvement"] >= 0.20
        and results["mcg"]["mean_log_ratio_ci95_bca"][1] < 0
        and results["gptq"]["mean_log_ratio_ci95_bca"][1] < 0
        and wins >= 12
    )
    payload = {
        "schema": "glm53-learned-t12-16expert-validation-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "experts": rows,
        "aggregate": {"versus": results, "expert_wins_vs_mcg": wins},
        "decision": "pass-build-learned-layer3" if passed else "fail-do-not-build",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], sort_keys=True))
    print(payload["decision"])


if __name__ == "__main__":
    main()
