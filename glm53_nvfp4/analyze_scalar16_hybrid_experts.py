"""Apply the frozen paired-expert gate to scalar16/MCG hybrid raw parts."""
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
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    rows = []
    inputs = []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != sha256_file(args.plan):
            raise RuntimeError("raw part does not belong to the frozen plan")
        rows.extend(raw["rows"])
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    values = {}
    detail = {}
    for row in rows:
        key = (int(row["expert"]), row["variant"])
        if key in values:
            raise RuntimeError(f"duplicate raw cell {key}")
        values[key] = float(row["full_expert_evaluation_nmse"])
        detail[key] = row
    expected = {(expert, variant) for expert in plan["experts"] for variant in ("mcg", "scalar16", "hybrid", "gptq-nvfp4")}
    if set(values) != expected:
        raise RuntimeError(f"raw coverage mismatch missing={sorted(expected-set(values))} extra={sorted(set(values)-expected)}")
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    comparisons = {}
    for control in ("gptq-nvfp4", "mcg", "scalar16"):
        ratios = np.asarray([math.log(values[(expert, "hybrid")] / values[(expert, control)]) for expert in plan["experts"]])
        indices = rng.integers(0, len(ratios), size=(plan["bootstrap"]["replicates"], len(ratios)))
        bootstrap = ratios[indices].mean(1)
        geometric_ratio = math.exp(float(ratios.mean()))
        comparisons[control] = {
            "geometric_mean_ratio": geometric_ratio,
            "geometric_mean_relative_improvement": 1.0 - geometric_ratio,
            "wins": int(sum(values[(expert, "hybrid")] < values[(expert, control)] for expert in plan["experts"])),
            "mean_log_ratio_ci95_bca": bca_mean_interval(ratios, bootstrap),
        }
    passed = (
        comparisons["gptq-nvfp4"]["geometric_mean_relative_improvement"] >= 0.10
        and comparisons["mcg"]["geometric_mean_relative_improvement"] >= 0.05
        and comparisons["gptq-nvfp4"]["wins"] >= 12
        and comparisons["mcg"]["wins"] >= 12
        and comparisons["gptq-nvfp4"]["mean_log_ratio_ci95_bca"][1] < 0
        and comparisons["mcg"]["mean_log_ratio_ci95_bca"][1] < 0
    )
    payload = {
        "schema": "glm53-p8-scalar16-mcg-hybrid-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "inputs": inputs,
        "geometric_mean_nmse": {variant: math.exp(float(np.mean([math.log(values[(expert, variant)]) for expert in plan["experts"]]))) for variant in ("hybrid", "scalar16", "mcg", "gptq-nvfp4")},
        "comparisons": comparisons,
        "experts": [{"expert": expert, "hybrid_policy": detail[(expert, "hybrid")]["policy"], **{f"{variant}_nmse": values[(expert, variant)] for variant in ("hybrid", "scalar16", "mcg", "gptq-nvfp4")}} for expert in plan["experts"]],
        "decision": "pass-freeze-hybrid-and-open-one-n32-teacher-kld-role" if passed else "fail-do-not-open-protected-kld",
        "protected_roles_opened": [],
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"geometric_mean_nmse": payload["geometric_mean_nmse"], "comparisons": comparisons, "decision": payload["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
