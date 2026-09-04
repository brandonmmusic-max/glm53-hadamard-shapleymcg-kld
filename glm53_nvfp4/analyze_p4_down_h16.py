"""Analyze P4 down-H16 against matched NVFP4 controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


ARMS = ("identity-gptq", "down-h16-gptq", "p4-down-h16")


def _gmean(values) -> float:
    return float(math.exp(np.log(np.asarray(values, dtype=np.float64)).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan, plan_sha = json.loads(args.plan.read_text()), sha256_file(args.plan)
    rows, inputs = [], []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != plan_sha or raw["protected_roles_opened"]:
            raise ValueError(f"invalid raw input {path}")
        rows.extend(raw["rows"])
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    by_expert = {}
    for row in rows:
        by_expert.setdefault(row["expert"], {})[row["arm"]] = row
    if sorted(by_expert) != sorted(plan["experts"]) or any(set(v) != set(ARMS) for v in by_expert.values()):
        raise ValueError("expert/arm coverage mismatch")
    experts = sorted(by_expert)
    output = {arm: [by_expert[e][arm]["evaluation_full_expert_nmse"] for e in experts] for arm in ARMS}
    weight = {arm: [by_expert[e][arm]["down_weight_nmse"] for e in experts] for arm in ARMS}
    candidate = np.asarray(output["p4-down-h16"])
    comparisons = {}
    passed = True
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    indexes = rng.integers(0, len(experts), size=(plan["bootstrap"]["replicates"], len(experts)))
    for control, threshold in (("identity-gptq", 0.05), ("down-h16-gptq", 0.01)):
        values = np.asarray(output[control])
        log_ratio = np.log(candidate / values)
        interval = bca_mean_interval(log_ratio, log_ratio[indexes].mean(axis=1))
        improvement = 1.0 - _gmean(candidate) / _gmean(values)
        wins = int((candidate < values).sum())
        comparisons[control] = {
            "relative_improvement": improvement,
            "wins": wins,
            "mean_log_ratio_ci95_bca": interval,
        }
        passed &= improvement >= threshold and wins >= 12 and interval[1] < 0
    payload = {
        "schema": "glm53-p4-down-h16-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": plan_sha},
        "inputs": inputs,
        "experts": experts,
        "down_weight_geometric_mean_nmse": {arm: _gmean(values) for arm, values in weight.items()},
        "routed_output_geometric_mean_nmse": {arm: _gmean(values) for arm, values in output.items()},
        "comparisons": comparisons,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-build-full-layer-p4" if passed else "fail-do-not-open-new-teacher-role",
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
