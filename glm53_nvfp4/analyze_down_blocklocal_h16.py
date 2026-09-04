"""Analyze a down-only block-local H16 screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


ARMS = ("identity-down", "fixed-h16-down", "blocklocal-h16-down")


def _gmean(values: list[float]) -> float:
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
    by_expert: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_expert.setdefault(row["expert"], {})[row["arm"]] = row
    if sorted(by_expert) != sorted(plan["experts"]):
        raise ValueError("expert coverage mismatch")
    if any(set(items) != set(ARMS) for items in by_expert.values()):
        raise ValueError("arm coverage mismatch")
    experts = sorted(by_expert)
    output_values = {
        arm: [by_expert[e][arm]["evaluation_full_expert_nmse"] for e in experts]
        for arm in ARMS
    }
    weight_values = {
        arm: [by_expert[e][arm]["down_weight_nmse"] for e in experts]
        for arm in ARMS
    }
    candidate = np.asarray(output_values["blocklocal-h16-down"])
    fixed = np.asarray(output_values["fixed-h16-down"])
    log_ratio = np.log(candidate / fixed)
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    indexes = rng.integers(0, len(experts), size=(plan["bootstrap"]["replicates"], len(experts)))
    interval = bca_mean_interval(log_ratio, log_ratio[indexes].mean(axis=1))
    improvement = 1.0 - _gmean(candidate.tolist()) / _gmean(fixed.tolist())
    wins = int((candidate < fixed).sum())
    passed = improvement >= 0.03 and wins >= 20 and interval[1] < 0
    payload = {
        "schema": "glm53-down-blocklocal-h16-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": plan_sha},
        "inputs": inputs,
        "experts": experts,
        "down_weight_geometric_mean_nmse": {arm: _gmean(values) for arm, values in weight_values.items()},
        "routed_output_geometric_mean_nmse": {arm: _gmean(values) for arm, values in output_values.items()},
        "routed_output_relative_improvement_vs_fixed_h16": improvement,
        "routed_output_relative_improvement_vs_identity": 1.0 - _gmean(candidate.tolist()) / _gmean(output_values["identity-down"]),
        "wins_vs_fixed_h16": wins,
        "mean_log_ratio_vs_fixed_ci95_bca": interval,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-build-layer3-candidate" if passed else "fail-do-not-open-new-teacher-role",
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
