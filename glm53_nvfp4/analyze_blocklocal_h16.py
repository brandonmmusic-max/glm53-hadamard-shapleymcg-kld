"""Analyze block-local signed-H16 against fixed H16 and identity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


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
    by_expert = {}
    for row in rows:
        by_expert.setdefault(row["expert"], {})[row["arm"]] = row
    if sorted(by_expert) != sorted(plan["experts"]):
        raise ValueError("expert coverage mismatch")
    arms = ("identity", "fixed-h16", "blocklocal-h16")
    if any(set(items) != set(arms) for items in by_expert.values()):
        raise ValueError("arm coverage mismatch")
    experts = sorted(by_expert)
    values = {arm: [by_expert[e][arm]["evaluation_full_expert_nmse"] for e in experts] for arm in arms}
    candidate, fixed = np.asarray(values["blocklocal-h16"]), np.asarray(values["fixed-h16"])
    log_ratio = np.log(candidate / fixed)
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    indexes = rng.integers(0, len(experts), size=(plan["bootstrap"]["replicates"], len(experts)))
    interval = bca_mean_interval(log_ratio, log_ratio[indexes].mean(axis=1))
    improvement = 1.0 - _gmean(values["blocklocal-h16"]) / _gmean(values["fixed-h16"])
    wins = int((candidate < fixed).sum())
    passed = improvement >= 0.03 and wins >= 12 and interval[1] < 0
    payload = {
        "schema": "glm53-blocklocal-signed-h16-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": plan_sha},
        "inputs": inputs,
        "experts": experts,
        "geometric_mean_nmse": {arm: _gmean(item) for arm, item in values.items()},
        "relative_improvement_vs_fixed_h16": improvement,
        "relative_improvement_vs_identity": 1.0 - _gmean(values["blocklocal-h16"]) / _gmean(values["identity"]),
        "wins_vs_fixed_h16": wins,
        "mean_log_ratio_vs_fixed_ci95_bca": interval,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-build-layer3-pseudoquant-candidate" if passed else "fail-do-not-open-v4-selection",
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
