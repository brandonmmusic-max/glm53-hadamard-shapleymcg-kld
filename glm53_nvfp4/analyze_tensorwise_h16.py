"""Analyze the frozen eight-expert tensorwise signed-H16 pilot."""
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
    plan = json.loads(args.plan.read_text())
    rows = []
    inputs = []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != sha256_file(args.plan):
            raise ValueError(f"plan mismatch in {path}")
        if raw["protected_roles_opened"]:
            raise ValueError(f"protected role opened in {path}")
        rows.extend(raw["rows"])
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    by_expert = {}
    for row in rows:
        by_expert.setdefault(row["expert"], {})[row["arm"]] = row
    if sorted(by_expert) != sorted(plan["experts"]):
        raise ValueError("expert coverage mismatch")
    arms = ("tensorwise-signed-h16", "identity-gptq", "fixed-h16-gptq")
    for expert, arm_rows in by_expert.items():
        if set(arm_rows) != set(arms):
            raise ValueError(f"arm coverage mismatch for expert {expert}")
    values = {
        arm: [by_expert[expert][arm]["evaluation_full_expert_nmse"] for expert in sorted(by_expert)]
        for arm in arms
    }
    candidate = values["tensorwise-signed-h16"]
    identity = values["identity-gptq"]
    fixed = values["fixed-h16-gptq"]
    improvement_identity = 1.0 - _gmean(candidate) / _gmean(identity)
    improvement_fixed = 1.0 - _gmean(candidate) / _gmean(fixed)
    wins_identity = sum(a < b for a, b in zip(candidate, identity, strict=True))
    wins_fixed = sum(a < b for a, b in zip(candidate, fixed, strict=True))
    log_ratio_identity = np.log(
        np.asarray(candidate, dtype=np.float64) / np.asarray(identity, dtype=np.float64)
    )
    log_ratio_fixed = np.log(
        np.asarray(candidate, dtype=np.float64) / np.asarray(fixed, dtype=np.float64)
    )
    bootstrap_spec = plan.get(
        "bootstrap", {"replicates": 20000, "seed": 2026090401, "unit": "expert"}
    )
    rng = np.random.default_rng(bootstrap_spec["seed"])
    indices = rng.integers(
        0,
        len(candidate),
        size=(bootstrap_spec["replicates"], len(candidate)),
    )
    identity_interval = bca_mean_interval(
        log_ratio_identity, log_ratio_identity[indices].mean(axis=1)
    )
    fixed_interval = bca_mean_interval(
        log_ratio_fixed, log_ratio_fixed[indices].mean(axis=1)
    )
    thresholds = plan.get("thresholds", {
        "minimum_improvement_vs_identity": 0.05,
        "minimum_wins_vs_identity": 6,
        "require_improvement_vs_fixed_h16": True,
        "require_identity_bca_upper_below_zero": False,
    })
    passed = (
        improvement_identity >= thresholds["minimum_improvement_vs_identity"]
        and wins_identity >= thresholds["minimum_wins_vs_identity"]
        and (
            not thresholds.get("require_improvement_vs_fixed_h16", True)
            or improvement_fixed > 0
        )
        and (
            not thresholds.get("require_identity_bca_upper_below_zero", False)
            or identity_interval[1] < 0
        )
    )
    payload = {
        "schema": "glm53-tensorwise-signed-h16-pilot-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "inputs": inputs,
        "experts": sorted(by_expert),
        "geometric_mean_nmse": {arm: _gmean(value) for arm, value in values.items()},
        "relative_improvement_vs_identity": improvement_identity,
        "relative_improvement_vs_fixed_h16": improvement_fixed,
        "wins_vs_identity": wins_identity,
        "wins_vs_fixed_h16": wins_fixed,
        "mean_log_ratio_ci95_bca": {
            "vs_identity": identity_interval,
            "vs_fixed_h16": fixed_interval,
        },
        "bootstrap": bootstrap_spec,
        "thresholds": thresholds,
        "decision": plan.get("pass_decision", "pass-scale-to-16") if passed else plan.get("fail_decision", "fail-do-not-open-v4-selection"),
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
