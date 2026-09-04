"""Analyze the disjoint-expert down-only H16 validation."""
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
            raise ValueError(f"invalid input {path}")
        rows.extend(raw["rows"]); inputs.append({"path": str(path), "sha256": sha256_file(path)})
    by_expert = {}
    for row in rows: by_expert.setdefault(row["expert"], {})[row["arm"]] = row
    arms = tuple(plan["arms"])
    if sorted(by_expert) != sorted(plan["experts"]) or any(set(v) != set(arms) for v in by_expert.values()):
        raise ValueError("coverage mismatch")
    experts = sorted(by_expert)
    values = {arm: [by_expert[e][arm]["evaluation_full_expert_nmse"] for e in experts] for arm in arms}
    down = np.asarray(values["down-h16"]); identity = np.asarray(values["identity"]); all_h = np.asarray(values["all-h16"])
    log_identity, log_all = np.log(down / identity), np.log(down / all_h)
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    choices = rng.integers(0, len(experts), size=(plan["bootstrap"]["replicates"], len(experts)))
    identity_ci = bca_mean_interval(log_identity, log_identity[choices].mean(1))
    all_ci = bca_mean_interval(log_all, log_all[choices].mean(1))
    improvement = 1.0 - _gmean(values["down-h16"]) / _gmean(values["identity"])
    wins = int((down < identity).sum())
    passed = improvement >= 0.05 and wins >= 20 and identity_ci[1] < 0 and all_ci[1] < math.log(1.02)
    payload = {"schema": "glm53-down-h16-disjoint-validation-analysis.v1", "plan": {"path": str(args.plan), "sha256": plan_sha}, "layer": plan["layer"], "inputs": inputs, "experts": experts, "geometric_mean_nmse": {arm: _gmean(v) for arm, v in values.items()}, "down_h16_vs_identity": {"relative_improvement": improvement, "wins": wins, "mean_log_ratio_ci95_bca": identity_ci}, "down_h16_vs_all_h16": {"relative_improvement": 1.0 - _gmean(values["down-h16"]) / _gmean(values["all-h16"]), "wins": int((down < all_h).sum()), "mean_log_ratio_ci95_bca": all_ci}, "decision_rule": plan["decision_rule"], "decision": f"pass-build-layer-{plan['layer']:03d}-down-h16-candidate" if passed else "fail-do-not-open-new-protected-role", "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
