"""Analyze projection-family H16 attribution with frozen multiplicity control."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _gmean(values: list[float]) -> float:
    return float(math.exp(np.log(np.asarray(values, dtype=np.float64)).mean()))


def _pvalue(values: np.ndarray) -> float:
    observed, exceed = abs(float(values.mean())), 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return exceed / (2 ** len(values))


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
    baseline = np.asarray(values["all-h16"])
    stats, raw_p = {}, {}
    for index, arm in enumerate(("gate-up-h16", "down-h16")):
        candidate = np.asarray(values[arm]); ratios = np.log(candidate / baseline)
        rng = np.random.default_rng(plan["bootstrap"]["seed"] + index)
        choices = rng.integers(0, len(experts), size=(plan["bootstrap"]["replicates"], len(experts)))
        raw_p[arm] = _pvalue(ratios)
        stats[arm] = {"relative_improvement_vs_all_h16": 1.0 - _gmean(values[arm]) / _gmean(values["all-h16"]), "wins_vs_all_h16": int((candidate < baseline).sum()), "mean_log_ratio_ci95_bca": bca_mean_interval(ratios, ratios[choices].mean(1)), "exact_pvalue_two_sided": raw_p[arm]}
    first, second = sorted(raw_p, key=raw_p.get)
    adjusted = {first: min(1.0, 2 * raw_p[first]), second: max(min(1.0, 2 * raw_p[first]), raw_p[second])}
    passing = [arm for arm in stats if stats[arm]["relative_improvement_vs_all_h16"] >= 0.03 and stats[arm]["wins_vs_all_h16"] >= 12 and stats[arm]["mean_log_ratio_ci95_bca"][1] < 0 and adjusted[arm] < 0.05]
    winner = max(passing, key=lambda arm: stats[arm]["relative_improvement_vs_all_h16"]) if passing else None
    payload = {"schema": "glm53-projection-h16-attribution-analysis.v1", "plan": {"path": str(args.plan), "sha256": plan_sha}, "inputs": inputs, "experts": experts, "geometric_mean_nmse": {arm: _gmean(v) for arm, v in values.items()}, "comparisons": stats, "holm_adjusted_pvalues": adjusted, "selected_arm": winner, "decision_rule": plan["decision_rule"], "decision": f"pass-fresh-validate-{winner}" if winner else "fail-do-not-open-v4-selection", "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
