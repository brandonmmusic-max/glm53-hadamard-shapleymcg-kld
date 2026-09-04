"""Analyze the pre-registered two-sided H16 expert screen."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


ARMS = ("identity", "input-h16", "output-h16", "two-sided-h16")


def _gmean(values: list[float]) -> float:
    return float(math.exp(np.log(np.asarray(values, dtype=np.float64)).mean()))


def _sign_flip_pvalue(values: np.ndarray) -> float:
    """Exact two-sided paired randomization p value for at most 20 pairs."""
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return exceed / total


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda name: pvalues[name])
    result = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[name]))
        result[name] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    plan_sha = sha256_file(args.plan)
    inputs = []
    rows = []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != plan_sha:
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
    for expert, arm_rows in by_expert.items():
        if set(arm_rows) != set(ARMS):
            raise ValueError(f"arm coverage mismatch for expert {expert}")
    experts = sorted(by_expert)
    values = {
        arm: [by_expert[expert][arm]["evaluation_full_expert_nmse"] for expert in experts]
        for arm in ARMS
    }
    baseline = np.asarray(values["identity"], dtype=np.float64)
    comparisons = {}
    raw_pvalues = {}
    bootstrap = plan["bootstrap"]
    for index, arm in enumerate(ARMS[1:]):
        candidate = np.asarray(values[arm], dtype=np.float64)
        log_ratio = np.log(candidate / baseline)
        rng = np.random.default_rng(bootstrap["seed"] + index)
        indices = rng.integers(
            0, len(experts), size=(bootstrap["replicates"], len(experts))
        )
        interval = bca_mean_interval(log_ratio, log_ratio[indices].mean(axis=1))
        raw_pvalues[arm] = _sign_flip_pvalue(log_ratio)
        comparisons[f"{arm}-vs-identity"] = {
            "relative_improvement": 1.0 - _gmean(values[arm]) / _gmean(values["identity"]),
            "wins": sum(a < b for a, b in zip(values[arm], values["identity"], strict=True)),
            "mean_log_ratio_ci95_bca": interval,
            "exact_pvalue_two_sided": raw_pvalues[arm],
        }
    primary_log_ratio = np.log(
        np.asarray(values["two-sided-h16"], dtype=np.float64)
        / np.asarray(values["input-h16"], dtype=np.float64)
    )
    rng = np.random.default_rng(bootstrap["seed"] + 100)
    indices = rng.integers(
        0, len(experts), size=(bootstrap["replicates"], len(experts))
    )
    primary_interval = bca_mean_interval(
        primary_log_ratio, primary_log_ratio[indices].mean(axis=1)
    )
    primary_improvement = (
        1.0
        - _gmean(values["two-sided-h16"]) / _gmean(values["input-h16"])
    )
    primary_wins = sum(
        a < b
        for a, b in zip(
            values["two-sided-h16"], values["input-h16"], strict=True
        )
    )
    passed = (
        primary_improvement >= 0.05
        and primary_wins >= 12
        and primary_interval[1] < 0
    )
    payload = {
        "schema": "glm53-two-sided-h16-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": plan_sha},
        "inputs": inputs,
        "experts": experts,
        "geometric_mean_nmse": {arm: _gmean(value) for arm, value in values.items()},
        "candidate_vs_identity": comparisons,
        "holm_adjusted_pvalues": _holm(raw_pvalues),
        "primary_two_sided_vs_input_h16": {
            "relative_improvement": primary_improvement,
            "wins": primary_wins,
            "mean_log_ratio_ci95_bca": primary_interval,
            "exact_pvalue_two_sided": _sign_flip_pvalue(primary_log_ratio),
        },
        "decision_rule": plan["decision_rule"],
        "decision": (
            "pass-build-layer3-pseudoquant-candidate"
            if passed
            else "fail-do-not-open-v4-selection"
        ),
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
