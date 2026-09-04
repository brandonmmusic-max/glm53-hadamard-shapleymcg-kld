"""Analyze the preregistered fixed-H128 P8 activation screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _gmean(values: np.ndarray) -> float:
    return float(np.exp(np.log(values).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    raw = json.loads(args.raw.read_text())
    by_expert = {int(cell["expert"]): cell for cell in raw["cells"]}
    if set(by_expert) != set(plan["experts"]):
        raise RuntimeError("raw experts do not match frozen plan")
    ordinary = np.asarray([by_expert[e]["ordinary_e4m3"]["nmse"] for e in plan["experts"]], dtype=np.float64)
    rotated = np.asarray([by_expert[e]["fixed_h128_e4m3"]["nmse"] for e in plan["experts"]], dtype=np.float64)
    closure = np.asarray([by_expert[e]["fixed_h128_unquantized_closure"]["nmse"] for e in plan["experts"]], dtype=np.float64)
    log_ratio = np.log(rotated / ordinary)
    boot = plan["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    indices = rng.integers(0, len(log_ratio), size=(boot["replicates"], len(log_ratio)))
    ci = bca_mean_interval(log_ratio, log_ratio[indices].mean(axis=1))
    improvement = 1.0 - _gmean(rotated) / _gmean(ordinary)
    wins = int((rotated < ordinary).sum())
    passed = improvement >= 0.10 and wins >= 12 and ci[1] < 0 and float(closure.max()) <= 1e-10
    payload = {
        "schema": "glm53-p8-h128-activation-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "raw": {"path": str(args.raw), "sha256": sha256_file(args.raw)},
        "geometric_mean_nmse": {"ordinary_e4m3": _gmean(ordinary), "fixed_h128_e4m3": _gmean(rotated)},
        "fixed_h128_relative_improvement": improvement,
        "fixed_h128_wins": wins,
        "mean_log_ratio_ci95_bca": [float(ci[0]), float(ci[1])],
        "maximum_unquantized_closure_nmse": float(closure.max()),
        "decision_rule": plan["decision_rule"],
        "decision": "pass-advance-to-mcg-codec-screen" if passed else "fail-do-not-advance-fixed-h128",
        "interpretation_boundary": plan["interpretation_boundary"],
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
