"""Analyze powered replication of fixed scaled-H128 family laws."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _gmean(values: np.ndarray) -> float:
    return float(math.exp(np.log(values).mean()))


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
    cells = {int(cell["expert"]): cell for cell in raw["cells"]}
    if set(cells) != set(plan["experts"]) or raw["protected_roles_opened"]:
        raise RuntimeError("raw result violates frozen coverage")
    ordinary = np.asarray([cells[e]["ordinary_e4m3"]["nmse"] for e in plan["experts"]])
    results = {}
    for index, arm in enumerate(plan["arms"]):
        candidate = np.asarray([cells[e]["arms"][arm]["metrics"]["nmse"] for e in plan["experts"]])
        closure = np.asarray([cells[e]["arms"][arm]["closure"]["nmse"] for e in plan["experts"]])
        ratios = np.log(candidate / ordinary)
        spec = plan["bootstrap"]
        rng = np.random.default_rng(spec["seed"] + index)
        choices = rng.integers(0, len(ratios), size=(spec["replicates"], len(ratios)))
        interval = bca_mean_interval(ratios, ratios[choices].mean(1), alpha=spec["alpha"])
        improvement = 1.0 - _gmean(candidate) / _gmean(ordinary)
        wins = int((candidate < ordinary).sum())
        passed = improvement >= 0.10 and wins >= 12 and interval[1] < 0 and float(closure.max()) <= 1e-10
        results[arm] = {
            "geometric_mean_nmse": _gmean(candidate),
            "relative_improvement": improvement,
            "wins": wins,
            "mean_log_ratio_ci97_5_bca": interval,
            "maximum_unquantized_closure_nmse": float(closure.max()),
            "passed": passed,
        }
    passing = [arm for arm, result in results.items() if result["passed"]]
    selected = max(passing, key=lambda arm: results[arm]["relative_improvement"]) if passing else None
    payload = {
        "schema": "glm53-p8-scaled-h128-replication-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "raw": {"path": str(args.raw), "sha256": sha256_file(args.raw)},
        "rotations": raw["rotations"],
        "ordinary_geometric_mean_nmse": _gmean(ordinary),
        "arms": results,
        "selected_arm": selected,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-advance-to-mcg-codec-screen" if selected else "fail-do-not-advance-scaled-h128",
        "interpretation_boundary": plan["interpretation_boundary"],
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
