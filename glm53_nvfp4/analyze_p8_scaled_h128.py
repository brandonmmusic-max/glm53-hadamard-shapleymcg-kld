"""Analyze disjoint validation for learned scaled-H128 P8 activation bases."""
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
    scaled = np.asarray([cells[e]["scaled_h128_e4m3"]["nmse"] for e in plan["experts"]])
    closure = np.asarray([cells[e]["scaled_h128_unquantized_closure"]["nmse"] for e in plan["experts"]])
    ratio = np.log(scaled / ordinary)
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(ratio), size=(spec["replicates"], len(ratio)))
    interval = bca_mean_interval(ratio, ratio[indices].mean(1))
    improvement = 1.0 - _gmean(scaled) / _gmean(ordinary)
    wins = int((scaled < ordinary).sum())
    passed = improvement >= 0.10 and wins >= 12 and interval[1] < 0 and float(closure.max()) <= 1e-10
    payload = {
        "schema": "glm53-p8-scaled-h128-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "raw": {"path": str(args.raw), "sha256": sha256_file(args.raw)},
        "rotations": raw["rotations"],
        "geometric_mean_nmse": {"ordinary_e4m3": _gmean(ordinary), "scaled_h128_e4m3": _gmean(scaled)},
        "scaled_h128_relative_improvement": improvement,
        "scaled_h128_wins": wins,
        "mean_log_ratio_ci95_bca": interval,
        "maximum_unquantized_closure_nmse": float(closure.max()),
        "selected_laws": {str(e): {"law": cells[e]["selected_law"], "alpha": cells[e]["selected_alpha"]} for e in plan["experts"]},
        "decision_rule": plan["decision_rule"],
        "decision": "pass-advance-to-mcg-codec-screen" if passed else "fail-do-not-advance-scaled-h128",
        "interpretation_boundary": plan["interpretation_boundary"],
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
