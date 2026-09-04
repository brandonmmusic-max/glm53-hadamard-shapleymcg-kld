"""Analyze scaled-H128 procedural-MCG K4 expert screen."""
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
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    plan_sha = sha256_file(args.plan)
    rows = []
    inputs = []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != plan_sha or raw["protected_roles_opened"]:
            raise RuntimeError(f"invalid input {path}")
        rows.extend(raw["rows"])
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    by_expert = {int(row["expert"]): row for row in rows}
    if len(by_expert) != len(rows) or set(by_expert) != set(plan["experts"]):
        raise RuntimeError("combined inputs do not exactly cover frozen experts")
    candidate = np.asarray([by_expert[e]["candidate_full_output_nmse"] for e in plan["experts"]])
    control = np.asarray([by_expert[e]["control_full_output_nmse"] for e in plan["experts"]])
    ratios = np.log(candidate / control)
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    choices = rng.integers(0, len(ratios), size=(spec["replicates"], len(ratios)))
    interval = bca_mean_interval(ratios, ratios[choices].mean(1))
    improvement = 1.0 - _gmean(candidate) / _gmean(control)
    wins = int((candidate < control).sum())
    projection = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        cand = np.asarray([by_expert[e]["projection_weight_nmse"][name]["candidate_effective_weight_nmse"] for e in plan["experts"]])
        ctl = np.asarray([by_expert[e]["projection_weight_nmse"][name]["control_weight_nmse"] for e in plan["experts"]])
        projection[name] = {
            "candidate_geometric_mean_nmse": _gmean(cand),
            "control_geometric_mean_nmse": _gmean(ctl),
            "candidate_relative_improvement": 1.0 - _gmean(cand) / _gmean(ctl),
        }
    bit_exact = all(bool(by_expert[e]["candidate_codec_decode_bit_exact"]) for e in plan["experts"])
    rate_ok = all(by_expert[e]["candidate_physical_bpw"] <= by_expert[e]["control_physical_bpw"] for e in plan["experts"])
    passed = rate_ok and improvement >= 0.10 and wins >= 12 and interval[1] < 0 and bit_exact and set(projection) == {"gate_proj", "up_proj", "down_proj"}
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-screen-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": plan_sha},
        "inputs": inputs,
        "stored_bpw": {"candidate": plan["candidate"]["physical_bpw"], "control": plan["control"]["physical_bpw"]},
        "projection_weight_nmse_first": projection,
        "full_output": {
            "candidate_geometric_mean_nmse": _gmean(candidate),
            "control_geometric_mean_nmse": _gmean(control),
            "relative_improvement": improvement,
            "wins": wins,
            "mean_log_ratio_ci95_bca": interval,
        },
        "codec_decode_bit_exact": bit_exact,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-build-layer3-pseudoquant-kld-overlay" if passed else "fail-do-not-build-layer3-overlay",
        "interpretation_boundary": plan["interpretation_boundary"],
        "isa_cost": plan["isa_cost"],
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
