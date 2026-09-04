"""Analyze the frozen all-expert scaled-H128 MCG attribution."""
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
    rows, inputs = [], []
    for path in args.input:
        payload = json.loads(path.read_text())
        if payload["plan"]["sha256"] != plan_sha or payload["protected_roles_opened"]:
            raise RuntimeError(f"invalid attribution input: {path}")
        rows.extend(payload["rows"])
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    by_expert = {int(row["expert"]): row for row in rows}
    expected = plan["experts"]
    if len(by_expert) != len(rows) or set(by_expert) != set(expected):
        raise RuntimeError("inputs do not exactly cover the frozen 288 experts")
    candidate = np.asarray([by_expert[e]["candidate_full_output_nmse"] for e in expected])
    control = np.asarray([by_expert[e]["control_full_output_nmse"] for e in expected])
    log_ratio = np.log(candidate / control)
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indexes = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    interval = bca_mean_interval(log_ratio, log_ratio[indexes].mean(1))
    eligible = [int(e) for e, c, b in zip(expected, candidate, control, strict=True) if c <= 0.95 * b]
    projection = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        cand = np.asarray([by_expert[e]["projection_weight_nmse"][name]["candidate_effective_weight_nmse"] for e in expected])
        base = np.asarray([by_expert[e]["projection_weight_nmse"][name]["control_weight_nmse"] for e in expected])
        projection[name] = {
            "candidate_geometric_mean_nmse": _gmean(cand),
            "control_geometric_mean_nmse": _gmean(base),
            "relative_improvement": 1.0 - _gmean(cand) / _gmean(base),
            "wins": int((cand < base).sum()),
        }
    advance = len(eligible) >= 144 and interval[1] < 0
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-all-experts-analysis.v1",
        "plan_sha256": plan_sha,
        "inputs": inputs,
        "experts": len(expected),
        "projection_weight_nmse": projection,
        "full_output": {
            "candidate_geometric_mean_nmse": _gmean(candidate),
            "control_geometric_mean_nmse": _gmean(control),
            "relative_improvement": 1.0 - _gmean(candidate) / _gmean(control),
            "wins": int((candidate < control).sum()),
            "mean_log_ratio_ci95_bca": [float(interval[0]), float(interval[1])],
        },
        "h128_eligible_experts": eligible,
        "h128_eligible_count": len(eligible),
        "policy_bytes": math.ceil(len(expected) / 8) + 3,
        "decision": "pass-build-identity-p8-fallback" if advance else "fail-uniform-h128-family",
        "protected_roles_opened": [],
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
