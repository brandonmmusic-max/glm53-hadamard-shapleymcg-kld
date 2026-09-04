"""Analyze the relaxed-development three-state P8 policy KLD run."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval, load_window_means
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected_rows = roles["roles"]["conditional-fit"]
    expected = [row["id"] for row in expected_rows]
    domains = {row["id"]: row["domain"] for row in expected_rows}
    candidate = load_window_means(args.candidate_run)
    control = load_window_means(args.control_run)
    if set(candidate) != set(expected) or set(control) != set(expected):
        raise RuntimeError("candidate and control must cover the exact n=32 role")
    c = np.asarray([candidate[key] for key in expected], dtype=np.float64)
    b = np.asarray([control[key] for key in expected], dtype=np.float64)
    delta = c - b
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indexes = rng.integers(0, len(delta), size=(spec["replicates"], len(delta)))
    bootstrap = delta[indexes].mean(1)
    interval = bca_mean_interval(delta, bootstrap)
    relative_improvement = float(1.0 - c.mean() / b.mean())
    by_domain: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(expected, delta, strict=True):
        by_domain[domains[key]].append(float(value))
    engineering_pass = relative_improvement >= 0.05
    strict_pass = float(delta.mean()) < 0 and interval[1] < 0
    payload = {
        "schema": "glm53-p8-joint-policy-relaxed-kld-analysis.v1",
        "decision": "pass-promote-to-shapley" if engineering_pass else "fail-development-promotion",
        "strict_quality_decision": "pass" if strict_pass else "fail",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "candidate_records_sha256": [sha256_file(path) for path in sorted(args.candidate_run.glob("*.parquet"))],
        "control_records_sha256": [sha256_file(path) for path in sorted(args.control_run.glob("*.parquet"))],
        "windows": len(expected),
        "candidate_mean_kld": float(c.mean()),
        "control_mean_kld": float(b.mean()),
        "mean_delta_kld": float(delta.mean()),
        "relative_improvement": relative_improvement,
        "delta_ci95_bca": [float(interval[0]), float(interval[1])],
        "delta_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "window_wins": int((delta < 0).sum()),
        "domain_mean_delta_kld": {key: float(np.mean(value)) for key, value in sorted(by_domain.items())},
        "engineering_promotion_rule": plan["engineering_promotion_rule"],
        "strict_claim_rule": plan["strict_claim_rule"],
        "confidence_interval_is_nonblocking_for_development": True,
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not engineering_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
