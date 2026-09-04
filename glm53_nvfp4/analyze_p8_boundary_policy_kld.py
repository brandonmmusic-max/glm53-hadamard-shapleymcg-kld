"""Analyze the pre-registered adaptive identity/H128 P8 linkage run."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _values(path: Path, expected: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload["role"] != "conditional-fit" or set(payload["windows"]) != set(expected):
        raise RuntimeError(f"run does not contain the exact conditional-fit role: {path}")
    return np.asarray(
        [payload["windows"][key]["mean_kld"] for key in expected], dtype=np.float64
    )


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
    if plan.get("status") != "adaptive-reuse-not-qualifying":
        raise RuntimeError("plan does not declare adaptive reuse")
    roles = json.loads(args.roles.read_text())
    expected_rows = roles["roles"]["conditional-fit"]
    expected = [item["id"] for item in expected_rows]
    domain_by_id = {item["id"]: item["domain"] for item in expected_rows}
    candidate = _values(args.candidate_run, expected)
    control = _values(args.control_run, expected)
    delta = candidate - control

    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(delta), size=(spec["replicates"], len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    interval = bca_mean_interval(delta, bootstrap)
    domains: dict[str, list[float]] = defaultdict(list)
    rows = []
    for key, cand, base, value in zip(expected, candidate, control, delta, strict=True):
        domains[domain_by_id[key]].append(float(value))
        rows.append(
            {
                "window": key,
                "domain": domain_by_id[key],
                "candidate_kld": float(cand),
                "control_kld": float(base),
                "delta_kld": float(value),
            }
        )

    mean_delta = float(delta.mean())
    threshold = -0.0014
    passed = mean_delta <= threshold and interval[1] < 0
    payload = {
        "schema": "glm53-p8-identity-h128-policy-adaptive-kld-analysis.v1",
        "evidence_status": "adaptive-reuse-not-qualifying",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "candidate_run_sha256": sha256_file(args.candidate_run),
        "control_run_sha256": sha256_file(args.control_run),
        "windows": len(expected),
        "candidate_mean_kld": float(candidate.mean()),
        "control_mean_kld": float(control.mean()),
        "mean_delta_kld": mean_delta,
        "relative_improvement": float(1.0 - candidate.mean() / control.mean()),
        "delta_ci95_bca": [float(interval[0]), float(interval[1])],
        "delta_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "threshold_delta_kld": threshold,
        "domain_mean_delta_kld": {
            name: float(np.mean(values)) for name, values in sorted(domains.items())
        },
        "window_wins": int((delta < 0).sum()),
        "largest_regression": max(rows, key=lambda row: row["delta_kld"]),
        "largest_improvement": min(rows, key=lambda row: row["delta_kld"]),
        "decision_rule": plan["decision_rule"],
        "decision": "adaptive-signal-pass" if passed else "adaptive-signal-fail",
        "window_results": rows,
        "confirmation_logits_opened": False,
        "ldlq_used": False,
        "limitations": [
            "These same conditional-fit windows informed the redesign, so this result cannot qualify or validate the policy.",
            "Both arms use decoded BF16 weights and matched E4M3 activation carriers, not a native P8 prologue.",
            "The protected 28-window confirmation role remains unopened.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
