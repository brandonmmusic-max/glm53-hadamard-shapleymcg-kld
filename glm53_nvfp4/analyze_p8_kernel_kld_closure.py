"""Analyze physical P8 kernel versus decoded-pseudoquant teacher KLD."""
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
    parser.add_argument("--kernel-run", type=Path, required=True)
    parser.add_argument("--pseudoquant-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected_rows = roles["roles"]["conditional-fit"]
    expected = [row["id"] for row in expected_rows]
    domain = {row["id"]: row["domain"] for row in expected_rows}
    kernel = load_window_means(args.kernel_run)
    pseudo = load_window_means(args.pseudoquant_run)
    if set(kernel) != set(expected) or set(pseudo) != set(expected):
        raise RuntimeError("kernel and pseudoquant runs must contain the exact n=32 role")
    k = np.asarray([kernel[key] for key in expected], dtype=np.float64)
    p = np.asarray([pseudo[key] for key in expected], dtype=np.float64)
    delta = k - p
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indexes = rng.integers(0, len(delta), size=(spec["replicates"], len(delta)))
    bootstrap = delta[indexes].mean(1)
    interval = bca_mean_interval(delta, bootstrap)
    by_domain: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(expected, delta, strict=True):
        by_domain[domain[key]].append(float(value))
    mean_delta = float(delta.mean())
    require_zero_in_ci = plan.get("schema") == (
        "glm53-p8-native-deterministic-kld-closure-plan.v2"
    )
    zero_in_ci = bool(interval[0] <= 0.0 <= interval[1])
    passed = abs(mean_delta) <= 0.0014 and (
        zero_in_ci if require_zero_in_ci else True
    )
    decision_rule = plan.get(
        "decision_rule", plan.get("engineering_decision_rule", "")
    )
    payload = {
        "schema": "glm53-p8-kernel-pseudoquant-kld-closure-analysis.v1",
        "decision": "pass-engineering-kld-closure" if passed else "fail",
        "evidence_status": "opened-role developmental engineering closure",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "kernel_run_sha256": [sha256_file(path) for path in sorted(args.kernel_run.glob("*.parquet"))],
        "pseudoquant_run_sha256": [sha256_file(path) for path in sorted(args.pseudoquant_run.glob("*.parquet"))],
        "windows": len(expected),
        "kernel_mean_kld": float(k.mean()),
        "pseudoquant_mean_kld": float(p.mean()),
        "mean_delta_kld": mean_delta,
        "relative_change": float(k.mean() / p.mean() - 1.0),
        "absolute_mean_delta_limit": 0.0014,
        "delta_ci95_bca": [float(interval[0]), float(interval[1])],
        "zero_in_delta_ci95_bca": zero_in_ci,
        "delta_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "window_wins_kernel": int((delta < 0).sum()),
        "domain_mean_delta_kld": {key: float(np.mean(value)) for key, value in sorted(by_domain.items())},
        "decision_rule": decision_rule,
        "strict_quality_claim_gate_relaxed": False,
        "engineering_window_gate_relaxed": True,
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
