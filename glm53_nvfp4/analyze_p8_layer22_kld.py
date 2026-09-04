"""Apply the frozen matched-path layer-22 P8 KLD decision rule."""
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
    return np.asarray([payload["windows"][key]["mean_kld"] for key in expected], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected_rows = roles["roles"]["conditional-fit"]
    expected = [item["id"] for item in expected_rows]
    domain_by_id = {item["id"]: item["domain"] for item in expected_rows}
    candidate = _values(args.candidate_run, expected)
    baseline = _values(args.baseline_run, expected)
    delta = candidate - baseline

    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(delta), size=(spec["replicates"], len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    interval = bca_mean_interval(delta, bootstrap)
    domains: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(expected, delta, strict=True):
        domains[domain_by_id[key]].append(float(value))
    domain_deltas = {name: float(np.mean(values)) for name, values in sorted(domains.items())}
    mean_delta = float(delta.mean())
    passed = mean_delta <= -0.0014 and interval[1] < 0
    payload = {
        "schema": "glm53-p8-layer22-matched-kld-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "candidate_run_sha256": sha256_file(args.candidate_run),
        "baseline_run_sha256": sha256_file(args.baseline_run),
        "windows": len(expected),
        "candidate_mean_kld": float(candidate.mean()),
        "baseline_mean_kld": float(baseline.mean()),
        "mean_delta_kld": mean_delta,
        "relative_improvement": float(1.0 - candidate.mean() / baseline.mean()),
        "delta_ci95_bca": interval,
        "delta_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "domain_mean_delta_kld": domain_deltas,
        "window_wins": int((delta < 0).sum()),
        "decision": "pass-pseudoquant-kld" if passed else "fail-pseudoquant-kld",
        "confirmation_logits_opened": False,
        "ldlq_used": False,
        "limitations": [
            "The candidate and control execute as matched BF16 overlays, not the native P8 prologue.",
            "This conditional-fit result is developmental and does not open protected confirmation data.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
