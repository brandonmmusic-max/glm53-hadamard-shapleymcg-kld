"""Apply the frozen external V5 validation rule to the P8 layer-3/20 subset."""
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
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected = roles["selection_waves"][str(plan["selection_wave"])]
    role_by_id = {item["id"]: item for item in roles["roles"]["selection"]}
    candidate = load_window_means(args.candidate_run)
    baseline = load_window_means(args.baseline_run)
    if set(candidate) != set(expected) or set(baseline) != set(expected):
        raise RuntimeError("runs do not cover exact V5 wave")
    c = np.asarray([candidate[key] for key in expected], dtype=np.float64)
    b = np.asarray([baseline[key] for key in expected], dtype=np.float64)
    delta = c - b
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    interval = bca_mean_interval(delta, delta[indices].mean(1))
    domains: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(expected, delta, strict=True):
        domains[role_by_id[key]["domain"]].append(float(value))
    domain_deltas = {name: float(np.mean(values)) for name, values in sorted(domains.items())}
    improvement = float(1.0 - c.mean() / b.mean())
    passed = improvement >= 0.03 and interval[1] < 0 and all(value < 0 for value in domain_deltas.values())
    payload = {
        "schema": "glm53-p8-supported-subset-v5-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "windows": len(expected),
        "candidate_mean_kld": float(c.mean()),
        "baseline_mean_kld": float(b.mean()),
        "mean_delta_kld": float(delta.mean()),
        "relative_improvement": improvement,
        "delta_ci95_bca": interval,
        "domain_mean_delta_kld": domain_deltas,
        "window_wins": int((delta < 0).sum()),
        "decision": "pass-external-v5-provisional" if passed else "fail-external-v5",
        "run_manifests": {
            "candidate": {"path": str(args.candidate_manifest), "sha256": sha256_file(args.candidate_manifest)},
            "baseline": {"path": str(args.baseline_manifest), "sha256": sha256_file(args.baseline_manifest)},
        },
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
