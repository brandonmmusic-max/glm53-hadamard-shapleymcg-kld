"""Analyze frozen layer-22 P8 per-projection KLD attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _values(path: Path, expected: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload.get("role") != "conditional-fit" or set(payload.get("windows", {})) != set(expected):
        raise RuntimeError(f"run does not cover exact conditional-fit role: {path}")
    return np.asarray([payload["windows"][key]["mean_kld"] for key in expected])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    baseline = _values(args.baseline_run, expected)
    if len(args.candidate_run) != len(PROJECTIONS):
        raise RuntimeError("requires exactly three candidate runs in frozen order")
    runs = {
        projection: _values(path, expected)
        for projection, path in zip(PROJECTIONS, args.candidate_run, strict=True)
    }
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    results = {}
    for projection, values in runs.items():
        delta = values - baseline
        interval = bca_mean_interval(delta, delta[indices].mean(1), alpha=1.0 / 60.0)
        mean_delta = float(delta.mean())
        passed = mean_delta <= -0.0004 and interval[1] < 0
        results[projection] = {
            "candidate_mean_kld": float(values.mean()),
            "baseline_mean_kld": float(baseline.mean()),
            "mean_delta_kld": mean_delta,
            "relative_improvement": float(1.0 - values.mean() / baseline.mean()),
            "delta_ci98_333_bca": interval,
            "window_wins": int((delta < 0).sum()),
            "domain_mean_delta_kld": {
                domain: float(delta[[item["domain"] == domain for item in roles["roles"]["conditional-fit"]]].mean())
                for domain in sorted({item["domain"] for item in roles["roles"]["conditional-fit"]})
            },
            "decision": "pass-directionally-supported" if passed else "fail-not-supported",
        }
    supported = [key for key, row in results.items() if row["decision"].startswith("pass")]
    payload = {
        "schema": "glm53-p8-layer22-projection-attribution-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "baseline_run_sha256": sha256_file(args.baseline_run),
        "candidate_run_sha256": {
            projection: sha256_file(path)
            for projection, path in zip(PROJECTIONS, args.candidate_run, strict=True)
        },
        "results": results,
        "supported_projections": supported,
        "decision": "pass-redesign-supported-projections" if supported else "fail-no-projection-supported",
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
