"""Analyze the frozen P8 layer-19/layer-20 KLD attribution diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _values(path: Path, expected: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload["role"] != "conditional-fit" or set(payload["windows"]) != set(expected):
        raise RuntimeError(f"run does not contain exact conditional-fit role: {path}")
    return np.asarray([payload["windows"][key]["mean_kld"] for key in expected])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--layer19-run", type=Path, required=True)
    parser.add_argument("--layer20-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    baseline = _values(args.baseline_run, expected)
    runs = {
        "layer19": _values(args.layer19_run, expected),
        "layer20": _values(args.layer20_run, expected),
    }
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    results = {}
    for name, values in runs.items():
        delta = values - baseline
        interval = bca_mean_interval(delta, delta[indices].mean(1), alpha=0.025)
        mean_delta = float(delta.mean())
        passed = mean_delta <= -0.0014 and interval[1] < 0
        results[name] = {
            "candidate_mean_kld": float(values.mean()),
            "baseline_mean_kld": float(baseline.mean()),
            "mean_delta_kld": mean_delta,
            "relative_improvement": float(1.0 - values.mean() / baseline.mean()),
            "delta_ci97_5_bca": interval,
            "window_wins": int((delta < 0).sum()),
            "decision": "pass-advance-layer" if passed else "fail-do-not-advance-layer",
        }
    payload = {
        "schema": "glm53-p8-layer19-20-ablation-kld-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "baseline_run_sha256": sha256_file(args.baseline_run),
        "candidate_run_sha256": {
            "layer19": sha256_file(args.layer19_run),
            "layer20": sha256_file(args.layer20_run),
        },
        "results": results,
        "decision": (
            "pass-one-or-more-layer"
            if any(row["decision"].startswith("pass") for row in results.values())
            else "fail-no-layer-advances"
        ),
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
