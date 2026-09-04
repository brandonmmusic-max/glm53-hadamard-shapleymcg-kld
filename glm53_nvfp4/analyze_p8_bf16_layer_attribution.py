"""Analyze the frozen BF16-matched per-layer P8 KLD attribution."""
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
        raise RuntimeError(f"run does not cover exact role: {path}")
    return np.asarray([payload["windows"][key]["mean_kld"] for key in expected])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--layer3-run", type=Path, required=True)
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
        "3": _values(args.layer3_run, expected),
        "19": _values(args.layer19_run, expected),
        "20": _values(args.layer20_run, expected),
    }
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    results = {}
    for layer, values in runs.items():
        delta = values - baseline
        interval = bca_mean_interval(delta, delta[indices].mean(1), alpha=1.0 / 60.0)
        mean_delta = float(delta.mean())
        passed = mean_delta <= -0.0004 and interval[1] < 0
        results[layer] = {
            "candidate_mean_kld": float(values.mean()),
            "baseline_mean_kld": float(baseline.mean()),
            "mean_delta_kld": mean_delta,
            "relative_improvement": float(1.0 - values.mean() / baseline.mean()),
            "delta_ci98_333_bca": interval,
            "window_wins": int((delta < 0).sum()),
            "decision": "pass-layer-supported" if passed else "fail-layer-not-supported",
        }
    supported = [int(layer) for layer, row in results.items() if row["decision"].startswith("pass")]
    payload = {
        "schema": "glm53-p8-bf16-matched-layer-attribution-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "baseline_run_sha256": sha256_file(args.baseline_run),
        "results": results,
        "supported_layers": supported,
        "decision": "pass-build-supported-subset" if supported else "fail-no-layer-supported",
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
