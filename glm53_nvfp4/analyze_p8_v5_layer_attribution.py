"""Analyze the frozen exploratory V5 P8 per-layer attribution."""
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
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--layer3-run", type=Path, required=True)
    parser.add_argument("--layer20-run", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--layer3-manifest", type=Path, required=True)
    parser.add_argument("--layer20-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected = roles["selection_waves"][str(plan["selection_wave"])]
    role_by_id = {item["id"]: item for item in roles["roles"]["selection"]}
    baseline_map = load_window_means(args.baseline_run)
    run_maps = {
        "3": load_window_means(args.layer3_run),
        "20": load_window_means(args.layer20_run),
    }
    if set(baseline_map) != set(expected) or any(set(run) != set(expected) for run in run_maps.values()):
        raise RuntimeError("runs do not cover exact V5 wave")

    baseline = np.asarray([baseline_map[key] for key in expected], dtype=np.float64)
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(expected), size=(spec["replicates"], len(expected)))
    results = {}
    for layer, run in run_maps.items():
        values = np.asarray([run[key] for key in expected], dtype=np.float64)
        delta = values - baseline
        interval = bca_mean_interval(delta, delta[indices].mean(1), alpha=0.025)
        domains: dict[str, list[float]] = defaultdict(list)
        for key, value in zip(expected, delta, strict=True):
            domains[role_by_id[key]["domain"]].append(float(value))
        domain_deltas = {name: float(np.mean(rows)) for name, rows in sorted(domains.items())}
        passed = delta.mean() <= -0.0006 and interval[1] < 0 and all(value < 0 for value in domain_deltas.values())
        results[layer] = {
            "candidate_mean_kld": float(values.mean()),
            "baseline_mean_kld": float(baseline.mean()),
            "mean_delta_kld": float(delta.mean()),
            "relative_improvement": float(1.0 - values.mean() / baseline.mean()),
            "delta_ci97_5_bca": interval,
            "domain_mean_delta_kld": domain_deltas,
            "window_wins": int((delta < 0).sum()),
            "decision": "pass-exploratory-layer-supported" if passed else "fail-exploratory-layer",
        }
    supported = [int(layer) for layer, row in results.items() if row["decision"].startswith("pass")]
    payload = {
        "schema": "glm53-p8-v5-layer-attribution-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "run_manifests": {
            "baseline": {"path": str(args.baseline_manifest), "sha256": sha256_file(args.baseline_manifest)},
            "layer3": {"path": str(args.layer3_manifest), "sha256": sha256_file(args.layer3_manifest)},
            "layer20": {"path": str(args.layer20_manifest), "sha256": sha256_file(args.layer20_manifest)},
        },
        "results": results,
        "supported_layers": supported,
        "decision": "pass-exploratory-attribution" if supported else "fail-no-v5-layer-supported",
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
