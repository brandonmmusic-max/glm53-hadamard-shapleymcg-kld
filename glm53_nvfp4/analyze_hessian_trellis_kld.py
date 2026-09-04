"""Apply the frozen paired-window KLD rule to the P8 pseudoquant gate."""
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
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    candidate = _values(args.candidate_run, expected)
    baseline = _values(args.baseline_run, expected)
    delta = candidate - baseline
    rng = np.random.default_rng(20260902)
    indices = rng.integers(0, len(delta), size=(20000, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    interval = bca_mean_interval(delta, bootstrap)
    mean_delta = float(delta.mean())
    passed = mean_delta <= -0.0014 and interval[1] < 0
    payload = {
        "schema": "glm53-hessian-trellis-p8-kld-analysis.v1",
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
        "window_wins": int((delta < 0).sum()),
        "decision": "pass-pseudoquant-kld" if passed else "fail-pseudoquant-kld",
        "limitations": [
            "The candidate executes as a dense BF16 overlay, not the P8 prologue.",
            "This conditional-fit result is developmental and does not open protected confirmation data.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
