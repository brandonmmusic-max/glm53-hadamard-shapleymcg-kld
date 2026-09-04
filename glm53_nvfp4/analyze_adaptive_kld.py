"""Paired KLD comparison explicitly labeled as adaptive reuse, never qualification."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval, load_window_means
from .shard_index import sha256_file


def _gmean(values: np.ndarray) -> float:
    return float(math.exp(np.log(values).mean()))


def _load(path: Path) -> dict[str, float]:
    if path.is_dir():
        return load_window_means(path)
    payload = json.loads(path.read_text())
    windows = payload.get("windows")
    if not isinstance(windows, dict):
        raise ValueError(f"run manifest has no window mapping: {path}")
    result = {str(key): float(value["mean_kld"]) for key, value in windows.items()}
    if not result:
        raise ValueError(f"run contains zero windows: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, action="append", required=True)
    parser.add_argument("--control-name", action="append", required=True)
    parser.add_argument("--runtime-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.control_run) != len(args.control_name):
        raise ValueError("control names and runs must match")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate = _load(args.candidate_run)
    proof = json.loads(args.runtime_proof.read_text())
    if proof.get("status") != "pass":
        raise RuntimeError("runtime proof failed")
    comparisons = {}
    for index, (name, path) in enumerate(zip(args.control_name, args.control_run, strict=True)):
        control = _load(path)
        if set(candidate) != set(control):
            raise ValueError(f"window mismatch for {name}")
        keys = sorted(candidate)
        c = np.asarray([candidate[k] for k in keys])
        b = np.asarray([control[k] for k in keys])
        delta = c - b
        rng = np.random.default_rng(2026090421 + index)
        samples = rng.integers(0, len(keys), size=(20000, len(keys)))
        comparisons[name] = {
            "windows": len(keys),
            "candidate_arithmetic_mean_kld": float(c.mean()),
            "control_arithmetic_mean_kld": float(b.mean()),
            "candidate_geometric_mean_kld": _gmean(c),
            "control_geometric_mean_kld": _gmean(b),
            "relative_geometric_mean_improvement": 1 - _gmean(c) / _gmean(b),
            "mean_delta_kld": float(delta.mean()),
            "delta_ci95_bca": bca_mean_interval(delta, delta[samples].mean(1)),
            "wins": int((delta < 0).sum()),
            "control_run": {"path": str(path), "sha256": sha256_file(path)},
        }
    payload = {
        "schema": "glm53-adaptive-reused-role-kld-analysis.v1",
        "status": "adaptive-diagnostic-only-not-qualification",
        "reason": "the V4 windows were opened by the prior down-H16 decision",
        "candidate_run": {"path": str(args.candidate_run), "sha256": sha256_file(args.candidate_run)},
        "runtime_proof": {"path": str(args.runtime_proof), "sha256": sha256_file(args.runtime_proof)},
        "comparisons": comparisons,
        "confirmation_logits_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
