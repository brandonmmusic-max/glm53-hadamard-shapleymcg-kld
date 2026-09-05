"""Analyze a three-arm V8 tune or conditional-fit gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval
from .shard_index import sha256_file


def _values(manifest: dict, expected: list[str], runtime_role: str) -> np.ndarray:
    windows = manifest.get("windows", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("role") != runtime_role
        or set(windows) != set(expected)
    ):
        raise RuntimeError("run does not contain the exact sealed role")
    return np.array([float(windows[key]["mean_kld"]) for key in expected])


def analyze(execution: dict, roles: dict, manifests: dict[str, dict]) -> dict:
    runtime_role = execution["runtime_role"]
    rows = roles["roles"][runtime_role]
    expected = [row["id"] for row in rows]
    if len(expected) != 32 or len(set(expected)) != 32:
        raise RuntimeError("V8 gate requires exactly 32 distinct windows")
    if set(manifests) != {"stock", "zero", "candidate"}:
        raise RuntimeError("V8 gate requires stock, zero, and candidate")
    values = {
        arm: _values(manifests[arm], expected, runtime_role) for arm in manifests
    }
    candidate = values["candidate"]
    stock = values["stock"]
    zero = values["zero"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(expected), size=(BOOTSTRAP_B, len(expected)))

    def comparison(reference: np.ndarray) -> dict:
        delta = candidate - reference
        bootstrap = delta[indices].mean(axis=1)
        return {
            "reference_mean_kld": float(reference.mean()),
            "mean_delta": float(delta.mean()),
            "relative_improvement": float(1.0 - candidate.mean() / reference.mean()),
            "delta_bca95": bca_mean_interval(delta, bootstrap),
            "windows_better": int(np.sum(delta < 0)),
        }

    domains = sorted({row["domain"] for row in rows})
    domain_delta = {}
    for domain in domains:
        mask = np.array([row["domain"] == domain for row in rows])
        domain_delta[domain] = float((candidate[mask] - stock[mask]).mean())
    threshold = execution["threshold"]
    checks = {
        "candidate_below_stock": float(candidate.mean()) < float(stock.mean()),
        "candidate_below_zero": float(candidate.mean()) < float(zero.mean()),
        "stock_delta_at_most": (
            float((candidate - stock).mean()) <= float(threshold["maximum_stock_delta"])
        ),
        "maximum_domain_delta": (
            max(domain_delta.values()) <= float(threshold["maximum_domain_delta"])
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-gate-analysis.v1",
        "stage": execution["stage"],
        "role": execution["analysis_role"],
        "windows": len(expected),
        "selected_arm": execution["selected_arm"],
        "candidate_mean_kld": float(candidate.mean()),
        "versus_stock": comparison(stock),
        "versus_zero": comparison(zero),
        "domain_mean_delta_vs_stock": domain_delta,
        "threshold": threshold,
        "checks": checks,
        "decision": "pass" if passed else "fail",
        "bootstrap": {
            "replicates": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "unit": "window",
            "blocking": False,
        },
        "protected_roles_opened": [],
        "claim_boundary": "adaptive development evidence; not qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    execution = json.loads(args.execution.read_text())
    roles = json.loads(args.roles.read_text())
    manifests = {}
    for arm in ("stock", "zero", "candidate"):
        path = args.run_root / f"run-{execution['run_ids'][arm]}.json"
        manifests[arm] = json.loads(path.read_text())
    result = analyze(execution, roles, manifests)
    result["execution_sha256"] = sha256_file(args.execution)
    result["roles_sha256"] = sha256_file(args.roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
