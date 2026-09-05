"""Analyze the sealed V8 train32 scalar-butterfly KLD grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval
from .shard_index import sha256_file


def _values(manifest: dict, expected: list[str]) -> np.ndarray:
    windows = manifest.get("windows", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("role") != "fit"
        or set(windows) != set(expected)
    ):
        raise RuntimeError("run manifest does not contain the exact fit role")
    return np.array([float(windows[key]["mean_kld"]) for key in expected])


def analyze(execution: dict, roles: dict, manifests: dict[str, dict]) -> dict:
    expected_rows = roles["roles"]["fit"]
    expected = [row["id"] for row in expected_rows]
    if len(expected) != 32 or len(set(expected)) != 32:
        raise RuntimeError("V8 train analysis requires exactly 32 distinct fit windows")
    if set(manifests) != set(execution["run_order"]):
        raise RuntimeError("analysis arms differ from sealed execution order")
    values = {arm: _values(manifests[arm], expected) for arm in manifests}
    stock = values["stock"]
    zero = values["zero"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(expected), size=(BOOTSTRAP_B, len(expected)))
    arms = {}
    for arm in execution["grid_arms"]:
        candidate = values[arm]
        delta_stock = candidate - stock
        delta_zero = candidate - zero
        boot_stock = delta_stock[indices].mean(axis=1)
        boot_zero = delta_zero[indices].mean(axis=1)
        arms[arm] = {
            "mean_kld": float(candidate.mean()),
            "mean_delta_vs_stock": float(delta_stock.mean()),
            "relative_improvement_vs_stock": float(1.0 - candidate.mean() / stock.mean()),
            "delta_vs_stock_bca95": bca_mean_interval(delta_stock, boot_stock),
            "mean_delta_vs_zero": float(delta_zero.mean()),
            "relative_improvement_vs_zero": float(1.0 - candidate.mean() / zero.mean()),
            "delta_vs_zero_bca95": bca_mean_interval(delta_zero, boot_zero),
            "windows_better_than_stock": int(np.sum(delta_stock < 0)),
            "windows_better_than_zero": int(np.sum(delta_zero < 0)),
        }
    selected = min(execution["grid_arms"], key=lambda arm: arms[arm]["mean_kld"])
    domains = sorted({row["domain"] for row in expected_rows})
    domain_rows = {}
    for domain in domains:
        mask = np.array([row["domain"] == domain for row in expected_rows])
        domain_rows[domain] = {
            arm: float(values[arm][mask].mean()) for arm in execution["run_order"]
        }
    return {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-train32-analysis.v1",
        "role": "fit/train32 adaptive search",
        "windows": len(expected),
        "selection_estimand": "equal-window mean all-causal teacher KLD",
        "stock_mean_kld": float(stock.mean()),
        "zero_mean_kld": float(zero.mean()),
        "arms": arms,
        "selected_arm": selected,
        "adaptive_decision": (
            "select-nonzero-for-tune" if selected != "zero" else "stop-zero-won"
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "unit": "window",
            "note": "descriptive control after adaptive nine-arm selection",
        },
        "domain_mean_kld": domain_rows,
        "protected_roles_opened": [],
        "claim_boundary": "adaptive fit-role selection only; not qualification evidence",
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
    for arm in execution["run_order"]:
        path = args.run_root / f"run-{execution['run_ids'][arm]}.json"
        manifest = json.loads(path.read_text())
        if manifest.get("run_id") != execution["run_ids"][arm]:
            raise RuntimeError(f"run id mismatch: {path}")
        manifests[arm] = manifest
    result = analyze(execution, roles, manifests)
    result["execution_sha256"] = sha256_file(args.execution)
    result["roles_sha256"] = sha256_file(args.roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
