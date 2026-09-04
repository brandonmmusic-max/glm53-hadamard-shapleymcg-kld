"""Frozen paired analysis for the GLM clipped-SwiGLU W6A8 repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval, load_window_means
from .shard_index import sha256_file

MARGIN = 0.0014
BLOCK_DELTA = 0.01


def _paired(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    delta = left - right
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_B, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "mean_delta_kld": float(delta.mean()),
        "delta_ci95_bca": bca_mean_interval(delta, bootstrap),
        "delta_ci95_percentile": tuple(float(x) for x in np.quantile(bootstrap, [0.025, 0.975])),
        "per_window_delta": [float(x) for x in delta],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--w3", type=Path, required=True)
    parser.add_argument("--clamp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["swiglu_limit"] != 10.0 or plan["primary_estimand"] != "KLD(C0) - KLD(R0)":
        raise RuntimeError("analysis plan does not match the frozen clamp analyzer")
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    if len(expected) != 32 or len(set(expected)) != 32:
        raise RuntimeError("clamp gate requires the exact conditional-fit32 role")
    mappings = {
        "R0": load_window_means(args.r0),
        "W3": load_window_means(args.w3),
        "C0": load_window_means(args.clamp),
    }
    for arm, values in mappings.items():
        if set(values) != set(expected):
            raise RuntimeError(f"arm {arm} does not contain the exact conditional-fit32 role")
    arrays = {
        arm: np.asarray([values[item] for item in expected], dtype=np.float64)
        for arm, values in mappings.items()
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    versus_r0 = _paired(arrays["C0"], arrays["R0"], rng)
    versus_w3 = _paired(arrays["C0"], arrays["W3"], rng)
    repaired = float(versus_r0["delta_ci95_bca"][1]) <= MARGIN
    blocked = float(versus_r0["mean_delta_kld"]) >= BLOCK_DELTA
    decision = (
        "pass-carrier-repaired"
        if repaired
        else "fail-ladder-blocked"
        if blocked
        else "continue-scale-and-staged-epilogue"
    )
    payload = {
        "schema": "glm53-w6a8-clamp-result.v1",
        "role": "conditional-fit",
        "windows": 32,
        "arm_mean_kld": {arm: float(values.mean()) for arm, values in arrays.items()},
        "clamp_vs_r0": versus_r0,
        "clamp_vs_w3": versus_w3,
        "thresholds": {"noninferiority_bca_upper_max": MARGIN, "ladder_block_mean_min": BLOCK_DELTA},
        "carrier_repaired": repaired,
        "ladder_blocked": blocked,
        "decision": decision,
        "bootstrap": {"method": "paired BCa", "replicates": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "window"},
        "analysis_plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "per_window": [
            {"window_id": item, **{arm: float(arrays[arm][index]) for arm in arrays}}
            for index, item in enumerate(expected)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
