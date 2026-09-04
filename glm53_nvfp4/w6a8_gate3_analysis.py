"""Frozen paired analysis for the conditional-fit W6A8 carrier diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval, load_window_means
from .shard_index import sha256_file

ARM_IDS = ("R0", "W0", "W1", "W2", "W3")
CONTRASTS = {
    "scatter": ("W1", "W0"),
    "fast_math": ("W2", "W1"),
    "dynamic_fc2_scale": ("W3", "W2"),
    "total_best_fixed_path": ("W3", "R0"),
}
NONINFERIOR_MARGIN = 0.0014
LADDER_BLOCK_DELTA = 0.01


def _parse_arm(value: str) -> tuple[str, Path]:
    arm, separator, raw_path = value.partition("=")
    if not separator or arm not in ARM_IDS or not raw_path:
        raise argparse.ArgumentTypeError("--arm must be one of R0,W0,W1,W2,W3 followed by =PATH")
    return arm, Path(raw_path)


def _contrast(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, object]:
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
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    if len(expected) != 32 or len(set(expected)) != 32:
        raise RuntimeError("the frozen gate-3 analysis requires 32 unique conditional-fit windows")
    if plan["arms_in_fixed_order"] != [
        {
            "id": "R0",
            "model": "exact decoded MXFP6 weights as BF16 overlay",
            "activation": "BF16 framework path",
            "epilogue": "framework routed reduction",
        },
        {
            "id": "W0",
            "model": "same MXFP6 weights",
            "activation": "E4M3 K32, unit input scales, fast math",
            "epilogue": "BF16 atomic routed accumulation",
        },
        {
            "id": "W1",
            "model": "same MXFP6 weights",
            "activation": "E4M3 K32, unit input scales, fast math",
            "epilogue": "deterministic route outputs plus top-k reduction",
        },
        {
            "id": "W2",
            "model": "same MXFP6 weights",
            "activation": "E4M3 K32, unit input scales, non-fast SiLU",
            "epilogue": "deterministic route outputs plus top-k reduction",
        },
        {
            "id": "W3",
            "model": "same MXFP6 weights",
            "activation": "E4M3 K32, unit FC1 scale, dynamic FC2 scale, non-fast SiLU",
            "epilogue": "deterministic route outputs plus top-k reduction",
        },
    ]:
        raise RuntimeError("analysis plan arm definitions do not match the frozen analyzer")

    arm_paths = dict(args.arm)
    if set(arm_paths) != set(ARM_IDS) or len(args.arm) != len(ARM_IDS):
        raise RuntimeError("provide each fixed arm exactly once")
    records = {arm: load_window_means(arm_paths[arm]) for arm in ARM_IDS}
    for arm, values in records.items():
        if set(values) != set(expected):
            raise RuntimeError(f"arm {arm} does not contain the exact conditional-fit32 role")
    arrays = {
        arm: np.asarray([records[arm][window] for window in expected], dtype=np.float64)
        for arm in ARM_IDS
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = {
        name: {
            "left": left,
            "right": right,
            **_contrast(arrays[left], arrays[right], rng=rng),
        }
        for name, (left, right) in CONTRASTS.items()
    }
    primary = contrasts["total_best_fixed_path"]
    mean_delta = float(primary["mean_delta_kld"])
    upper = float(primary["delta_ci95_bca"][1])
    carrier_repaired = upper <= NONINFERIOR_MARGIN
    ladder_blocked = mean_delta >= LADDER_BLOCK_DELTA
    if carrier_repaired:
        decision = "pass-carrier-repaired"
    elif ladder_blocked:
        decision = "fail-ladder-blocked"
    else:
        decision = "continue-w4-w5-diagnosis"

    payload = {
        "schema": "glm53-w6a8-gate3-result.v1",
        "role": "conditional-fit",
        "windows": len(expected),
        "estimand": "equal-window mean KLD(W3) minus KLD(R0)",
        "arm_mean_kld": {arm: float(arrays[arm].mean()) for arm in ARM_IDS},
        "contrasts": contrasts,
        "bootstrap": {
            "method": "paired BCa",
            "replicates": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "unit": "window",
        },
        "thresholds": {
            "carrier_repaired_bca_upper_max": NONINFERIOR_MARGIN,
            "ladder_blocked_mean_delta_min": LADDER_BLOCK_DELTA,
        },
        "carrier_repaired": carrier_repaired,
        "ladder_blocked": ladder_blocked,
        "decision": decision,
        "analysis_plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "arm_inputs": {
            arm: {"path": str(arm_paths[arm]), "tree_files": len(list(arm_paths[arm].glob("*.parquet")))}
            for arm in ARM_IDS
        },
        "per_window": [
            {
                "window_id": window,
                **{arm: float(arrays[arm][index]) for arm in ARM_IDS},
            }
            for index, window in enumerate(expected)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
