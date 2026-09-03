"""Equal-window paired confirmation analysis locked before holdout access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .shard_index import sha256_file

BOOTSTRAP_SEED = 20260902
BOOTSTRAP_B = 20000


def bca_mean_interval(values: np.ndarray, bootstrap: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import norm
    observed = values.mean()
    z0 = norm.ppf(np.clip(np.mean(bootstrap < observed), 1e-9, 1 - 1e-9))
    jack = np.array([np.delete(values, i).mean() for i in range(len(values))])
    center = jack.mean()
    numerator = np.sum((center - jack) ** 3)
    denominator = 6 * np.sum((center - jack) ** 2) ** 1.5
    acceleration = numerator / denominator if denominator > 0 else 0.0
    quantiles = []
    for target in (alpha / 2, 1 - alpha / 2):
        z = norm.ppf(target)
        adjusted = norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        quantiles.append(float(np.quantile(bootstrap, np.clip(adjusted, 0, 1))))
    return quantiles[0], quantiles[1]


def load_window_means(run_dir: Path) -> dict[str, float]:
    import pyarrow.parquet as pq

    result = {}
    for path in sorted(run_dir.glob("*.parquet")):
        frame = pq.read_table(path, columns=["window_id", "kld"]).to_pandas()
        ids = frame["window_id"].unique()
        if len(ids) != 1 or ids[0] in result:
            raise RuntimeError(f"invalid record identity: {path}")
        result[str(ids[0])] = float(frame["kld"].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--stock-run", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("selection", "confirmation"), default="confirmation")
    args = parser.parse_args()
    roles = json.loads(args.roles.read_text())
    expected = [item["id"] for item in roles["roles"][args.role]]
    candidate = load_window_means(args.candidate_run)
    stock = load_window_means(args.stock_run)
    if set(candidate) != set(expected) or set(stock) != set(expected):
        raise RuntimeError(f"paired runs do not contain the exact {args.role} role")
    candidate_values = np.array([candidate[key] for key in expected])
    stock_values = np.array([stock[key] for key in expected])
    delta = candidate_values - stock_values
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_B, len(delta)))
    boot_delta = delta[indices].mean(axis=1)
    boot_ratio = candidate_values[indices].mean(axis=1) / stock_values[indices].mean(axis=1)
    bca = bca_mean_interval(delta, boot_delta)
    percentile = tuple(float(x) for x in np.quantile(boot_delta, [0.025, 0.975]))
    candidate_mean = float(candidate_values.mean())
    stock_mean = float(stock_values.mean())
    improvement = 1.0 - candidate_mean / stock_mean
    passed = improvement >= 0.10 and bca[1] < 0.0
    decision = ("pass" if passed else "fail") if args.role == "confirmation" else ("continue" if float(delta.mean()) < 0.0 else "stop")
    payload = {
        "schema": f"glm53-nvfp4-v2.paired-{args.role}.v1",
        "role": args.role,
        "estimand": "equal-window mean of candidate KLD minus stock KLD",
        "windows": len(expected),
        "candidate_mean_kld": candidate_mean,
        "stock_mean_kld": stock_mean,
        "mean_delta_kld": float(delta.mean()),
        "relative_improvement": improvement,
        "delta_ci95_percentile": percentile,
        "delta_ci95_bca": bca,
        "ratio_ci95_percentile": tuple(float(x) for x in np.quantile(boot_ratio, [0.025, 0.975])),
        "bootstrap": {"replicates": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "window"},
        "threshold": {"minimum_relative_improvement": 0.10, "maximum_bca_upper_delta": 0.0},
        "decision": decision,
        "roles_sha256": sha256_file(args.roles),
        "candidate_records": [{"window_id": key, "mean_kld": candidate[key]} for key in expected],
        "stock_records": [{"window_id": key, "mean_kld": stock[key]} for key in expected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
