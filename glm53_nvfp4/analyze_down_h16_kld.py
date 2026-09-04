"""Apply the frozen V4 down-only H16 end-to-end KLD decision."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval, load_window_means
from .shard_index import sha256_file


def _gmean(values: np.ndarray) -> float:
    return float(math.exp(np.log(values).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--stock-run", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--stock-manifest", type=Path, required=True)
    parser.add_argument("--runtime-proof", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    proof = json.loads(args.runtime_proof.read_text())
    if (
        proof.get("status") != "pass"
        or proof.get("layers") != [3]
        or proof.get("ranks") != [0, 1, 2, 3]
        or proof.get("forward_pairs") != 0
        or proof.get("mid_forward_pairs") != 4
    ):
        raise RuntimeError("candidate mid-only runtime proof did not pass exactly")
    expected = roles["selection_waves"][str(plan["selection_wave"])]
    role_by_id = {item["id"]: item for item in roles["roles"]["selection"]}
    candidate = load_window_means(args.candidate_run)
    stock = load_window_means(args.stock_run)
    if set(candidate) != set(expected) or set(stock) != set(expected):
        raise RuntimeError("paired runs do not cover the exact frozen wave")
    candidate_values = np.asarray([candidate[key] for key in expected], dtype=np.float64)
    stock_values = np.asarray([stock[key] for key in expected], dtype=np.float64)
    delta = candidate_values - stock_values
    bootstrap = plan["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    indices = rng.integers(0, len(expected), size=(bootstrap["replicates"], len(expected)))
    delta_ci = bca_mean_interval(delta, delta[indices].mean(1))
    candidate_gmean, stock_gmean = _gmean(candidate_values), _gmean(stock_values)
    relative = 1.0 - candidate_gmean / stock_gmean
    domain_rows: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(expected, delta, strict=True):
        domain_rows[role_by_id[key]["domain"]].append(float(value))
    domain_deltas = {domain: float(np.mean(items)) for domain, items in sorted(domain_rows.items())}
    negative_domains = sum(value < 0 for value in domain_deltas.values())
    passed = relative >= 0.03 and delta_ci[1] < 0 and negative_domains >= 3
    payload = {
        "schema": "glm53-down-h16-v4-selection-kld-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "role": "selection",
        "windows": len(expected),
        "candidate_arithmetic_mean_kld": float(candidate_values.mean()),
        "stock_arithmetic_mean_kld": float(stock_values.mean()),
        "candidate_geometric_mean_kld": candidate_gmean,
        "stock_geometric_mean_kld": stock_gmean,
        "relative_geometric_mean_improvement": relative,
        "mean_delta_kld": float(delta.mean()),
        "delta_ci95_bca": delta_ci,
        "domain_mean_delta_kld": domain_deltas,
        "negative_domain_count": negative_domains,
        "wins": int((delta < 0).sum()),
        "bootstrap": bootstrap,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-qualified-down-h16-beats-stock" if passed else "fail-not-qualified",
        "runtime_proof": {"path": str(args.runtime_proof), "sha256": sha256_file(args.runtime_proof)},
        "run_manifests": {
            "candidate": {"path": str(args.candidate_manifest), "sha256": sha256_file(args.candidate_manifest)},
            "stock": {"path": str(args.stock_manifest), "sha256": sha256_file(args.stock_manifest)},
        },
        "candidate_records": [{"id": key, "mean_kld": candidate[key]} for key in expected],
        "stock_records": [{"id": key, "mean_kld": stock[key]} for key in expected],
        "confirmation_logits_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
