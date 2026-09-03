"""Analyze sealed layer-Shapley coalition KLD and choose the 6-bpw allocation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    means = {}
    evidence = []
    for cid in design["coalitions"]:
        run_id = f"shapley-{cid}"
        path = args.run_root / f"run-{run_id}.json"
        run = json.loads(path.read_text())
        if run.get("status") != "complete" or run.get("role") != "conditional-fit":
            raise RuntimeError(f"incomplete or wrong-role Shapley run {path}")
        means[cid] = float(run["mean_of_window_means"])
        evidence.append({"coalition_id": cid, "run": str(path), "sha256": sha256_file(path), "mean_kld": means[cid]})

    samples: dict[int, list[float]] = {layer: [] for layer in design["layers"]}
    running_allocations = []
    for path in design["permutations"]:
        ids = path["coalition_ids"]
        for i, layer in enumerate(path["order"]):
            samples[layer].append(means[ids[i]] - means[ids[i + 1]])
        current = {layer: float(np.mean(values)) for layer, values in samples.items() if values}
        top = sorted(current, key=lambda layer: (-current[layer], layer))[: design["allocation"]["mxfp6_layers"]]
        running_allocations.append(top)

    rows = []
    for layer in design["layers"]:
        values = np.asarray(samples[layer], dtype=np.float64)
        rows.append({
            "layer": layer,
            "marginal_kld_reductions": values.tolist(),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
            "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
            "samples": len(values),
        })
    selected = sorted(
        row["layer"]
        for row in sorted(rows, key=lambda row: (-row["mean"], row["layer"]))[
            : design["allocation"]["mxfp6_layers"]
        ]
    )
    overlap = len(set(running_allocations[0]) & set(running_allocations[-1])) if len(running_allocations) > 1 else None
    full_id = next(cid for cid, layers in design["coalitions"].items() if len(layers) == len(design["layers"]))
    empty_id = next(cid for cid, layers in design["coalitions"].items() if not layers)
    payload = {
        "schema": "glm53-nvfp4-v3.layer-shapley-analysis.v1",
        "design": str(args.design),
        "design_sha256": sha256_file(args.design),
        "baseline_empty_kld": means[empty_id],
        "full_mxfp6_kld": means[full_id],
        "full_path_delta": means[empty_id] - means[full_id],
        "sum_mean_shapley": float(sum(row["mean"] for row in rows)),
        "efficiency_remainder": float((means[empty_id] - means[full_id]) - sum(row["mean"] for row in rows)),
        "per_layer": rows,
        "selected_mxfp6_layers": selected,
        "selected_nvfp4_layers": sorted(set(design["layers"]) - set(selected)),
        "convergence": {
            "top35_after_each_permutation": running_allocations,
            "top35_overlap_after_first_vs_final": overlap,
            "top35_jaccard_after_first_vs_final": overlap / 35.0 if overlap is not None else None,
        },
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_mxfp6_layers": selected, "efficiency_remainder": payload["efficiency_remainder"]}, sort_keys=True))


if __name__ == "__main__":
    main()
