"""Freeze a broad P8 projection screen over the materialized block Hessians."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--hessian-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--invalidated-causal-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    invalidation = json.loads(args.invalidated_causal_plan.read_text())
    if invalidation.get("decision") != "invalidated-before-result":
        raise RuntimeError("causal all-layer attempt was not explicitly invalidated")
    payload = {
        "schema": "glm53-p8-blockhessian-layer-sweep-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-only broad projection screen using already materialized block Hessians",
        "layers": list(range(7, 45)),
        "experts_per_layer": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71],
        "projections": ["gate_proj", "up_proj"],
        "candidate": {
            "product": "P8",
            "bits": 4,
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0",
            "law": "procedural MCG alpha=2.0",
            "encoder": "native-tile Viterbi with block-diagonal Hessian feedback and static within-group activation order",
            "physical_bpw": 4.25,
            "charged_bpw": 4.5,
        },
        "control": {"name": "block-Hessian GPTQ NVFP4", "bpw": 4.5},
        "estimand": "per-expert geometric mean of gate/up block-Hessian-weighted reconstruction error; paired log candidate/control ratio",
        "per_layer_decision_rule": "screen-positive only if geometric-mean error improves at least 10 percent, paired expert BCa upper mean-log-ratio bound is below zero, and at least 12 of 16 experts improve",
        "ranking_rule": "report every layer and rank screen-positive layers by geometric-mean weighted-error improvement",
        "interpretation_boundary": (
            "these stored Hessians contain independent 16x16 input groups and no held-out activations or down-projection carrier; "
            "the result is a projection-level design prior, not causal MoE NMSE, end-to-end KLD, or Shapley attribution"
        ),
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 20000, "seed": 2026090428},
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is a quality product",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits are not read",
        "inputs": {
            "source_index": _file(args.source_index),
            "hessian_manifest": _file(args.hessian_manifest),
            "roles": _file(args.roles),
            "invalidated_causal_plan": _file(args.invalidated_causal_plan),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
