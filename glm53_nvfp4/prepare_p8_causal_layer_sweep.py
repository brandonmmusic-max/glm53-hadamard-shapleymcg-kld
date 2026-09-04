"""Freeze the exhaustive causal P8 layer screen before execution."""
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
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--encoder-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-p8-causal-all-layer-sweep-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering exhaustive causal pseudoquant screen before any new subset KLD",
        "layers": list(range(3, 45)),
        "experts_per_layer": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71],
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 256, "count": 128},
        },
        "candidate": {
            "product": "P8",
            "bits": 4,
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0",
            "law": "procedural MCG alpha=2.0",
            "encoder": "native-tile Viterbi plus full-Hessian GPTQ-style inter-group error feedback and static within-group activation order",
            "physical_bpw": 4.25,
            "charged_bpw": 4.5,
        },
        "control": {"name": "full-Hessian GPTQ NVFP4", "bpw": 4.5},
        "shared_carrier": "candidate and control both use E4M3 K32 activation pseudoquant in the causal MoE calculation",
        "per_layer_decision_rule": (
            "eligible for later KLD subset design only if geometric-mean full-expert NMSE improves at least "
            "10 percent, paired expert BCa upper mean-log-ratio bound is below zero, and at least 12 of 16 "
            "experts improve"
        ),
        "ranking_rule": "report every layer and rank eligible layers by geometric-mean causal full-expert NMSE improvement; do not treat this local ranking as Shapley attribution",
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 20000, "seed": 20260904},
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is a quality product",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits are not read",
        "inputs": {
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_manifest),
            "roles": _file(args.roles),
            "encoder_analysis": _file(args.encoder_analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
