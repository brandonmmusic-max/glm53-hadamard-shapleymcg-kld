"""Freeze the 16-expert Hessian-aware P8 trellis projection screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, action="append", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-hessian-trellis-16expert-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering qualification before end-to-end KLD",
        "layer": 3,
        "experts": EXPERTS,
        "projections": ["gate_proj", "up_proj"],
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
            "encoder": "native-tile Viterbi plus GPTQ-style full-Hessian inter-group error feedback and static within-group activation order",
            "scale_refit_iterations": 2,
            "stored_bpw": 4.25,
            "isa_cost": "mxf8f6f4 issues twice the MMA instructions of NVFP4",
        },
        "controls": {
            "primary": "full-Hessian GPTQ NVFP4, 4.5 bpw",
            "diagnostic": ["plain MCG K4 Viterbi", "scalar E4M3 K32"],
        },
        "estimand": "per expert, geometric mean of gate/up held-out routed projection-output NMSE; paired log ratio candidate/control",
        "decision_rule": "pass only if geometric-mean candidate improvement versus GPTQ NVFP4 is at least 10 percent, the paired expert BCa upper bound on mean log ratio is below zero, and at least 12 of 16 experts improve",
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 20000, "seed": 20260904},
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
        "prior": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.prior_analysis
        ],
        "inputs": {
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_manifest), "sha256": sha256_file(args.capture_manifest)},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
