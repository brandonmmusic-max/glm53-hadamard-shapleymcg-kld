"""Freeze checkpoint-family projection-wise MCG alpha fitting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_learned_t12_fit import FIT_EXPERTS, VALIDATION_EXPERTS
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--prior-scalar-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-p8-mcg-projection-alpha-fit-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-only checkpoint-family procedural-law tuning",
        "layer": 3,
        "fit_experts": FIT_EXPERTS,
        "reserved_validation_experts": VALIDATION_EXPERTS,
        "projections": ["gate_proj", "up_proj", "down_proj"],
        "sampling": {
            "hessian_fit": {"offset": 0, "count": 256},
            "alpha_selection": {"offset": 256, "count": 128},
            "strategy": "domain-balanced routed REAP samples per expert",
        },
        "alpha_grid": [1.5, 1.75, 2.0, 2.25, 2.5, 2.75],
        "selection_rule": "independently per projection choose the alpha with minimum geometric-mean held-out projection-output NMSE across all 16 fit experts; ties choose the smaller alpha; freeze all three before causal validation on reserved experts",
        "encoder": "K4 procedural MCG Viterbi plus GPTQ-style full-Hessian inter-group feedback, static within-group act order, and two UE8M0 scale refits",
        "rate": "4.25 physical bpw; charge 4.5 bpw against GPTQ NVFP4; three alpha constants have negligible model-level storage",
        "runtime": "procedural MCG decoder, zero LUT bytes, E4M3/UE8M0 K32 P8",
        "isa_cost": "P8 mxf8f6f4 issues twice the MMA instructions of NVFP4",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "inputs": {
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_manifest), "sha256": sha256_file(args.capture_manifest)},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
            "prior_scalar_analysis": {"path": str(args.prior_scalar_analysis), "sha256": sha256_file(args.prior_scalar_analysis)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
