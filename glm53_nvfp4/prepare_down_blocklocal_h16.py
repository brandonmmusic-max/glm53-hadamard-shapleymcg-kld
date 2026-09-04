"""Freeze a down-only, per-physical-block structured-H16 screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [
    1, 11, 22, 32, 43, 53, 64, 75, 85, 94, 104, 113, 123, 132, 142, 151,
    160, 168, 177, 185, 194, 202, 211, 219, 228, 236, 245, 253, 262, 270,
    279, 287,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-offset", type=int, default=1152)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-down-blocklocal-h16-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only engineering screen; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": args.evaluation_offset, "count": 128},
        },
        "candidate": {
            "name": "down-only per-expert per-physical-block structured D-P-H16",
            "scope": "gate_proj and up_proj remain identically unrotated in every arm; only down_proj changes",
            "variants_per_block": 256,
            "transform_family": "structured-dph16",
            "variant_zero": "fixed H16",
            "selection_method": "block-gptq-hessian",
            "selection": "minimum Hessian-weighted exact 16-column GPTQ reconstruction error per physical down_proj input block",
            "global_scale_refits": 2,
            "materialization": "full-Hessian GPTQ after block-local selection",
            "runtime_contract": "per-expert down-projection prologue applies its block-indexed signed permutation and H16 before native NVFP4 MMA",
            "index_overhead": "one byte per 16-value input block per expert down tensor; 256 deterministic descriptors occupy 2560 bytes",
            "ldlq": False,
        },
        "controls": [
            "identity down_proj full-Hessian GPTQ",
            "fixed H16 down_proj full-Hessian GPTQ",
        ],
        "common_path": "one identical identity-GPTQ gate/up reconstruction is reused by all arms",
        "format": "exact ModelOpt NVFP4 E2M1 plus E4M3 per 16",
        "stored_bpw": 4.5,
        "estimands": [
            "paired per-expert down_proj weight NMSE",
            "paired per-expert routed full-MoE output NMSE",
        ],
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090417},
        "decision_rule": (
            "promote only if block-local down H16 improves geometric-mean routed output NMSE "
            "at least 3 percent versus fixed down-only H16, wins at least 20/32 experts, "
            "and the paired BCa upper mean-log-ratio bound is below zero"
        ),
        "protected_boundary": "no teacher-logit role is opened by this screen; confirmation logits remain unopened",
        "prior": {"path": str(args.prior_analysis), "sha256": sha256_file(args.prior_analysis)},
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
