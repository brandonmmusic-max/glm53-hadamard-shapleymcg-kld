"""Freeze a block-local signed-H16 engineering screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [5, 15, 25, 35, 45, 55, 65, 71, 80, 90, 100, 110, 120, 130, 140, 150]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--variants-per-block", type=int, default=16)
    parser.add_argument(
        "--transform-family",
        choices=("signed-h16", "structured-dph16"),
        default="signed-h16",
    )
    parser.add_argument("--evaluation-offset", type=int, default=512)
    parser.add_argument(
        "--selection-method",
        choices=("rtn-hessian", "block-gptq-hessian"),
        default="rtn-hessian",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prior = json.loads(args.prior_analysis.read_text())
    if (
        prior.get("decision") not in {
            "fail-do-not-open-v4-selection",
            "fail-do-not-open-new-teacher-role",
        }
        or prior.get("protected_roles_opened") != []
    ):
        raise RuntimeError("prior analysis is not an eligible preserved fit-only failure")
    payload = {
        "schema": "glm53-blocklocal-signed-h16-screen-plan.v1",
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
            "name": "per-expert-per-tensor-per-physical-block signed H16",
            "variants_per_block": args.variants_per_block,
            "transform_family": args.transform_family,
            "variant_zero": "fixed H16",
            "selection": (
                "minimum Hessian-weighted exact 16-column GPTQ reconstruction error independently in each physical input block"
                if args.selection_method == "block-gptq-hessian"
                else "minimum Hessian-weighted exact-grid RTN reconstruction error independently in each physical input block"
            ),
            "selection_method": args.selection_method,
            "global_scale_refits": 2,
            "materialization": "full-Hessian GPTQ after block-local selection",
            "runtime_contract": "each projection applies its own block-diagonal input transform in its fused prologue",
            "index_overhead": (
                "one byte per physical input block; about 0.00033 bpw for a "
                "1536-row projection; 256 signed-permutation descriptors occupy 2560 bytes"
                if args.transform_family == "structured-dph16"
                else "four bits per physical input block for the 16-member bank"
            ),
        },
        "controls": ["identity full-Hessian GPTQ", "fixed H16 full-Hessian GPTQ"],
        "format": "exact ModelOpt NVFP4 E2M1 plus E4M3 per 16",
        "stored_bpw": 4.5,
        "estimand": "paired per-expert log causal full-MoE output-NMSE ratio",
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090406},
        "decision_rule": (
            "promote only if block-local signed H16 improves geometric-mean NMSE "
            "at least 3 percent versus fixed H16, wins at least 12/16 experts, and "
            "the paired BCa upper mean-log-ratio bound is below zero"
        ),
        "protected_boundary": "V4 selection and 28 confirmation logits remain unopened",
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
