"""Freeze the causal full-expert P8 pseudoquant screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_hessian_trellis_screen import EXPERTS
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-hessian-trellis-full-16expert-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering causal pseudoquant screen before layer KLD",
        "layer": args.layer,
        "experts": EXPERTS,
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
            "causal_down_input": "candidate gate/up on E4M3 hidden carrier, SwiGLU clamp 10, then E4M3 K32 carrier",
            "physical_bpw": 4.25,
            "charged_bpw_for_gate": 4.5,
            "rate_note": "the acceptance comparison charges a reserved 0.25 padding plane so the candidate does not claim a rate advantage",
            "isa_cost": "mxf8f6f4 issues twice the MMA instructions of NVFP4",
        },
        "control": {
            "name": "full-Hessian GPTQ NVFP4",
            "bpw": 4.5,
            "carrier_for_this_encoder_test": "same E4M3 P8 activation carrier; this isolates weight encoder quality and is not a runtime-speed comparison",
        },
        "estimand": "per-expert route-weighted full MoE output NMSE after causal gate/up/down pseudoquant; paired log candidate/control ratio",
        "decision_rule": "pass only if geometric-mean full-expert NMSE improves at least 10 percent, paired expert BCa upper mean-log-ratio bound is below zero, and at least 12 of 16 experts improve",
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 20000, "seed": 20260904},
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
        "projection_analysis": {"path": str(args.projection_analysis), "sha256": sha256_file(args.projection_analysis)},
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
