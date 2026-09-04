"""Freeze the exact-NVFP4 P4 trellis plus down-H16 causal screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-p4-down-h16-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only engineering screen; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 1280, "count": 128},
        },
        "candidate": {
            "name": "P4 K4 procedural-MCG trellis plus down-only fixed H16",
            "encoder": "native 16x16 Viterbi plus full-Hessian GPTQ-style inter-group error feedback, static in-group act order, two joint scale refits",
            "runtime_endpoint": "packed E2M1 nibbles plus E4M3/16 scales for mxf4nvf4 m16n8k64",
            "runtime_decoder": "per-element sliding 16-bit state using procedural MCG; no dense BF16 reconstruction and no runtime LUT",
            "bits": 4,
            "stored_bpw": 4.5,
            "rotation": "fixed H16 on down_proj input blocks only",
            "ldlq": False,
        },
        "controls": [
            "identity full-Hessian GPTQ NVFP4 at 4.5 bpw",
            "down-only fixed-H16 full-Hessian GPTQ NVFP4 at 4.5 bpw",
        ],
        "common_path": "gate/up identity-GPTQ reconstructions are identical in every arm",
        "estimands": ["down_proj weight NMSE", "routed full-expert output NMSE"],
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090418},
        "decision_rule": (
            "promote only if P4 improves geometric-mean routed output NMSE at least 5 percent "
            "versus identity GPTQ and at least 1 percent versus fixed down-H16 GPTQ, wins at "
            "least 12/16 experts versus each, and both paired BCa upper bounds are below zero"
        ),
        "isa_cost_statement": "P4 targets the NVFP4 mxf4nvf4 speed class; speed is unproven until the fused prologue is benchmarked",
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
