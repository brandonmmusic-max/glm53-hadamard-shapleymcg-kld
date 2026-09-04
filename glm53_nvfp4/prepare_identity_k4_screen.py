"""Freeze the lossless-K4 endpoint optimizer screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_p4_down_h16_screen import EXPERTS
from .shard_index import sha256_file


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
        "schema": "glm53-identity-k4-down-h16-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only engineering screen; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 1408, "count": 128},
        },
        "candidate": {
            "name": "lossless identity-K4 stream plus down-only fixed H16",
            "encoder": "full-Hessian coordinate descent over all 16 physical E2M1 nibbles, static in-group act order, output-aware E4M3/16 and FP32 global scale refits",
            "sweeps": 2,
            "stored_bpw": 4.5,
            "runtime_endpoint": "ordinary packed E2M1 plus E4M3/16 consumed directly by mxf4nvf4",
            "reference_codec": "K4 sliding state; low four state bits decode bijectively to the physical nibble; bit-exact endpoint closure required",
            "runtime_table_bytes": 0,
            "rotation": "fixed H16 on down_proj input blocks only",
            "ldlq": False,
        },
        "controls": ["identity full-Hessian GPTQ NVFP4", "down-only H16 full-Hessian GPTQ NVFP4"],
        "common_path": "gate/up identity-GPTQ reconstructions are identical in every arm",
        "estimands": ["down_proj weight NMSE", "routed full-expert output NMSE"],
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090419},
        "decision_rule": (
            "promote only if optimized K4 improves geometric-mean routed output NMSE at least "
            "5 percent versus identity GPTQ and at least 1 percent versus down-H16 GPTQ, wins "
            "at least 12/16 experts versus each, and both paired BCa upper bounds are below zero"
        ),
        "isa_cost_statement": "the endpoint is native NVFP4; direct nibble consumption dominates an equivalent K4 decode prologue, but speed remains unproven until measured",
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
