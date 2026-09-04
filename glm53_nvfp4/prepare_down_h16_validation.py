"""Freeze a disjoint-expert validation for the discovered down-only H16 arm."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [1, 11, 22, 32, 43, 53, 64, 75, 85, 94, 104, 113, 123, 132, 142, 151, 160, 168, 177, 185, 194, 202, 211, 219, 228, 236, 245, 253, 262, 270, 279, 287]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument(
        "--protected-boundary",
        default="V4 selection and 28 confirmation logits remain unopened until this rule passes",
    )
    parser.add_argument("--evaluation-offset", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-down-h16-disjoint-validation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only disjoint-expert validation; no teacher logits",
        "layer": args.layer,
        "experts": EXPERTS,
        "disjoint_from_discovery_experts": True,
        "sampling": {"strategy": "domain-balanced routed REAP samples per expert", "hessian_fit": {"offset": 0, "count": 256}, "evaluation": {"offset": args.evaluation_offset, "count": 128}},
        "arms": {
            "identity": {"gate_up": "I", "down": "I"},
            "gate-up-h16": {"gate_up": "H16", "down": "I"},
            "down-h16": {"gate_up": "I", "down": "H16"},
            "all-h16": {"gate_up": "H16", "down": "H16"},
        },
        "quantization": "full-Hessian GPTQ, exact ModelOpt NVFP4 E2M1 plus E4M3/16, 4.5 bpw",
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090408},
        "decision_rule": f"promote down-only H16 at layer {args.layer} only if it improves geometric-mean NMSE at least 5 percent versus identity, wins at least 20/32 experts versus identity, the paired BCa upper log-ratio bound versus identity is below zero, and its BCa upper log-ratio bound versus all-H16 is below log(1.02)",
        "protected_boundary": args.protected_boundary,
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
