"""Freeze projection-family H16 attribution before looking at results."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-projection-h16-attribution-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only projection-family attribution; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {"strategy": "domain-balanced routed REAP samples per expert", "hessian_fit": {"offset": 0, "count": 256}, "evaluation": {"offset": 896, "count": 128}},
        "arms": {
            "identity": {"gate_up": "I", "down": "I"},
            "gate-up-h16": {"gate_up": "H16", "down": "I"},
            "down-h16": {"gate_up": "I", "down": "H16"},
            "all-h16": {"gate_up": "H16", "down": "H16"},
        },
        "quantization": "full-Hessian GPTQ, exact ModelOpt NVFP4 E2M1 plus E4M3/16, 4.5 bpw",
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090407},
        "multiplicity": "Holm-adjust exact paired sign-flip tests for gate-up-H16 and down-H16 versus all-H16",
        "decision_rule": "advance a projection-specific arm only if its improvement versus all-H16 is at least 3 percent, it wins at least 12/16 experts, its paired BCa upper log-ratio bound is below zero, and its Holm-adjusted p value is below 0.05; if both pass choose the larger improvement and require fresh fit-role validation",
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
