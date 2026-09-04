"""Freeze the first input-plus-output in-block H16 screen for GLM experts."""
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
        "schema": "glm53-two-sided-h16-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only engineering screen; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 384, "count": 128},
        },
        "arms": {
            "identity": {"input": "I", "output": "I"},
            "input-h16": {"input": "H16", "output": "I"},
            "output-h16": {"input": "I", "output": "H16"},
            "two-sided-h16": {"input": "H16", "output": "H16"},
        },
        "quantization": {
            "method": "full-Hessian GPTQ",
            "format": "exact ModelOpt NVFP4 E2M1 plus E4M3 per 16",
            "gate_up_scale": "one jointly refit FP32 global scale per expert arm",
            "stored_bpw": 4.5,
        },
        "estimand": "paired per-expert log causal full-MoE output-NMSE ratio",
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090405},
        "multiplicity": "Holm-adjust the three candidate-versus-identity comparisons; primary two-sided-versus-input-H16 must independently pass",
        "decision_rule": (
            "promote two-sided H16 only if it improves geometric-mean NMSE at least "
            "5 percent versus input-only H16, wins at least 12/16 experts, and the "
            "unadjusted paired BCa upper mean-log-ratio bound is below zero; output-only "
            "is diagnostic and cannot replace the preregistered primary"
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
