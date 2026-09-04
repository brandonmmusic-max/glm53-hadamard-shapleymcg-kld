"""Freeze ridge tuning for the lossless-K4 endpoint encoder."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
        "schema": "glm53-identity-k4-ridge-tuning-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only hyperparameter tuning; no teacher logits",
        "layer": 3,
        "experts": [1, 22, 43, 64, 85, 104, 123, 142],
        "sampling": {"strategy": "domain-balanced", "hessian_fit": {"offset": 0, "count": 256}, "evaluation": {"offset": 1408, "count": 128}},
        "candidate": {
            "name": "lossless identity-K4 plus down-H16 ridge calibration",
            "ridge_ratio_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "sweeps": 1,
            "scale_refit": "output-aware E4M3/16 and FP32 global coordinate refit",
            "stored_bpw": 4.5,
            "ldlq": False,
        },
        "control": "down-only H16 full-Hessian GPTQ NVFP4",
        "selection_rule": "select one checkpoint-family ridge ratio by minimum geometric-mean held-out routed output NMSE; advance to disjoint-expert validation only if it improves at least 1 percent versus control",
        "validation_boundary": "the second 16-expert panel and teacher-logit roles remain unopened",
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
