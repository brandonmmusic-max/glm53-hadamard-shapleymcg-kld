"""Freeze disjoint-expert validation of the selected identity-K4 ridge."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [160, 168, 177, 185, 194, 202, 211, 219, 228, 236, 245, 253, 262, 270, 279, 287]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tuning = json.loads(args.tuning_analysis.read_text())
    if tuning["decision"] != "pass-freeze-and-validate":
        raise ValueError("tuning analysis did not authorize validation")
    payload = {
        "schema": "glm53-identity-k4-down-h16-validation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "disjoint-expert fit-only validation; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {"strategy": "domain-balanced", "hessian_fit": {"offset": 0, "count": 256}, "evaluation": {"offset": 1536, "count": 128}},
        "candidate": {
            "name": "lossless identity-K4 plus down-only H16",
            "ridge_ratio": tuning["selected_ridge_ratio"],
            "sweeps": 1,
            "stored_bpw": 4.5,
            "runtime_table_bytes": 0,
            "ldlq": False,
        },
        "controls": ["identity full-Hessian GPTQ NVFP4", "down-only H16 full-Hessian GPTQ NVFP4"],
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090420},
        "decision_rule": "promote only if optimized K4 improves geometric-mean routed output NMSE at least 5 percent versus identity GPTQ and at least 1 percent versus down-H16 GPTQ, wins at least 12/16 experts versus each, and both paired BCa upper bounds are below zero",
        "protected_boundary": "teacher-logit roles and 28 confirmation logits remain unopened",
        "prior": {"path": str(args.tuning_analysis), "sha256": sha256_file(args.tuning_analysis)},
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
