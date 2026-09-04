"""Freeze the no-LDLQ P8 physical scale-contract comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "glm53-p8-scale-contract-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-developmental",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "role": "fit",
            "strategy": "domain-balanced",
            "offset": 448,
            "count": 128,
        },
        "arms": {
            "native_ue8m0_sfb": "normalized procedural-MCG E4M3 plus physical UE8M0/32 scale consumed by MX MMA",
            "identity_sfb": "multiply physical scale in prologue, round fully-scaled weights to E4M3, then use identity MX SFB",
        },
        "metrics_order": [
            "per-projection weight NMSE and exact fraction",
            "routed full-expert output NMSE",
        ],
        "estimand": "paired per-expert log routed-output NMSE ratio, identity SFB divided by native UE8M0 SFB",
        "decision_thresholds": {
            "maximum_identity_sfb_relative_regression": 0.02,
            "minimum_identity_sfb_nonregressions": 12,
        },
        "decision_rule": "use identity SFB only if its geometric-mean routed-output NMSE regresses no more than 2 percent and it is non-inferior on at least 12 of 16 experts; otherwise use the physical UE8M0 scale as native MX SFB. The BCa interval is reported as a development control and is not a stop rule.",
        "bootstrap": {"method": "paired BCa", "unit": "expert", "replicates": 50000, "seed": 2026090427},
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "window_uncertainty_policy": "recorded development control; scientific KLD claim gate remains unchanged",
        "isa_cost": "P8 mxf8f6f4 has twice the MMA issue count of NVFP4",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_root / "capture-manifest.json"), "sha256": sha256_file(args.capture_root / "capture-manifest.json")},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
            "dense": {"path": str(args.dense), "sha256": sha256_file(args.dense)},
            "codec": {"path": str(args.codec), "sha256": sha256_file(args.codec)},
        },
        "analysis_lock": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "screen_code_sha256": sha256_file(repo / "glm53_nvfp4/screen_p8_scale_contract.py"),
            "codec_code_sha256": sha256_file(repo / "glm53_nvfp4/trellis_mxf.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
