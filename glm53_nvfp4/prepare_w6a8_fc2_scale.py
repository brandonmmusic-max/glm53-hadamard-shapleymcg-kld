"""Freeze the per-expert FC2 E4M3 input-scale calibration screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-w6a8-fc2-scale-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit",
        "layer": 3,
        "experts": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71],
        "phase_grid": [2.0 ** (index / 8.0) for index in range(8)],
        "tuning": "offset 0 n32 domain-balanced REAP routes; choose a2 independently per expert by minimum routed-output NMSE",
        "validation": "freeze the 16 per-expert phases, compare against unit a2 on offset 32 n32 without reselection",
        "a1_gscale": 1.0,
        "dynamic_fc2_scale": False,
        "decision_rule": {
            "material": "validation geometric-mean output NMSE improves at least 5 percent and at least 12 of 16 experts improve",
            "expand": "only a material result permits fitting all 288 layer-3 experts and an end-to-end KLD arm",
            "fail": "otherwise retain unit a2 and treat native E4M3 activation quantization as the unresolved P8 blocker",
        },
        "protected_boundary": "selection, confirmation, final, and 28 confirmation logits remain unopened",
        "ldlq": False,
        "prior_stage_result": {"path": str(args.stage_result), "sha256": sha256_file(args.stage_result)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
