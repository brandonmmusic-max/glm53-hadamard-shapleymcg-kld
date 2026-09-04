"""Freeze the native-compatible K32 E4M3 block-scale search screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--fc2-scale-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-w6a8-block-scale-search-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit",
        "layer": 3,
        "experts": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71],
        "samples": "offset 32 n32 domain-balanced REAP routes per expert",
        "baseline": "amax containment UE8M0 exponent per K32 block",
        "candidate": "choose exponent offset in {0,-1,-2} per K32 block by minimum local E4M3 reconstruction SSE; emit unchanged E4M3 payload plus one UE8M0 byte",
        "metric": "decoded-weight pseudoquant routed-output NMSE with both activation hops",
        "decision_rule": {
            "implement_kernel": "candidate geometric-mean NMSE improves at least 15 percent and at least 12 of 16 experts improve",
            "reject": "otherwise do not add prologue search cost",
        },
        "runtime_boundary": "this is pseudoquant only; a pass requires a bit-exact device implementation and renewed KLD gate",
        "protected_boundary": "selection, confirmation, final, and 28 confirmation logits remain unopened",
        "ldlq": False,
        "prior": {
            "stage": {"path": str(args.stage_result), "sha256": sha256_file(args.stage_result)},
            "fc2_scale": {"path": str(args.fc2_scale_result), "sha256": sha256_file(args.fc2_scale_result)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
