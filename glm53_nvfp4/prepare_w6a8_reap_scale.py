"""Freeze the REAP expert-output input-scale screen before device execution."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--dense-receipt", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    experts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]
    phase_grid = [2.0 ** (index / 8.0) for index in range(8)]
    payload = {
        "schema": "glm53-w6a8-reap-scale-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit",
        "layer": 3,
        "experts": experts,
        "sampling": "64 domain-balanced routed REAP samples per expert, split selection offset 0 n32 and validation offset 32 n32",
        "arms": [
            {"id": "S0", "a1_gscale": 1.0, "a2_gscale": 1.0},
            {"id": "S1-grid", "a1_gscale": phase_grid, "a2_gscale": 1.0},
        ],
        "selection_rule": "choose one shared a1 phase minimizing geometric-mean expert output NMSE on offset 0; ties choose smaller phase",
        "validation_rule": "evaluate the frozen shared phase on offset 32 without reselection",
        "decision_rule": {
            "fc1_scale_material": "validation geometric-mean output NMSE improves at least 5 percent and at least 12 of 16 experts improve",
            "fc1_scale_not_dominant": "otherwise retain unit a1 and proceed to isolated FC2 activation/epilogue diagnosis",
        },
        "held_constant": [
            "exact packed and decoded MXFP6 weights",
            "real layer-3 hidden states and route weights",
            "model swiglu_limit 10",
            "precise activation math",
            "dynamic FC2 scaling",
        ],
        "protected_boundary": "selection, confirmation, final, and the 28 confirmation logits remain unopened",
        "isa_cost_statement": "P8 mxf8f6f4 has twice the MMA issue count of NVFP4; this is a fit-role carrier diagnosis, not a speed claim",
        "ldlq": False,
        "inputs": {
            "native_receipt": {"path": str(args.native_receipt), "sha256": sha256_file(args.native_receipt)},
            "dense_receipt": {"path": str(args.dense_receipt), "sha256": sha256_file(args.dense_receipt)},
            "capture_manifest": {"path": str(args.capture_manifest), "sha256": sha256_file(args.capture_manifest)},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
