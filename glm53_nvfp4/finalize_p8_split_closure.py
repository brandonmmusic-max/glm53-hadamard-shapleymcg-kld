"""Validate and seal the frozen P8 split-prefill arithmetic repeat."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    repeat = json.loads(args.repeat.read_text())
    thresholds = plan["thresholds"]
    expected = {(row["bits"], row["tokens"]) for row in plan["required_cells"]}
    observed = {(row["bits"], row["tokens"]) for row in repeat["cells"]}
    cells_pass = expected == observed and all(
        row["pass"]
        and row["finite"] == thresholds["finite"]
        and row["cosine"] > thresholds["cosine_min_exclusive"]
        and row["relative_l2"] < thresholds["relative_l2_max_exclusive"]
        for row in repeat["cells"]
    )
    runtime_match = repeat["runtime_dynamic"]["sha256"] == plan["sources"]["runtime_dynamic"]
    scale_match = repeat.get("weight_scale_contract") == plan.get("weight_scale_contract")
    codebook_match = repeat.get("codebook_selection") == plan.get("codebook_selection")
    boundary_match = repeat.get("boundary") == plan.get("activation_boundary")
    passed = (
        repeat["decision"] == "pass"
        and cells_pass
        and runtime_match
        and scale_match
        and codebook_match
        and boundary_match
    )
    payload = {
        "schema": "glm53-p8-mcg-split-closure-result.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass-split-prefill-arithmetic-closure" if passed else "fail",
        "checks": {
            "required_cells": cells_pass,
            "runtime_source_identity": runtime_match,
            "weight_scale_contract": scale_match,
            "explicit_mcg_codebook": codebook_match,
            "activation_boundary": boundary_match,
        },
        "inputs": {"plan": sha256_file(args.plan), "repeat": sha256_file(args.repeat)},
        "scope": "split FC1/FC2 device arithmetic only; not end-to-end KLD, production-model integration, speed, or determinism qualification",
        "table_bytes": 0,
        "ldlq": False,
        "strict_kld_gate_relaxed": False,
        "engineering_blocker_relaxed": True,
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
