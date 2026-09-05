"""Close one complete V8 down-only butterfly candidate build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan_sha = sha256_file(args.plan)
    rotation_sha = sha256_file(args.rotation)
    rows = [json.loads(path.read_text()) for path in args.receipt]
    if len(rows) != 4:
        raise ValueError("complete TP4 build requires four receipts")
    ranges = sorted(tuple(row["expert_range"]) for row in rows)
    if ranges != [(0, 72), (72, 144), (144, 216), (216, 288)]:
        raise ValueError(f"expert coverage mismatch: {ranges}")
    if any(row["plan_sha256"] != plan_sha for row in rows):
        raise RuntimeError("receipt/plan mismatch")
    if any(row["rotation"]["sha256"] != rotation_sha for row in rows):
        raise RuntimeError("receipt/rotation mismatch")
    angles = {row["angle_pi"] for row in rows}
    if len(angles) != 1:
        raise RuntimeError("mixed rotation angles")
    if any(row["physical"]["payload_bpw"] > 4.50001 for row in rows):
        raise RuntimeError("down payload exceeds exact NVFP4 rate tolerance")
    if any(row["rotation"]["table_bytes"] != 1024 for row in rows):
        raise RuntimeError("rotation table is not exactly 1024 bytes")
    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "pass" or validation.get("experts") != 288:
        raise RuntimeError("layer validation did not pass")
    overlay_path = args.candidate / "OVERLAY.json"
    index_path = args.candidate / "model.safetensors.index.json"
    overlay = json.loads(overlay_path.read_text())
    if overlay.get("redirected_tensors") != 288 * 3:
        raise RuntimeError("candidate does not redirect exactly three down tensors per expert")
    logical = sum(row["logical_elements"] for row in rows)
    physical = sum(row["physical"]["payload_bytes"] for row in rows)
    result = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-full-build.v1",
        "status": "pass",
        "plan_sha256": plan_sha,
        "angle_pi": next(iter(angles)),
        "experts": 288,
        "projection": "down_proj",
        "logical_elements": logical,
        "physical_payload_bytes": physical,
        "physical_payload_bpw": 8.0 * physical / logical,
        "rotation": {
            "path": str(args.rotation.resolve()),
            "sha256": rotation_sha,
            "table_bytes": 1024,
            "per_block_or_expert_descriptor_bytes": 0,
        },
        "receipts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.receipt
        ],
        "validation": {
            "path": str(args.validation.resolve()),
            "sha256": sha256_file(args.validation),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "index_sha256": sha256_file(index_path),
            "overlay_sha256": sha256_file(overlay_path),
            "redirected_tensors": overlay["redirected_tensors"],
        },
        "runtime": (
            "physical ModelOpt NVFP4 down weights plus existing Humming BF16 mid-rotation hook; "
            "rotation is not yet fused into the MMA prologue"
        ),
        "protected_roles_opened": [],
        "ldlq": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
