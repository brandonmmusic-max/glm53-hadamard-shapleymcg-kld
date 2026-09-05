"""Close a header-repaired overlay against the original physical build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-full-build", type=Path, required=True)
    parser.add_argument("--source-candidate", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repair-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    original = json.loads(args.source_full_build.read_text())
    source_receipt_path = args.source_candidate / "BF16_LAYER_RECEIPT.json"
    target_receipt_path = args.candidate / "BF16_LAYER_RECEIPT.json"
    if (
        original.get("status") != "pass"
        or original.get("candidate_receipt_sha256") != sha256_file(source_receipt_path)
    ):
        raise RuntimeError("source physical build does not close")
    source_receipt = json.loads(source_receipt_path.read_text())
    target_receipt = json.loads(target_receipt_path.read_text())
    repairs = [json.loads(path.read_text()) for path in args.repair_receipt]
    if len(repairs) != len(source_receipt["chunks"]) or any(
        item.get("status") != "pass" or item.get("payload_identity")
        != "the output payload is byte-for-byte copied from source"
        for item in repairs
    ):
        raise RuntimeError("header repair set is incomplete")
    source_map = {item["path"]: item["sha256"] for item in source_receipt["chunks"]}
    target_map = {item["path"]: item["sha256"] for item in target_receipt["chunks"]}
    if (
        {item["source"]: item["source_sha256"] for item in repairs} != source_map
        or {item["output"]: item["output_sha256"] for item in repairs} != target_map
    ):
        raise RuntimeError("repair receipts do not bridge the two overlays exactly")
    payload = {
        **original,
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-full-build.v2",
        "candidate_receipt_sha256": sha256_file(target_receipt_path),
        "source_full_build": {
            "path": str(args.source_full_build.resolve()),
            "sha256": sha256_file(args.source_full_build),
        },
        "source_candidate_receipt_sha256": sha256_file(source_receipt_path),
        "header_repairs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.repair_receipt
        ],
        "header_repair_interpretation": (
            "safetensors metadata-only correction; every BF16 tensor payload byte is unchanged"
        ),
        "finalizer_code_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
