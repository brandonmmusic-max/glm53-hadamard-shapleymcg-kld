"""Fit-only independent selection of Qwen R_in and R_mid candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .block_rotation import hadamard16
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument(
        "--exclude-fixed",
        action="store_true",
        help="select the best learned matrix per family for a diagnostic arm",
    )
    args = parser.parse_args()
    if len(args.candidate) != 4 or len(args.receipt) != 4:
        raise ValueError("expected identity/H16 learned candidates for both families")

    grouped: dict[str, list[dict]] = {"in": [], "mid": []}
    for candidate, receipt_path in zip(args.candidate, args.receipt, strict=True):
        receipt = json.loads(receipt_path.read_text())
        family = receipt["algorithm"].get("family")
        if family not in grouped or receipt["layer"] != args.layer:
            raise ValueError(f"invalid family/layer in {receipt_path}")
        score_key = f"learned-{receipt['init']}"
        grouped[family].append(
            {
                "name": score_key,
                "candidate": str(candidate),
                "candidate_sha256": sha256_file(candidate),
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "fit_score": receipt["scores"][score_key],
            }
        )

    output_tensors = {}
    selections = {}
    for family, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"expected two learned candidates for {family}")
        receipt = json.loads(Path(rows[0]["receipt"]).read_text())
        if not args.exclude_fixed:
            rows.extend(
                {
                    "name": fixed,
                    "candidate": fixed,
                    "candidate_sha256": fixed,
                    "receipt": rows[0]["receipt"],
                    "receipt_sha256": rows[0]["receipt_sha256"],
                    "fit_score": receipt["scores"][fixed],
                }
                for fixed in ("identity", "had16")
            )
        winner = min(rows, key=lambda row: (row["fit_score"], row["name"]))
        key = f"layer_{args.layer:03d}_{family}"
        if winner["name"] == "identity":
            rotation = torch.eye(16)
        elif winner["name"] == "had16":
            rotation = hadamard16()
        else:
            rotation = load_file(winner["candidate"], device="cpu")[key]
        output_tensors[key] = rotation.clone().contiguous()
        selections[family] = {"candidates": rows, "winner": winner}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output_tensors,
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v6.independently-selected-qwen-rotations.v1",
            "layer": str(args.layer),
        },
    )
    result = {
        "schema": "glm53-nvfp4-v6.independent-qwen-family-selection.v1",
        "role": "fit",
        "layer": args.layer,
        "selections": selections,
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    args.selection_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.selection_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
