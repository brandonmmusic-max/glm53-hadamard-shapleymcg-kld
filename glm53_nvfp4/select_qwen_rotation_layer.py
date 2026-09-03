"""Select one independently learned R_in/R_mid pair using fit-role scores."""
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
        "--include-fixed",
        action="store_true",
        help="also allow identity or fixed H16 to win the fit-only selection",
    )
    args = parser.parse_args()
    if len(args.candidate) != len(args.receipt) or len(args.candidate) < 2:
        raise ValueError("candidate and receipt lists must have equal length >= 2")
    rows = []
    for candidate, receipt_path in zip(args.candidate, args.receipt, strict=True):
        receipt = json.loads(receipt_path.read_text())
        if receipt["layer"] != args.layer:
            raise ValueError(f"wrong layer in {receipt_path}")
        score_key = f"learned-{receipt['init']}"
        rows.append(
            {
                "candidate": str(candidate),
                "candidate_sha256": sha256_file(candidate),
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "init": receipt["init"],
                "fit_score": receipt["scores"][score_key],
            }
        )
    keys = (f"layer_{args.layer:03d}_in", f"layer_{args.layer:03d}_mid")
    choices = list(rows)
    if args.include_fixed:
        first_receipt = json.loads(args.receipt[0].read_text())
        choices.extend(
            {
                "candidate": name,
                "candidate_sha256": name,
                "receipt": str(args.receipt[0]),
                "receipt_sha256": sha256_file(args.receipt[0]),
                "init": name,
                "fit_score": first_receipt["scores"][name],
            }
            for name in ("identity", "had16")
        )
    winner = min(
        choices, key=lambda row: (row["fit_score"], row["candidate_sha256"])
    )
    if winner["candidate"] == "identity":
        fixed = torch.eye(16)
        tensors = {key: fixed.clone().contiguous() for key in keys}
    elif winner["candidate"] == "had16":
        fixed = hadamard16()
        tensors = {key: fixed.clone().contiguous() for key in keys}
    else:
        tensors = load_file(winner["candidate"], device="cpu")
        if set(tensors) != set(keys):
            raise ValueError(f"rotation keys mismatch: {sorted(tensors)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: tensors[key].contiguous() for key in keys},
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v5.selected-qwen-rotation-pair.v1",
            "layer": str(args.layer),
            "selected_init": winner["init"],
        },
    )
    result = {
        "schema": "glm53-nvfp4-v5.qwen-rotation-fit-selection.v1",
        "role": "fit",
        "layer": args.layer,
        "candidates": choices,
        "selected_init": winner["init"],
        "selected_fit_score": winner["fit_score"],
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    args.selection_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.selection_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
