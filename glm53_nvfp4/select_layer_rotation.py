"""Select the lower fit-surrogate learned rotation from two sealed initializations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != len(args.receipt) or len(args.candidate) < 2:
        raise ValueError("candidate and receipt lists must have equal length >= 2")
    rows = []
    for candidate, receipt_path in zip(args.candidate, args.receipt, strict=True):
        receipt = json.loads(receipt_path.read_text())
        if receipt["layer"] != args.layer:
            raise ValueError(f"wrong layer in {receipt_path}")
        rows.append({"candidate": str(candidate), "candidate_sha256": sha256_file(candidate), "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path), "init": receipt["init"], "fit_score": receipt["scores"]["learned"]})
    winner = min(rows, key=lambda row: (row["fit_score"], row["candidate_sha256"]))
    key = f"layer_{args.layer:03d}"
    tensor = load_file(winner["candidate"], device="cpu")[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: tensor.contiguous()}, str(args.output), metadata={"schema": "glm53-nvfp4-v3.selected-learned-block16-layer.v1", "layer": str(args.layer), "selected_init": winner["init"]})
    result = {"schema": "glm53-nvfp4-v3.rotation-fit-selection.v1", "role": "fit", "layer": args.layer, "candidates": rows, "selected_init": winner["init"], "selected_fit_score": winner["fit_score"], "output": {"path": str(args.output), "sha256": sha256_file(args.output)}}
    args.selection_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.selection_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"layer": args.layer, "selected_init": winner["init"], "fit_score": winner["fit_score"]}))


if __name__ == "__main__":
    main()
