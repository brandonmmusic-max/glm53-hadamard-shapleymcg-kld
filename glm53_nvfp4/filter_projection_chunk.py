"""Create an immutable projection-only safetensors chunk and provenance receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--projection", choices=("gate_proj", "up_proj", "down_proj"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite projection chunk or receipt")
    source = load_file(str(args.input), device="cpu")
    selected = {name: tensor for name, tensor in source.items() if f".{args.projection}." in name}
    if not selected:
        raise ValueError(f"input contains no {args.projection} tensors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(selected, str(args.output), metadata={"schema": "glm53-projection-only-chunk.v1", "projection": args.projection})
    receipt = {
        "schema": "glm53-projection-only-chunk-receipt.v1",
        "projection": args.projection,
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output), "bytes": args.output.stat().st_size, "tensors": len(selected)},
        "selection_rule": f"tensor name contains '.{args.projection}.'",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
