"""Bundle one selected block rotation per routed layer for runtime loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from .block_rotation import orthogonality_error
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    tensors = {}
    inputs = []
    for path in args.input:
        item = load_file(str(path), device="cpu")
        if len(item) != 1:
            raise ValueError(f"{path} must contain exactly one rotation")
        key, tensor = next(iter(item.items()))
        if key in tensors:
            raise ValueError(f"duplicate {key}")
        if orthogonality_error(tensor) > 2e-4:
            raise ValueError(f"nonorthogonal {key}")
        tensors[key] = tensor.contiguous()
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    expected = {f"layer_{layer:03d}" for layer in range(3, 45)}
    if set(tensors) != expected:
        raise ValueError(f"rotation coverage mismatch: missing={sorted(expected-set(tensors))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={"schema": "glm53-nvfp4-v3.learned-block16-bundle.v1"})
    receipt = {"schema": "glm53-nvfp4-v3.rotation-bundle-receipt.v1", "layers": len(tensors), "inputs": inputs, "output": {"path": str(args.output), "sha256": sha256_file(args.output)}}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["output"], sort_keys=True))


if __name__ == "__main__":
    main()
