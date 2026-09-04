"""Decode one exact ModelOpt NVFP4 chunk to a BF16 weight-only control chunk."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .modelopt import PackedNVFP4, dequantize
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    started = time.time()
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(args.input), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
        keys = set(handle.keys())
        weight_names = sorted(name for name in keys if name.endswith(".weight"))
        expected = {
            suffix_name
            for name in weight_names
            for suffix_name in (name, f"{name}_scale", f"{name}_scale_2")
        }
        if keys != expected:
            raise RuntimeError("input is not a complete weight/scale/scale_2 ModelOpt chunk")
        for name in weight_names:
            packed = PackedNVFP4(
                weight=handle.get_tensor(name),
                weight_scale=handle.get_tensor(f"{name}_scale"),
                weight_scale_2=handle.get_tensor(f"{name}_scale_2"),
            )
            tensors[name] = dequantize(packed).to(torch.bfloat16).contiguous()

    layer = source_metadata.get("layer")
    if layer is None:
        raise RuntimeError("source chunk has no layer metadata")
    metadata = {
        "schema": "glm53-gptq-nvfp4-bf16-control-chunk.v1",
        "layer": str(layer),
        "source_chunk_sha256": sha256_file(args.input),
        "decode": "exact ModelOpt E2M1 times logical E4M3/16 scale times FP32 global scale, then BF16 round",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    receipt = {
        "schema": "glm53-gptq-nvfp4-bf16-control-chunk-receipt.v1",
        "source": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": sha256_file(args.input),
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "layer": int(layer),
        "weight_tensors": len(tensors),
        "decode": metadata["decode"],
        "output_dtype": "BF16",
        "elapsed_seconds": time.time() - started,
        "ldlq": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
