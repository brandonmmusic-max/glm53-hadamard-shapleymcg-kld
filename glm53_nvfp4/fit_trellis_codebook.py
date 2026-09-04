"""Fit one global EXL3-state to E2M1 table on GLM BF16 weight tiles."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_nvfp4 import optimize_e2m1_codebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--initial-law", choices=("mcg", "mul1", "sqg-normal"), default="mcg")
    parser.add_argument("--initial-compander-scale", type=float, default=2.75)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--search-grid", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid expert range")

    started = time.time()
    source = IndexedCheckpoint(args.source, args.source_index)
    carrier = IndexedCheckpoint(args.carrier)
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    training = []
    tensor_names = []
    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{projection}.weight"
            weight = source.get(name).to(args.device).float()
            global_scale = carrier.get(f"{name}_scale_2").to(args.device)
            training.append((weight, global_scale))
            tensor_names.append(name)

    codebook, history = optimize_e2m1_codebook(
        training,
        bits=args.bits,
        initial_law=args.initial_law,
        initial_compander_scale=args.initial_compander_scale,
        iterations=args.iterations,
        search_grid=args.search_grid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"codebook_e4m3": codebook},
        str(args.output),
        metadata={
            "schema": "glm53-native-nvfp4-learned-codebook.v1",
            "endpoint": "exact E2M1",
            "bits": str(args.bits),
            "initial_law": args.initial_law,
        },
    )
    receipt = {
        "schema": "glm53-native-nvfp4-codebook-fit.v1",
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "tensor_names": tensor_names,
        "bits": args.bits,
        "initial_law": args.initial_law,
        "initial_compander_scale": args.initial_compander_scale,
        "iterations": args.iterations,
        "search_grid": args.search_grid,
        "ldlq": False,
        "rotation": False,
        "history": [step.__dict__ for step in history],
        "source_index": {
            "path": str(args.source_index),
            "sha256": sha256_file(args.source_index),
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
