"""Empirically validate the carrier's physical NVFP4 ABI against pinned BF16 weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .modelopt import PackedNVFP4, dequantize, unpack_codes, unswizzle_block_scale
from .shard_index import IndexedCheckpoint, sha256_file


def error_metrics(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    reconstructed = reconstructed.float()
    delta = reconstructed - reference
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "relative_rmse": float(delta.square().mean().sqrt() / reference.square().mean().sqrt()),
        "mean_absolute_error": float(delta.abs().mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(reference.flatten(), reconstructed.flatten(), dim=0)),
    }


def wrong_swizzled_storage_dequant(packed: PackedNVFP4) -> torch.Tensor:
    codes = unpack_codes(packed.weight, low_first=True)
    blocks = codes.reshape(codes.shape[0], codes.shape[1] // 16, 16)
    scale = unswizzle_block_scale(packed.weight_scale, blocks.shape[0], blocks.shape[1]).float()
    return (blocks * scale[..., None] * packed.weight_scale_2.float()).reshape_as(codes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    source = IndexedCheckpoint(args.source, args.source_index)
    carrier = IndexedCheckpoint(args.carrier)
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}"
    results = {}
    source_files: set[str] = set()
    carrier_files: set[str] = set()
    for projection in ("gate_proj", "up_proj", "down_proj"):
        stem = f"{prefix}.{projection}"
        source_name = f"{stem}.weight"
        source_files.add(source.weight_map[source_name])
        values = {}
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            name = f"{stem}.{suffix}"
            carrier_files.add(carrier.weight_map[name])
            values[suffix] = carrier.get(name)
        packed = PackedNVFP4(**values)
        reference = source.get(source_name)
        results[projection] = {
            "shape": list(reference.shape),
            "exact_low_nibble_logical_scale": error_metrics(reference, dequantize(packed, low_first=True)),
            "wrong_high_nibble_first": error_metrics(reference, dequantize(packed, low_first=False)),
            "wrong_kernel_swizzle_in_checkpoint": error_metrics(reference, wrong_swizzled_storage_dequant(packed)),
            "global_scale": float(packed.weight_scale_2),
        }

    payload = {
        "schema": "glm53-nvfp4-v2.carrier-abi-validation.v1",
        "source_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
        "layer": args.layer,
        "expert": args.expert,
        "source_files": [
            {"path": name, "bytes": (args.source / name).stat().st_size, "sha256": sha256_file(args.source / name)}
            for name in sorted(source_files)
        ],
        "carrier_files": [
            {"path": name, "bytes": (args.carrier / name).stat().st_size, "sha256": sha256_file(args.carrier / name)}
            for name in sorted(carrier_files)
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
