"""Decode MXFP6 layer chunks to BF16 and report source-relative weight NMSE."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .shard_index import IndexedCheckpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from b12x.quantization.mxfp6.fp6_checkpoint import dequantize_linear_from_fp6

    source = IndexedCheckpoint(args.source, args.source_index)
    with safe_open(str(args.input), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        names = set(handle.keys())
        weight_names = sorted(
            name
            for name in names
            if name.endswith(".weight") and ".mlp.experts." in name
        )
        if not weight_names:
            raise ValueError("input contains no MXFP6 expert weights")
        output: dict[str, torch.Tensor] = {}
        totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"error": 0.0, "energy": 0.0}
        )
        for index, name in enumerate(weight_names):
            stem = name[: -len(".weight")]
            required = {
                name,
                f"{stem}.weight_scale",
                f"{stem}.weight_scale_2",
                f"{stem}.input_scale",
            }
            missing = required.difference(names)
            if missing:
                raise ValueError(f"incomplete MXFP6 tensor family: {sorted(missing)}")
            decoded = dequantize_linear_from_fp6(
                handle.get_tensor(name).to(args.device),
                handle.get_tensor(f"{stem}.weight_scale").to(args.device),
                fmt="e2m3",
                weight_scale_2=handle.get_tensor(f"{stem}.weight_scale_2").to(
                    args.device
                ),
            ).to(torch.bfloat16)
            reference = source.get(name).to(args.device).to(torch.bfloat16)
            delta = decoded.float() - reference.float()
            projection = name.rsplit(".", 2)[-2]
            totals[projection]["error"] += float(delta.double().square().sum().item())
            totals[projection]["energy"] += float(
                reference.float().double().square().sum().item()
            )
            output[name] = decoded.cpu().contiguous()
            if index % 24 == 23:
                print(json.dumps({"decoded_tensors": index + 1}), flush=True)
            del decoded, reference, delta

    per_projection = {}
    for projection, values in sorted(totals.items()):
        per_projection[projection] = {
            **values,
            "weight_nmse": values["error"] / values["energy"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        str(args.output),
        metadata={
            "schema": "glm53-mxfp6-decoded-bf16-layer-chunk.v1",
            "layer": metadata.get("layer", "unknown"),
            "expert_range": metadata.get("expert_range", "unknown"),
            "source_mxfp6_sha256": sha256_file(args.input),
        },
    )
    payload = {
        "schema": "glm53-mxfp6-decoded-bf16-layer-chunk-receipt.v1",
        "input": {
            "path": str(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": sha256_file(args.input),
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "source_index_sha256": sha256_file(args.source_index),
        "tensor_count": len(output),
        "per_projection": per_projection,
        "interpretation": "exact decoded MXFP6 weights stored as BF16; activation and epilogue quantization are absent",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "per_projection": per_projection}, sort_keys=True))


if __name__ == "__main__":
    main()
