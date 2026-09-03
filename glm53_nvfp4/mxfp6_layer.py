"""Encode one GLM routed-expert range as native B12X MXFP6/W6A8 tensors."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .block_rotation import apply_weight_rotation, hadamard16, load_layer_rotation
from .shard_index import IndexedCheckpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=288)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rotation", choices=("identity", "had16", "learned"), required=True
    )
    parser.add_argument("--rotation-file", type=Path)
    parser.add_argument(
        "--rotation-scope", choices=("gate-up", "all"), default="all"
    )
    parser.add_argument("--block-scale-rule", choices=("mse", "ceil"), default="mse")
    args = parser.parse_args()
    if not (3 <= args.layer <= 44 and 0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid layer or expert range")
    if args.rotation == "learned" and args.rotation_file is None:
        raise ValueError("learned rotation requires --rotation-file")

    # This module intentionally imports the codec only after parsing.  The
    # pinned runtime image provides the exact codec used by its W6A8 kernel.
    from b12x.quantization.mxfp6.fp6_checkpoint import quantize_linear_to_fp6

    started = time.time()
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    if args.rotation == "identity":
        rotation_in = rotation_mid = None
    elif args.rotation == "had16":
        rotation_in = hadamard16(device=device)
        rotation_mid = rotation_in if args.rotation_scope == "all" else None
    else:
        rotation_in = load_layer_rotation(
            args.rotation_file, args.layer, kind="in", device=device
        )
        rotation_mid = (
            load_layer_rotation(
                args.rotation_file, args.layer, kind="mid", device=device
            )
            if args.rotation_scope == "all"
            else None
        )

    prefix = checkpoint.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    output: dict[str, torch.Tensor] = {}
    source_files: set[str] = set()
    logical_elements = 0
    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{projection}.weight"
            source_files.add(checkpoint.weight_map[name])
            weight = checkpoint.get(name).to(device)
            rotation = rotation_mid if projection == "down_proj" else rotation_in
            if rotation is not None:
                weight = apply_weight_rotation(weight, rotation)
            encoded = quantize_linear_to_fp6(
                weight,
                source_format="mxfp6_w6a8",
                use_gpu=True,
                block_scale_rule=args.block_scale_rule,
            )
            stem = f"{base}.{projection}"
            output[f"{stem}.weight"] = encoded.weight.contiguous()
            output[f"{stem}.weight_scale"] = encoded.weight_scale.contiguous()
            output[f"{stem}.weight_scale_2"] = encoded.weight_scale_2.reshape(())
            output[f"{stem}.input_scale"] = encoded.input_scale.reshape(())
            logical_elements += int(weight.numel())
            del weight, encoded
        if expert % 8 == 7:
            print(json.dumps({"layer": args.layer, "expert": expert}), flush=True)
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v3.b12x-mxfp6-layer-chunk.v1",
            "source_format": "mxfp6_w6a8",
            "layer": str(args.layer),
            "expert_range": f"{args.expert_start}:{args.expert_end}",
            "rotation": args.rotation,
            "rotation_scope": args.rotation_scope,
        },
    )
    payload_bytes = sum(t.numel() * t.element_size() for t in output.values())
    receipt = {
        "schema": "glm53-nvfp4-v3.mxfp6-layer-chunk-receipt.v1",
        "layer": args.layer,
        "expert_start": args.expert_start,
        "expert_end": args.expert_end,
        "algorithm": {
            "format": "B12X MXFP6 E2M3 weights / E4M3 activations",
            "source_format": "mxfp6_w6a8",
            "group_size": 32,
            "block_scale_rule": args.block_scale_rule,
            "rotation": args.rotation,
            "rotation_scope": args.rotation_scope,
            "rotation_file": str(args.rotation_file) if args.rotation_file else None,
        },
        "source_files": [
            {
                "path": name,
                "bytes": (args.source / name).stat().st_size,
                "sha256": sha256_file(args.source / name),
            }
            for name in sorted(source_files)
        ],
        "logical_elements": logical_elements,
        "payload_bytes": payload_bytes,
        "payload_bpw": 8.0 * payload_bytes / logical_elements,
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensors": len(output),
        },
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "payload_bpw": receipt["payload_bpw"]}))


if __name__ == "__main__":
    main()
