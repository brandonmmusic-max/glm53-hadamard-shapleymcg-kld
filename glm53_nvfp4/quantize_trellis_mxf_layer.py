"""Materialize a frozen trellis-MXF layer range and dense BF16 pseudoquant overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, pack_ue8m0, quantize_trellis_mxf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--codec-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=(3, 4), default=4)
    parser.add_argument("--alphabet", choices=("e2m1", "e2m3", "e3m2", "e4m3"), default="e2m3")
    parser.add_argument("--block-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--law", choices=("mcg", "sqg-xor-cheb-t12"), default="sqg-xor-cheb-t12")
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--scale-refinement-iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (3 <= args.layer <= 44 and 0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid layer or expert range")

    started = time.time()
    source = IndexedCheckpoint(args.source, args.source_index)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    prefix = source.expert_prefix(args.layer, args.expert_start).split(f"layers.{args.layer}.")[0]
    dense_tensors: dict[str, torch.Tensor] = {}
    codec_tensors: dict[str, torch.Tensor] = {}
    source_files: set[str] = set()
    codebook = None
    logical_elements = 0
    max_bf16_rounding = 0.0
    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{projection}.weight"
            source_files.add(source.weight_map[name])
            weight = source.get(name).to(device).float()
            payload = quantize_trellis_mxf(
                weight,
                bits=args.bits,
                alphabet=args.alphabet,
                law=args.law,
                compander_scale=args.alpha,
                block_size=args.block_size,
                scale_refinement_iterations=args.scale_refinement_iterations,
            )
            scale_codes = pack_ue8m0(payload.scales)
            decoded = decode_trellis_mxf(
                payload.trellis,
                payload.codebook_e4m3,
                scale_codes,
                bits=args.bits,
                block_size=args.block_size,
                rows=weight.shape[0],
                width=weight.shape[1],
                device=device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"codec closure failed for {name}")
            dense = decoded.to(torch.bfloat16).cpu().contiguous()
            max_bf16_rounding = max(
                max_bf16_rounding,
                float((dense.to(device).float() - decoded).abs().max().item()),
            )
            dense_tensors[name] = dense
            codec_tensors[f"{base}.{projection}.trellis"] = payload.trellis
            codec_tensors[f"{base}.{projection}.scale_ue8m0"] = scale_codes
            if codebook is None:
                codebook = payload.codebook_e4m3
                codec_tensors["codec.codebook_e4m3"] = codebook
            elif not torch.equal(codebook, payload.codebook_e4m3):
                raise RuntimeError("state LUT changed within a frozen layer chunk")
            logical_elements += weight.numel()
            del weight, payload, decoded, dense
        if expert % 8 == 7:
            print(json.dumps({"layer": args.layer, "expert": expert}), flush=True)
            torch.cuda.empty_cache()

    metadata = {
        "schema": "glm53-trellis-mxf-dense-pseudoquant-chunk.v1",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "bits": str(args.bits),
        "alphabet": args.alphabet,
        "block_size": str(args.block_size),
        "law": args.law,
        "alpha": str(args.alpha),
    }
    args.dense_output.parent.mkdir(parents=True, exist_ok=True)
    args.codec_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dense_tensors, str(args.dense_output), metadata=metadata)
    save_file(
        codec_tensors,
        str(args.codec_output),
        metadata={**metadata, "schema": "glm53-trellis-mxf-codec-chunk.v1"},
    )
    codebook_sha256 = hashlib.sha256(codebook.numpy().tobytes()).hexdigest()
    receipt = {
        "schema": "glm53-trellis-mxf-layer-chunk-receipt.v1",
        "layer": args.layer,
        "expert_start": args.expert_start,
        "expert_end": args.expert_end,
        "algorithm": {
            "bits": args.bits,
            "alphabet": args.alphabet,
            "block_size": args.block_size,
            "scale_format": "UE8M0",
            "law": args.law,
            "alpha": args.alpha,
            "scale_refinement_iterations": args.scale_refinement_iterations,
            "stored_bpw": args.bits + 8.0 / args.block_size,
            "runtime_status": "pseudoquant only; decode-to-mxf8f6f4 kernel not yet qualified",
        },
        "codebook_sha256": codebook_sha256,
        "source_files": [
            {"path": name, "bytes": (args.source / name).stat().st_size, "sha256": sha256_file(args.source / name)}
            for name in sorted(source_files)
        ],
        "logical_elements": logical_elements,
        "dense_bf16_rounding_max_abs": max_bf16_rounding,
        "dense_output": {"path": str(args.dense_output), "bytes": args.dense_output.stat().st_size, "sha256": sha256_file(args.dense_output)},
        "codec_output": {"path": str(args.codec_output), "bytes": args.codec_output.stat().st_size, "sha256": sha256_file(args.codec_output)},
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "elapsed_seconds": receipt["elapsed_seconds"], "stored_bpw": receipt["algorithm"]["stored_bpw"]}, sort_keys=True))


if __name__ == "__main__":
    main()
