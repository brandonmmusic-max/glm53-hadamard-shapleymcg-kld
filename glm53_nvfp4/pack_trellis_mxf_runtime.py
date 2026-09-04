"""Losslessly bridge trellis-E2M3 pseudoquant weights into the B12X MXFP6 ABI."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-input", type=Path, required=True)
    parser.add_argument("--codec-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from b12x.quantization.mxfp6.fp6_checkpoint import (
        dequantize_linear_from_fp6,
        quantize_linear_to_fp6,
    )

    started = time.time()
    with safe_open(str(args.dense_input), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        names = sorted(handle.keys())
    if metadata.get("schema") != "glm53-trellis-mxf-dense-pseudoquant-chunk.v1":
        raise ValueError("dense input is not a trellis-MXF pseudoquant chunk")
    if metadata.get("alphabet") != "e2m3" or metadata.get("block_size") != "32":
        raise ValueError("B12X bridge currently supports only E2M3 block32")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output: dict[str, torch.Tensor] = {}
    mismatches = 0
    max_abs = 0.0
    logical_elements = 0
    with safe_open(str(args.dense_input), framework="pt", device="cpu") as handle:
        for index, name in enumerate(names):
            weight = handle.get_tensor(name).to(device)
            encoded = quantize_linear_to_fp6(
                weight,
                source_format="mxfp6_w6a8",
                use_gpu=True,
                block_scale_rule="mse",
            )
            decoded = dequantize_linear_from_fp6(
                encoded.weight.to(device),
                encoded.weight_scale.to(device),
                fmt="e2m3",
                weight_scale_2=encoded.weight_scale_2.to(device),
            )
            delta = decoded - weight.float()
            mismatches += int((delta != 0).sum().item())
            max_abs = max(max_abs, float(delta.abs().max().item()))
            stem = name[: -len(".weight")]
            output[name] = encoded.weight.contiguous()
            output[f"{stem}.weight_scale"] = encoded.weight_scale.contiguous()
            output[f"{stem}.weight_scale_2"] = encoded.weight_scale_2.reshape(())
            output[f"{stem}.input_scale"] = encoded.input_scale.reshape(())
            logical_elements += weight.numel()
            del weight, encoded, decoded, delta
            if index % 24 == 23:
                print(json.dumps({"packed_tensors": index + 1}), flush=True)
                torch.cuda.empty_cache()
    if mismatches:
        raise RuntimeError(f"MXFP6 bridge was not lossless: mismatches={mismatches} max_abs={max_abs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v3.b12x-mxfp6-layer-chunk.v1",
            "source_format": "mxfp6_w6a8",
            "layer": metadata["layer"],
            "expert_range": metadata["expert_range"],
            "rotation": "identity",
            "rotation_scope": "all",
            "trellis_source_sha256": sha256_file(args.codec_input),
            "bridge_lossless": "true",
        },
    )
    payload_bytes = sum(t.numel() * t.element_size() for t in output.values())
    receipt = {
        "schema": "glm53-trellis-mxf-to-b12x-bridge-receipt.v1",
        "dense_input_sha256": sha256_file(args.dense_input),
        "codec_input_sha256": sha256_file(args.codec_input),
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)},
        "layer": int(metadata["layer"]),
        "expert_range": metadata["expert_range"],
        "logical_elements": logical_elements,
        "payload_bytes": payload_bytes,
        "payload_bpw": 8.0 * payload_bytes / logical_elements,
        "lossless_bridge": True,
        "decoded_mismatches": mismatches,
        "decoded_max_abs": max_abs,
        "runtime_storage_boundary": "B12X bridge stores 6.25 bpw; it validates pseudoquant KLD only and does not prove a 4.25-bpw trellis runtime",
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "payload_bpw": receipt["payload_bpw"], "lossless_bridge": True}, sort_keys=True))


if __name__ == "__main__":
    main()
