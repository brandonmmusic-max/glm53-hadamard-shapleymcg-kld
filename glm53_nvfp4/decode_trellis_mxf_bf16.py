"""Decode one frozen P8 trellis chunk to its exact BF16 pseudoquant weights."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .shard_index import sha256_file
from .trellis_mxf import decode_trellis_mxf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    started = time.time()
    decoded: dict[str, torch.Tensor] = {}
    with safe_open(args.codec, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        bits = int(metadata["bits"])
        block_size = int(metadata["block_size"])
        codebook = source.get_tensor("codec.codebook_e4m3")
        trellis_names = sorted(name for name in source.keys() if name.endswith(".trellis"))
        if not trellis_names:
            raise RuntimeError("codec contains no trellis tensors")
        for index, trellis_name in enumerate(trellis_names, start=1):
            prefix = trellis_name.removesuffix(".trellis")
            trellis = source.get_tensor(trellis_name)
            scales = source.get_tensor(f"{prefix}.scale_ue8m0")
            width = trellis.shape[0] * 16
            rows = trellis.shape[1] * 16
            weight = decode_trellis_mxf(
                trellis,
                codebook,
                scales,
                bits=bits,
                block_size=block_size,
                rows=rows,
                width=width,
                device=args.device,
            )
            decoded[f"{prefix}.weight"] = weight.to(torch.bfloat16).cpu().contiguous()
            if index % 24 == 0:
                print(json.dumps({"decoded_tensors": index}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(decoded, str(args.output), metadata=metadata)
    receipt = {
        "schema": "glm53-p8.trellis-mxf-bf16-decode.v1",
        "codec": str(args.codec.resolve()),
        "codec_sha256": sha256_file(args.codec),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "weight_tensors": len(decoded),
        "decode": "bit-exact procedural-MCG trellis plus E4M3 codebook and UE8M0/32 scales, then BF16 round",
        "elapsed_seconds": time.time() - started,
        "ldlq": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
