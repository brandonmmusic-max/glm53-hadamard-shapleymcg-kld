"""Materialize one immutable layer-shared SO(16) butterfly rotation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from .block_rotation import butterfly16, orthogonality_error
from .shard_index import sha256_file


def build(layer: int, angle_pi: float) -> tuple[torch.Tensor, dict]:
    if not 3 <= layer <= 44:
        raise ValueError("layer must be 3..44")
    if not math.isfinite(angle_pi) or abs(angle_pi) > 0.25:
        raise ValueError("angle_pi must be finite and within [-0.25, 0.25]")
    angle = math.pi * angle_pi
    rotation = butterfly16(angle, dtype=torch.float32).contiguous()
    fp32_gram_error = orthogonality_error(rotation)
    bf16 = rotation.to(torch.bfloat16).float()
    bf16_gram_error = orthogonality_error(bf16)
    if fp32_gram_error > 2e-6:
        raise RuntimeError(f"FP32 butterfly is not orthogonal: {fp32_gram_error}")
    return rotation, {
        "layer": layer,
        "kind": "mid",
        "parameterization": "four-stage H16-wired butterfly with one shared Givens angle",
        "angle_pi": angle_pi,
        "angle_radians": angle,
        "stored_parameters": "materialized FP32 16x16 matrix",
        "table_bytes": rotation.numel() * rotation.element_size(),
        "fp32_gram_max_abs": fp32_gram_error,
        "runtime_bf16_gram_max_abs": bf16_gram_error,
        "ldlq": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--angle-pi",
        type=float,
        required=True,
        help="one shared butterfly angle expressed as a multiple of pi",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite rotation or receipt")
    rotation, receipt = build(args.layer, args.angle_pi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer_{args.layer:03d}_mid": rotation},
        str(args.output),
        metadata={
            "schema": "glm53-rotation-v8.shared-butterfly16.v1",
            "layer": str(args.layer),
            "scope": "mid-only",
            "angle_pi": format(args.angle_pi, ".17g"),
        },
    )
    receipt["schema"] = "glm53-rotation-v8.shared-butterfly16-receipt.v1"
    receipt["output"] = {
        "path": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
