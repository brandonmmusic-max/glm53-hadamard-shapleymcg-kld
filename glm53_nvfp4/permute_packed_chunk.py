"""Apply an exact within-group signed permutation to packed NVFP4 codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .make_rotation_canary import negative_identity16, signed_permutation16
from .modelopt import pack_codes
from .shard_index import sha256_file


def transform(
    tensors: dict, *, layer: int, scope: str, kind: str = "signed-permutation"
) -> tuple[dict, list[str]]:
    rotation = (
        signed_permutation16()
        if kind == "signed-permutation"
        else negative_identity16()
    )
    projections = {"gate_proj", "up_proj"} if scope == "gate-up" else {"down_proj"}
    output = {}
    changed = []
    marker = f"layers.{layer}.mlp.experts."
    for name, tensor in tensors.items():
        projection = next((p for p in projections if f".{p}.weight" in name), None)
        if marker in name and projection is not None and name.endswith(".weight"):
            output[name] = _transform_packed_signed_permutation(tensor, rotation)
            changed.append(name)
        else:
            output[name] = tensor
    return output, changed


def _transform_packed_signed_permutation(
    packed: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """Permute raw FP4 nibbles without erasing the signed-zero code."""
    if packed.ndim != 2 or packed.shape[-1] % 8:
        raise ValueError(f"expected packed [out,in/2] with group-16 K, got {tuple(packed.shape)}")
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    groups = codes.reshape(codes.shape[0], -1, 16)
    source = rotation.abs().argmax(dim=0)
    columns = torch.arange(16)
    if not torch.equal(rotation.abs().sum(dim=0), torch.ones(16)) or not torch.equal(
        rotation.abs().sum(dim=1), torch.ones(16)
    ):
        raise ValueError("rotation is not an exact signed permutation")
    result = groups[..., source].clone()
    negative = rotation[source, columns] < 0
    result[..., negative] ^= 0x08
    return pack_codes(result.reshape_as(codes)).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scope", choices=("gate-up", "mid-only"), required=True)
    parser.add_argument(
        "--kind", choices=("signed-permutation", "negative-identity"),
        default="signed-permutation",
    )
    args = parser.parse_args()
    tensors = load_file(str(args.input), device="cpu")
    output, changed = transform(
        tensors, layer=args.layer, scope=args.scope, kind=args.kind
    )
    if not changed:
        raise RuntimeError("no packed weight tensors matched the requested layer/scope")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v7.exact-packed-permutation-canary.v1",
            "layer": str(args.layer),
            "scope": args.scope,
        },
    )
    payload = {
        "schema": "glm53-nvfp4-v7.exact-packed-permutation-canary-receipt.v1",
        "role": "diagnostic-invariance",
        "layer": args.layer,
        "scope": args.scope,
        "kind": args.kind,
        "changed_tensors": changed,
        "unchanged_scale_tensors": True,
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"changed": len(changed), **payload["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
