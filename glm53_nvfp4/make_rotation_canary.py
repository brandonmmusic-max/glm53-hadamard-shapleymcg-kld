"""Write a deterministic within-group signed-permutation rotation canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .block_rotation import orthogonality_error
from .shard_index import sha256_file


def signed_permutation16() -> torch.Tensor:
    """An exact SO(16) lane permutation with signs and no group movement."""
    permutation = torch.arange(16).reshape(8, 2).flip(1).reshape(-1)
    result = torch.eye(16)[:, permutation]
    # Two sign flips keep determinant +1 and exercise signed-code handling.
    result[:, 0] *= -1
    result[:, 2] *= -1
    return result


def negative_identity16() -> torch.Tensor:
    """Exact sign-only SO(16) canary with no lane movement."""
    return -torch.eye(16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--kind", choices=("signed-permutation", "negative-identity"),
        default="signed-permutation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="write the transpose for an activation-side orientation diagnostic",
    )
    args = parser.parse_args()
    if not 3 <= args.layer <= 44:
        raise ValueError("layer must be 3..44")
    rotation = (
        signed_permutation16()
        if args.kind == "signed-permutation"
        else negative_identity16()
    )
    if args.transpose:
        rotation = rotation.T.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"layer_{args.layer:03d}_in": rotation.contiguous(),
            f"layer_{args.layer:03d}_mid": rotation.clone().contiguous(),
        },
        str(args.output),
        metadata={"schema": "glm53-nvfp4-v7.signed-permutation-canary.v1"},
    )
    payload = {
        "schema": "glm53-nvfp4-v7.signed-permutation-canary-receipt.v1",
        "role": "diagnostic-invariance",
        "layer": args.layer,
        "construction": args.kind + (" transposed" if args.transpose else ""),
        "determinant": float(torch.linalg.det(rotation)),
        "orthogonality_max_abs": orthogonality_error(rotation),
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
