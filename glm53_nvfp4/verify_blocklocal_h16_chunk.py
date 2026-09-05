"""Verify saved physical block-local H16 payloads against BF16 pseudoquant chunks."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .block_rotation import apply_weight_rotation
from .modelopt import PackedNVFP4, dequantize
from .quantize_blocklocal_h16_layer import descriptor_bank
from .shard_index import sha256_file


WEIGHT_RE = re.compile(
    r"^(.*\.experts\.(\d+)\.(gate_proj|up_proj|down_proj))\.weight$"
)


def verify(
    dense_path: Path,
    physical_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "glm53-rotation-v6.blocklocal-h16-layer3-chunk-receipt.v1":
        raise RuntimeError("unexpected chunk receipt schema")
    expected_outputs = receipt["outputs"]
    for name, path in (("dense", dense_path), ("physical", physical_path)):
        if (
            path.stat().st_size != expected_outputs[name]["bytes"]
            or sha256_file(path) != expected_outputs[name]["sha256"]
        ):
            raise RuntimeError(f"{name} artifact differs from chunk receipt")
    first, last = receipt["expert_range"]
    expected_pairs = {
        (expert, projection)
        for expert in range(first, last)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    bank = torch.stack(
        descriptor_bank(
            256,
            torch.device("cpu"),
            include_identity=bool(receipt.get("include_identity", False)),
        )
    )
    seen: set[tuple[int, str]] = set()
    maximum_abs_bf16_error = 0.0
    with safe_open(str(dense_path), framework="pt", device="cpu") as dense, safe_open(
        str(physical_path), framework="pt", device="cpu"
    ) as physical:
        dense_keys = set(dense.keys())
        physical_keys = set(physical.keys())
        for name in sorted(dense_keys):
            match = WEIGHT_RE.match(name)
            if match is None:
                raise RuntimeError(f"unexpected dense tensor {name}")
            stem, expert_text, projection = match.groups()
            expert = int(expert_text)
            pair = (expert, projection)
            if pair in seen or pair not in expected_pairs:
                raise RuntimeError(f"duplicate or out-of-range dense tensor {name}")
            seen.add(pair)
            required = {
                f"{stem}.weight",
                f"{stem}.weight_scale",
                f"{stem}.weight_scale_2",
                f"{stem}.rotation_index",
            }
            if not required.issubset(physical_keys):
                raise RuntimeError(f"physical payload incomplete for {stem}")
            indexes = physical.get_tensor(f"{stem}.rotation_index").long()
            packed = PackedNVFP4(
                weight=physical.get_tensor(f"{stem}.weight"),
                weight_scale=physical.get_tensor(f"{stem}.weight_scale"),
                weight_scale_2=physical.get_tensor(f"{stem}.weight_scale_2"),
            )
            if indexes.ndim != 1 or indexes.numel() != packed.weight_scale.shape[1]:
                raise RuntimeError(f"descriptor geometry mismatch for {stem}")
            rotation = bank[indexes]
            effective = apply_weight_rotation(
                dequantize(packed), rotation.transpose(-1, -2)
            ).to(torch.bfloat16)
            stored = dense.get_tensor(name)
            if not torch.equal(effective, stored):
                difference = float((effective.float() - stored.float()).abs().max())
                raise RuntimeError(f"dense/physical closure failed for {name}: {difference}")
            maximum_abs_bf16_error = max(
                maximum_abs_bf16_error,
                float((effective.float() - stored.float()).abs().max()),
            )
        expected_physical = {
            suffix
            for name in dense_keys
            for suffix in (
                name,
                name + "_scale",
                name + "_scale_2",
                name.removesuffix(".weight") + ".rotation_index",
            )
        }
        if physical_keys != expected_physical:
            missing = sorted(expected_physical - physical_keys)
            extra = sorted(physical_keys - expected_physical)
            raise RuntimeError(f"physical inventory mismatch missing={missing[:3]} extra={extra[:3]}")
    if seen != expected_pairs:
        raise RuntimeError(f"expert/projection coverage mismatch: {len(seen)} != {len(expected_pairs)}")
    return {
        "schema": "glm53-rotation-v6.blocklocal-h16-chunk-verification.v1",
        "status": "pass",
        "expert_range": [first, last],
        "projection_pairs": len(seen),
        "dense_sha256": sha256_file(dense_path),
        "physical_sha256": sha256_file(physical_path),
        "receipt_sha256": sha256_file(receipt_path),
        "verifier_sha256": sha256_file(Path(__file__)),
        "closure": "bitwise BF16 equality after physical NVFP4 decode and indexed inverse rotation",
        "maximum_abs_bf16_error": maximum_abs_bf16_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = verify(args.dense, args.physical, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
