"""Fit one monotone 4 KiB E4M3 XOR-T12 law for a checkpoint family."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import _best_power2_scales, _prepare_mxf_tiles
from .trellis_nvfp4 import (
    _encode_tiles,
    _sqg_xor_cheb_t12_rank_lut_e4m3,
    sqg_xor_rank_permutation,
)


def _pava(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted increasing isotonic regression on CPU float64 tensors."""
    means: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, mass) in enumerate(zip(values.tolist(), weights.tolist(), strict=True)):
        means.append(value)
        masses.append(mass)
        starts.append(index)
        ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            merged_mass = masses[-2] + masses[-1]
            merged_mean = (
                means[-2] * masses[-2] + means[-1] * masses[-1]
            ) / merged_mass
            means[-2:] = [merged_mean]
            masses[-2:] = [merged_mass]
            ends[-2:] = [ends[-1]]
            starts.pop()
    result = torch.empty_like(values)
    for mean, start, end in zip(means, starts, ends, strict=True):
        result[start:end] = mean
    return result


@torch.no_grad()
def fit_t12(
    weights: list[torch.Tensor],
    *,
    bits: int,
    iterations: int,
    prior_strength: float,
    tailbite_context: int = 128,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    if not weights or iterations <= 0 or prior_strength <= 0:
        raise ValueError("weights, positive iterations, and positive prior are required")
    device = weights[0].device
    table = _sqg_xor_cheb_t12_rank_lut_e4m3().clone()
    state_bucket = (sqg_xor_rank_permutation(bits) >> 4).to(device)
    history = []
    for iteration in range(iterations):
        expanded = table.index_select(0, (sqg_xor_rank_permutation(bits) >> 4)).to(device)
        levels = expanded.view(torch.float8_e4m3fn).float().unique(sorted=True)
        prior = table.view(torch.float8_e4m3fn).float().double()
        sums = prior * prior_strength
        counts = torch.full((4096,), prior_strength, dtype=torch.float64)
        assigned_error = 0.0
        assigned_values = 0
        for weight in weights:
            scales = _best_power2_scales(weight, levels, 32)
            tiles = _prepare_mxf_tiles(weight, scales, 32)
            quantized, states = _encode_tiles(
                tiles, expanded, bits=bits, tailbite_context=tailbite_context
            )
            buckets = state_bucket.index_select(
                0, (states.to(torch.int64).flatten() & 0xFFFF)
            ).cpu()
            source = tiles.flatten().double().cpu()
            sums.scatter_add_(0, buckets, source)
            counts.scatter_add_(0, buckets, torch.ones_like(source))
            assigned_error += float((quantized.double() - tiles.double()).square().sum().item())
            assigned_values += tiles.numel()
        isotonic = _pava(sums / counts, counts)
        updated = isotonic.float().to(torch.float8_e4m3fn).view(torch.uint8)
        updated[(updated & 0x7F) == 0] = 0
        history.append({
            "iteration": iteration,
            "assigned_mse": assigned_error / assigned_values,
            "changed_buckets": int((updated != table).sum().item()),
            "visited_buckets": int((counts > prior_strength).sum().item()),
        })
        table = updated.contiguous()
    return table, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite learned-law evidence")
    experts = [int(value) for value in args.experts.split(",")]
    source = IndexedCheckpoint(args.source, args.source_index)
    prefix = source.expert_prefix(args.layer, experts[0]).split(
        f"layers.{args.layer}."
    )[0]
    started = time.time()
    weights = []
    names = []
    for expert in experts:
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{projection}.weight"
            weights.append(source.get(name).to(args.device).float())
            names.append(name)
    table, history = fit_t12(
        weights,
        bits=4,
        iterations=args.iterations,
        prior_strength=args.prior_strength,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"codebook_t12_e4m3": table},
        str(args.output),
        metadata={
            "schema": "glm53-learned-xor-t12-e4m3.v1",
            "bits": "4",
            "table_bytes": "4096",
            "monotone": "true",
        },
    )
    receipt = {
        "schema": "glm53-learned-xor-t12-e4m3-receipt.v1",
        "layer": args.layer,
        "experts": experts,
        "tensor_names": names,
        "bits": 4,
        "table_bytes": 4096,
        "monotone": True,
        "iterations": args.iterations,
        "prior_strength": args.prior_strength,
        "history": history,
        "source_index_sha256": sha256_file(args.source_index),
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)},
        "elapsed_seconds": time.time() - started,
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
