"""Fit three tiny checkpoint-family E4M3 scalar codebooks without protected data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import _best_power2_scales, alphabet_levels, sqg_scalar16_codebook


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _project_ordered_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Project centers to distinct ordered finite E4M3 values."""
    raw = values.float().to(torch.float8_e4m3fn).view(torch.uint8)
    decoded = raw.view(torch.float8_e4m3fn).float()
    if bool(torch.isfinite(decoded).all()) and decoded.unique().numel() == 16 and bool((decoded[1:] > decoded[:-1]).all()):
        return raw.contiguous()
    finite = alphabet_levels("e4m3")
    finite_raw = finite.to(torch.float8_e4m3fn).view(torch.uint8)
    chosen: list[int] = []
    previous = -1
    for position, value in enumerate(values.float()):
        distances = (finite - value).abs()
        distances[: previous + 1] = torch.inf
        remaining = 15 - position
        if remaining:
            distances[finite.numel() - remaining :] = torch.inf
        index = int(distances.argmin())
        if not torch.isfinite(distances[index]):
            raise RuntimeError("could not preserve 16 distinct E4M3 codebook levels")
        chosen.append(index)
        previous = index
    return finite_raw[torch.tensor(chosen)].contiguous()


@torch.no_grad()
def fit_projection_codebook(
    source: IndexedCheckpoint,
    names: list[str],
    *,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    table = sqg_scalar16_codebook(device="cpu")
    history = []
    for iteration in range(iterations):
        levels = table.to(device).view(torch.float8_e4m3fn).float()
        sums = torch.zeros(16, dtype=torch.float64, device=device)
        counts = torch.zeros(16, dtype=torch.float64, device=device)
        squared_error = torch.zeros((), dtype=torch.float64, device=device)
        values_seen = 0
        for name in names:
            weight = source.get(name).to(device).float()
            scales = _best_power2_scales(weight, levels, 32)
            normalized = weight.reshape(weight.shape[0], -1, 32) / scales[..., None]
            distance = (normalized[..., None] - levels).abs()
            assignment = distance.argmin(-1)
            flat_assignment = assignment.flatten()
            flat_values = normalized.flatten().double()
            sums.scatter_add_(0, flat_assignment, flat_values)
            counts.scatter_add_(0, flat_assignment, torch.ones_like(flat_values))
            squared_error += (flat_values - levels[flat_assignment].double()).square().sum()
            values_seen += flat_values.numel()
            del weight, scales, normalized, distance, assignment, flat_assignment, flat_values
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), levels.double())
        table = _project_ordered_e4m3(centers.cpu())
        history.append({
            "iteration": iteration,
            "normalized_mse_before_update": float((squared_error / values_seen).item()),
            "values_seen": values_seen,
            "empty_levels": int((counts == 0).sum().item()),
            "codebook": table.view(torch.float8_e4m3fn).float().tolist(),
        })
    return table, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite scalar-law evidence")
    plan = json.loads(args.plan.read_text())
    experts = [int(value) for value in args.experts.split(",")]
    if args.layer != plan["layer"] or experts != plan["fit_experts"] or args.iterations != 2:
        raise ValueError("fit invocation does not match the frozen scalar16 plan")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    prefix = source.expert_prefix(args.layer, experts[0]).split(f"layers.{args.layer}.")[0]
    tensors: dict[str, torch.Tensor] = {}
    histories = {}
    names_by_projection = {}
    started = time.time()
    for projection in PROJECTIONS:
        names = [
            f"{prefix}layers.{args.layer}.mlp.experts.{expert}.{projection}.weight"
            for expert in experts
        ]
        names_by_projection[projection] = names
        table, history = fit_projection_codebook(
            source, names, iterations=args.iterations, device=device
        )
        tensors[f"{projection}_codebook_e4m3"] = table
        histories[projection] = history
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={
        "schema": "glm53-scalar16-projection-codebooks.v1",
        "total_runtime_table_bytes": "48",
        "initialization": "SQG-normal-16",
    })
    receipt = {
        "schema": "glm53-scalar16-projection-codebook-fit.v1",
        "layer": args.layer,
        "experts": experts,
        "iterations": args.iterations,
        "initialization": "SQG-normal quantile centers projected to E4M3",
        "update": "per-projection Lloyd centroids after UE8M0/K32 scale fit, exact ordered E4M3 projection",
        "runtime_table_bytes": 48,
        "names_by_projection": names_by_projection,
        "history": histories,
        "source_index_sha256": sha256_file(args.source_index),
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)},
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
