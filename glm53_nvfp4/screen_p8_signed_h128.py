"""Fit and validate kernel-compatible signed H128 activation bases."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .canary_mxfp6_reap import _metrics, _qdq_e4m3_k32
from .capture import LayerCapture
from .p8_h128 import hadamard128_last
from .shard_index import IndexedCheckpoint, sha256_file


def _candidate_signs(expert: int, block: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    signs = torch.ones(count, 128, dtype=torch.float32)
    for candidate in range(1, count):
        material = f"{seed}:{expert}:{block}:{candidate}".encode()
        local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        signs[candidate] = torch.randint(0, 2, (128,), generator=generator).mul_(2).sub_(1).float()
    return signs.to(device)


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_out = F.linear(hidden.float(), gate.float()).clamp(max=10.0)
    up_out = F.linear(hidden.float(), up.float()).clamp(-10.0, 10.0)
    return F.silu(gate_out) * up_out


def _candidate_block_errors(
    middle_block: torch.Tensor,
    down_block: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    # [C,N,128] -> signed H128 -> E4M3 K32 -> inverse basis -> [C,N,H].
    signed = middle_block[None].float() * signs[:, None, :]
    rotated = hadamard128_last(signed)
    quantized = _qdq_e4m3_k32(rotated, 1.0, "amax").float()
    reconstructed = hadamard128_last(quantized) * signs[:, None, :]
    return F.linear(reconstructed - middle_block[None], down_block.float())


def _fit_signs(
    expert: int,
    middle: torch.Tensor,
    down: torch.Tensor,
    base_error: torch.Tensor,
    route: torch.Tensor,
    candidate_count: int,
    seed: int,
) -> tuple[torch.Tensor, list[int], float]:
    blocks = middle.shape[-1] // 128
    signs_by_block = []
    choices = []
    selected_errors = []
    libraries = []
    # Independent initialization keeps the frozen search deterministic.
    for block in range(blocks):
        sl = slice(block * 128, (block + 1) * 128)
        signs = _candidate_signs(expert, block, candidate_count, seed, middle.device)
        errors = _candidate_block_errors(middle[:, sl], down[:, sl], signs)
        scores = ((errors * route[None, :, None]).double().square().sum(dim=(1, 2)))
        choice = int(scores.argmin().item())
        libraries.append(signs)
        signs_by_block.append(signs[choice].clone())
        choices.append(choice)
        selected_errors.append(errors[choice].clone())
    current = torch.stack(selected_errors).sum(dim=0)
    # Static block order and exactly two coordinate passes are preregistered.
    for _ in range(2):
        for block in range(blocks):
            sl = slice(block * 128, (block + 1) * 128)
            errors = _candidate_block_errors(middle[:, sl], down[:, sl], libraries[block])
            without = base_error + current - selected_errors[block]
            scores = (((without[None] + errors) * route[None, :, None]).double().square().sum(dim=(1, 2)))
            choice = int(scores.argmin().item())
            current = current - selected_errors[block] + errors[choice]
            selected_errors[block] = errors[choice].clone()
            signs_by_block[block] = libraries[block][choice].clone()
            choices[block] = choice
    objective = float((((base_error + current) * route[:, None]).double().square().sum()).item())
    return torch.cat(signs_by_block), choices, objective


def _evaluate(
    hidden: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    signs: torch.Tensor,
) -> dict[str, dict[str, float]]:
    reference_middle = _middle(hidden, gate, up)
    reference = F.linear(reference_middle, down) * route[:, None]
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    middle = _middle(carrier, gate, up)
    ordinary = F.linear(_qdq_e4m3_k32(middle, 1.0, "amax").float(), down) * route[:, None]
    signed = middle * signs
    rotated = hadamard128_last(signed)
    reconstructed = hadamard128_last(_qdq_e4m3_k32(rotated, 1.0, "amax").float()) * signs
    candidate = F.linear(reconstructed, down) * route[:, None]
    closed = hadamard128_last(hadamard128_last(reference_middle * signs)) * signs
    closure = F.linear(closed, down) * route[:, None]
    return {
        "ordinary_e4m3": _metrics(ordinary, reference),
        "signed_h128_e4m3": _metrics(candidate, reference),
        "signed_h128_unquantized_closure": _metrics(closure, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotations-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.output, args.rotations_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    plan = json.loads(args.plan.read_text())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    captures = {}
    for role, spec in plan["sampling"].items():
        captures[role] = LayerCapture(
            args.capture_root, plan["layer"], args.roles,
            max_samples=spec["count"], sample_offset=spec["offset"],
            data_role=spec["role"], sampling_strategy=spec["strategy"],
        )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(f"layers.{plan['layer']}.")[0]
    cells = []
    rotation_tensors = {}
    family = plan["candidate_family"]
    started = time.time()
    for expert in plan["experts"]:
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        gate = source.get(f"{base}.gate_proj.weight").to(device).float()
        up = source.get(f"{base}.up_proj.weight").to(device).float()
        down = source.get(f"{base}.down_proj.weight").to(device).float()
        fit_hidden_cpu, fit_route_cpu = captures["selection"].samples(expert)
        fit_hidden = fit_hidden_cpu.to(device).float()
        fit_route = fit_route_cpu.to(device).float()
        fit_reference = F.linear(_middle(fit_hidden, gate, up), down)
        fit_carrier = _qdq_e4m3_k32(fit_hidden, 1.0, "amax").float()
        fit_middle = _middle(fit_carrier, gate, up)
        base_error = F.linear(fit_middle, down) - fit_reference
        signs, choices, fit_objective = _fit_signs(
            expert, fit_middle, down, base_error, fit_route,
            family["candidates_per_block"], family["seed"],
        )
        val_hidden_cpu, val_route_cpu = captures["validation"].samples(expert)
        val_hidden = val_hidden_cpu.to(device).float()
        val_route = val_route_cpu.to(device).float()
        metrics = _evaluate(val_hidden, val_route, gate, up, down, signs)
        cell = {
            "expert": expert,
            "selection_objective_sse": fit_objective,
            "selected_candidate_by_block": choices,
            "validation_samples": int(val_hidden.shape[0]),
            **metrics,
        }
        cells.append(cell)
        rotation_tensors[f"expert.{expert}.down_sign"] = signs.to(torch.float16).cpu().contiguous()
        print(json.dumps(cell, sort_keys=True), flush=True)
        del gate, up, down, fit_hidden, fit_route, fit_reference, fit_carrier, fit_middle, base_error, signs, val_hidden, val_route
        torch.cuda.empty_cache()
    args.rotations_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(rotation_tensors, str(args.rotations_output), metadata={
        "schema": "glm53-p8-signed-h128-rotations.v1",
        "layer": str(plan["layer"]),
        "basis": "normalized-sylvester-h128",
        "role": "fit",
        "ldlq": "false",
    })
    payload = {
        "schema": "glm53-p8-signed-h128-screen-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "rotations": {"path": str(args.rotations_output), "sha256": sha256_file(args.rotations_output), "bytes": args.rotations_output.stat().st_size},
        "cells": cells,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
