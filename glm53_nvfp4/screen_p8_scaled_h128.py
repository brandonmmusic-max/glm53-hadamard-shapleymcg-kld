"""Fit simple checkpoint-family diagonal laws for the H128 P8 boundary."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .canary_mxfp6_reap import _metrics, _qdq_e4m3_k32
from .capture import LayerCapture
from .p8_h128 import diagonal_h128_reconstruct, hadamard128_last
from .shard_index import IndexedCheckpoint, sha256_file


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_out = F.linear(hidden.float(), gate.float()).clamp(max=10.0)
    up_out = F.linear(hidden.float(), up.float()).clamp(-10.0, 10.0)
    return F.silu(gate_out) * up_out


def _diagonal(law: str, alpha: float, act_rms: torch.Tensor, weight_rms: torch.Tensor, clamp: tuple[float, float]) -> torch.Tensor:
    loga = act_rms.clamp_min(1e-8).log()
    logw = weight_rms.clamp_min(1e-8).log()
    if law == "balance":
        logd = alpha * (logw - loga)
    elif law == "whiten":
        logd = -alpha * loga
    elif law == "sensitivity":
        logd = alpha * logw
    elif law == "smoothquant":
        logd = -alpha * loga + (1.0 - alpha) * logw
    else:
        raise ValueError(law)
    logd = logd.reshape(-1, 128)
    logd = logd - logd.mean(dim=-1, keepdim=True)
    return logd.exp().clamp(*clamp).reshape(-1)


def _evaluate(hidden: torch.Tensor, route: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, diagonal: torch.Tensor) -> dict[str, dict[str, float]]:
    exact_middle = _middle(hidden, gate, up)
    reference = F.linear(exact_middle, down) * route[:, None]
    carrier_middle = _middle(_qdq_e4m3_k32(hidden, 1.0, "amax").float(), gate, up)
    ordinary = F.linear(_qdq_e4m3_k32(carrier_middle, 1.0, "amax").float(), down) * route[:, None]
    reconstructed = diagonal_h128_reconstruct(carrier_middle, diagonal, lambda value: _qdq_e4m3_k32(value, 1.0, "amax"))
    candidate = F.linear(reconstructed, down) * route[:, None]
    closed = hadamard128_last(hadamard128_last(exact_middle * diagonal)) / diagonal
    closure = F.linear(closed, down) * route[:, None]
    return {
        "ordinary_e4m3": _metrics(ordinary, reference),
        "scaled_h128_e4m3": _metrics(candidate, reference),
        "scaled_h128_unquantized_closure": _metrics(closure, reference),
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
    captures = {
        name: LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=spec["count"], sample_offset=spec["offset"], data_role=spec["role"], sampling_strategy=spec["strategy"])
        for name, spec in plan["sampling"].items()
    }
    family = plan["candidate_family"]
    clamp = tuple(float(v) for v in family["clamp"])
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(f"layers.{plan['layer']}.")[0]
    cells = []
    rotations = {}
    started = time.time()
    for expert in plan["experts"]:
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        gate = source.get(f"{base}.gate_proj.weight").to(device).float()
        up = source.get(f"{base}.up_proj.weight").to(device).float()
        down = source.get(f"{base}.down_proj.weight").to(device).float()
        fit_hidden_cpu, fit_route_cpu = captures["selection"].samples(expert)
        fit_hidden = fit_hidden_cpu.to(device).float()
        fit_route = fit_route_cpu.to(device).float()
        fit_reference = F.linear(_middle(fit_hidden, gate, up), down) * fit_route[:, None]
        fit_middle = _middle(_qdq_e4m3_k32(fit_hidden, 1.0, "amax").float(), gate, up)
        act_rms = fit_middle.double().square().mean(dim=0).sqrt().float()
        weight_rms = down.double().square().mean(dim=0).sqrt().float()
        candidates = [("unit", 0.0, torch.ones_like(act_rms))]
        for law in family["laws"]:
            for alpha in family["alpha_grid"]:
                candidates.append((law, float(alpha), _diagonal(law, float(alpha), act_rms, weight_rms, clamp)))
        scored = []
        for law, alpha, diagonal in candidates:
            reconstructed = diagonal_h128_reconstruct(fit_middle, diagonal, lambda value: _qdq_e4m3_k32(value, 1.0, "amax"))
            actual = F.linear(reconstructed, down) * fit_route[:, None]
            sse = float((actual - fit_reference).double().square().sum().item())
            scored.append((sse, law, alpha, diagonal))
        selected_sse, selected_law, selected_alpha, selected_diagonal = min(scored, key=lambda item: item[0])
        val_hidden_cpu, val_route_cpu = captures["validation"].samples(expert)
        metrics = _evaluate(val_hidden_cpu.to(device).float(), val_route_cpu.to(device).float(), gate, up, down, selected_diagonal)
        cell = {
            "expert": expert,
            "selected_law": selected_law,
            "selected_alpha": selected_alpha,
            "selection_sse": selected_sse,
            "diagonal_min": float(selected_diagonal.min().item()),
            "diagonal_max": float(selected_diagonal.max().item()),
            **metrics,
        }
        cells.append(cell)
        rotations[f"expert.{expert}.down_diagonal"] = selected_diagonal.to(torch.float16).cpu().contiguous()
        print(json.dumps(cell, sort_keys=True), flush=True)
        del gate, up, down, fit_hidden, fit_route, fit_reference, fit_middle, act_rms, weight_rms, candidates, scored, selected_diagonal
        torch.cuda.empty_cache()
    args.rotations_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(rotations, str(args.rotations_output), metadata={"schema": "glm53-p8-scaled-h128-rotations.v1", "layer": str(plan["layer"]), "role": "fit", "ldlq": "false"})
    payload = {
        "schema": "glm53-p8-scaled-h128-screen-raw.v1",
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
