"""Measure fixed H128 at the P8 E4M3 activation boundary on REAP routes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .canary_mxfp6_reap import _metrics, _qdq_e4m3_k32
from .capture import LayerCapture
from .p8_h128 import transform_uncoupled_weights, uncoupled_boundary
from .shard_index import IndexedCheckpoint, sha256_file


def _ordinary(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, *, quantized: bool) -> torch.Tensor:
    x = _qdq_e4m3_k32(hidden, 1.0, "amax") if quantized else hidden.float()
    gate_out = F.linear(x.float(), gate.float()).clamp(max=10.0)
    up_out = F.linear(x.float(), up.float()).clamp(-10.0, 10.0)
    middle = F.silu(gate_out) * up_out
    if quantized:
        middle = _qdq_e4m3_k32(middle, 1.0, "amax")
    return F.linear(middle.float(), down.float())


def _rotated(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, *, quantized: bool) -> torch.Tensor:
    gate_t, up_t, down_t = transform_uncoupled_weights(gate, up, down)
    x = _qdq_e4m3_k32(hidden, 1.0, "amax") if quantized else hidden.float()
    gate_out = F.linear(x.float(), gate_t)
    up_out = F.linear(x.float(), up_t)
    middle = uncoupled_boundary(gate_out, up_out)
    if quantized:
        middle = _qdq_e4m3_k32(middle, 1.0, "amax")
    return F.linear(middle.float(), down_t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    sampling = plan["sampling"]
    capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=sampling["count"],
        sample_offset=sampling["offset"],
        data_role=sampling["role"],
        sampling_strategy=sampling["strategy"],
    )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(f"layers.{plan['layer']}.")[0]
    cells = []
    started = time.time()
    for expert in plan["experts"]:
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device).float()
        route = route_cpu.to(device).float()
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        gate = source.get(f"{base}.gate_proj.weight").to(device).float()
        up = source.get(f"{base}.up_proj.weight").to(device).float()
        down = source.get(f"{base}.down_proj.weight").to(device).float()
        reference = _ordinary(hidden, gate, up, down, quantized=False) * route[:, None]
        ordinary = _ordinary(hidden, gate, up, down, quantized=True) * route[:, None]
        rotated_exact = _rotated(hidden, gate, up, down, quantized=False) * route[:, None]
        rotated_quant = _rotated(hidden, gate, up, down, quantized=True) * route[:, None]
        cell = {
            "expert": expert,
            "samples": int(hidden.shape[0]),
            "ordinary_e4m3": _metrics(ordinary, reference),
            "fixed_h128_e4m3": _metrics(rotated_quant, reference),
            "fixed_h128_unquantized_closure": _metrics(rotated_exact, reference),
        }
        cells.append(cell)
        print(json.dumps(cell, sort_keys=True), flush=True)
        del hidden, route, gate, up, down, reference, ordinary, rotated_exact, rotated_quant
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-p8-h128-activation-screen-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "cells": cells,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
