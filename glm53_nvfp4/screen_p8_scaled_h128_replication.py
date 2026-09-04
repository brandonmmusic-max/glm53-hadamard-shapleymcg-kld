"""Execute powered, disjoint replication of fixed scaled-H128 family laws."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .canary_mxfp6_reap import _metrics, _qdq_e4m3_k32
from .capture import LayerCapture
from .p8_h128 import diagonal_h128_reconstruct, hadamard128_last
from .screen_p8_scaled_h128 import _diagonal, _middle
from .shard_index import IndexedCheckpoint, sha256_file


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
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(f"layers.{plan['layer']}.")[0]
    rotations = {}
    cells = []
    started = time.time()
    for expert in plan["experts"]:
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        gate = source.get(f"{base}.gate_proj.weight").to(device).float()
        up = source.get(f"{base}.up_proj.weight").to(device).float()
        down = source.get(f"{base}.down_proj.weight").to(device).float()
        cal_hidden_cpu, _ = captures["calibration"].samples(expert)
        cal_middle = _middle(_qdq_e4m3_k32(cal_hidden_cpu.to(device).float(), 1.0, "amax").float(), gate, up)
        act_rms = cal_middle.double().square().mean(0).sqrt().float()
        weight_rms = down.double().square().mean(0).sqrt().float()
        diagonals = {
            "balance-0.5": _diagonal("balance", 0.5, act_rms, weight_rms, (0.25, 4.0)),
            "whiten-0.5": _diagonal("whiten", 0.5, act_rms, weight_rms, (0.25, 4.0)),
        }
        eval_hidden_cpu, eval_route_cpu = captures["evaluation"].samples(expert)
        hidden = eval_hidden_cpu.to(device).float()
        route = eval_route_cpu.to(device).float()
        exact_middle = _middle(hidden, gate, up)
        reference = F.linear(exact_middle, down) * route[:, None]
        carrier_middle = _middle(_qdq_e4m3_k32(hidden, 1.0, "amax").float(), gate, up)
        ordinary = F.linear(_qdq_e4m3_k32(carrier_middle, 1.0, "amax").float(), down) * route[:, None]
        arms = {}
        for name, diagonal in diagonals.items():
            reconstructed = diagonal_h128_reconstruct(carrier_middle, diagonal, lambda value: _qdq_e4m3_k32(value, 1.0, "amax"))
            actual = F.linear(reconstructed, down) * route[:, None]
            closed = hadamard128_last(hadamard128_last(exact_middle * diagonal)) / diagonal
            closure = F.linear(closed, down) * route[:, None]
            arms[name] = {"metrics": _metrics(actual, reference), "closure": _metrics(closure, reference)}
            rotations[f"{name}.expert.{expert}.down_diagonal"] = diagonal.to(torch.float16).cpu().contiguous()
        cell = {"expert": expert, "samples": int(hidden.shape[0]), "ordinary_e4m3": _metrics(ordinary, reference), "arms": arms}
        cells.append(cell)
        print(json.dumps(cell, sort_keys=True), flush=True)
        del gate, up, down, cal_middle, act_rms, weight_rms, diagonals, hidden, route, exact_middle, reference, carrier_middle, ordinary
        torch.cuda.empty_cache()
    args.rotations_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(rotations, str(args.rotations_output), metadata={"schema": "glm53-p8-scaled-h128-replication-rotations.v1", "layer": str(plan["layer"]), "role": "fit", "ldlq": "false"})
    payload = {
        "schema": "glm53-p8-scaled-h128-replication-raw.v1",
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
