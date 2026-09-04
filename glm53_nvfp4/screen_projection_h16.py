"""Evaluate which routed-expert projection family benefits from input H16."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_rotation import hadamard16
from .capture import LayerCapture
from .screen_twosided_h16 import _full_nmse, _middle, _quantize_pair, _quantize_single
from .shard_index import IndexedCheckpoint, sha256_file


ARMS = ("identity", "gate-up-h16", "down-h16", "all-h16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--expert-start-index", type=int, required=True)
    parser.add_argument("--expert-end-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-grid", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    experts = plan["experts"][args.expert_start_index : args.expert_end_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit, evaluation = plan["sampling"]["hessian_fit"], plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=fit["count"], sample_offset=fit["offset"], sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=evaluation["count"], sample_offset=evaluation["offset"], sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    identity, h16 = torch.eye(16, device=device), hadamard16(device=device)
    rotations = {"identity": (identity, identity), "gate-up-h16": (h16, identity), "down-h16": (identity, h16), "all-h16": (h16, h16)}
    rows, started = [], time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        stem = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {name: source.get(f"{stem}.{name}.weight").to(device).float() for name in ("gate_proj", "up_proj", "down_proj")}
        reference_middle = _middle(eval_hidden, weights["gate_proj"], weights["up_proj"])
        reference_output = F.linear(reference_middle, weights["down_proj"])
        for arm in ARMS:
            gate_rotation, down_rotation = rotations[arm]
            gate, up = _quantize_pair(weights["gate_proj"], weights["up_proj"], fit_hidden, fit_route, gate_rotation, identity, args.search_grid)
            fit_middle, eval_middle = _middle(fit_hidden, gate, up), _middle(eval_hidden, gate, up)
            down = _quantize_single(weights["down_proj"], fit_middle, fit_route, down_rotation, identity, args.search_grid)
            actual = F.linear(eval_middle, down)
            rows.append({"expert": expert, "arm": arm, "evaluation_full_expert_nmse": _full_nmse(reference_output, actual, eval_route)})
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, reference_middle, reference_output
        torch.cuda.empty_cache()
    payload = {"schema": "glm53-projection-h16-attribution-raw.v1", "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "expert_slice": [args.expert_start_index, args.expert_end_index], "rows": rows, "elapsed_seconds": time.time() - started, "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
