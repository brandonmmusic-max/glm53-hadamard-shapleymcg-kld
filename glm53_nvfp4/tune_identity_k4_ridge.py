"""Tune one checkpoint-family ridge for the lossless-K4 down encoder."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import full_gptq_quantize, refine_global_scale
from .block_rotation import apply_activation_rotation, apply_weight_rotation, hadamard16
from .capture import LayerCapture
from .endpoint_codec import optimize_identity_k4_endpoint, pack_identity_k4_stream
from .modelopt import dequantize
from .output_aware import route_weighted_hessian
from .screen_blocklocal_h16 import _full_nmse, _full_quant, _middle
from .shard_index import IndexedCheckpoint, sha256_file


@torch.no_grad()
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    experts = plan["experts"][args.expert_start_index:args.expert_end_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit, evaluation = plan["sampling"]["hessian_fit"], plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(args.capture_root, 3, args.roles, max_samples=fit["count"], sample_offset=fit["offset"], sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, 3, args.roles, max_samples=evaluation["count"], sample_offset=evaluation["offset"], sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(3, experts[0]).split("layers.3.")[0]
    identity, h16 = torch.eye(16, device=device), hadamard16(device=device)
    rows, started = [], time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        base = f"{prefix}layers.3.mlp.experts.{expert}"
        weights = {p: source.get(f"{base}.{p}.weight").to(device).float() for p in ("gate_proj", "up_proj", "down_proj")}
        gs = refine_global_scale(weights["gate_proj"], weights["up_proj"], search_grid=8, iterations=2)
        gate = _full_quant(weights["gate_proj"], fit_hidden, fit_route, identity, gs, 8)
        up = _full_quant(weights["up_proj"], fit_hidden, fit_route, identity, gs, 8)
        fit_mid, eval_mid = _middle(fit_hidden, gate, up), _middle(eval_hidden, gate, up)
        reference = F.linear(_middle(eval_hidden, weights["gate_proj"], weights["up_proj"]), weights["down_proj"])
        transformed = apply_weight_rotation(weights["down_proj"], h16)
        fit_rotated = apply_activation_rotation(fit_mid, h16)
        hessian = route_weighted_hessian(fit_rotated, fit_route)
        down_scale = refine_global_scale(transformed, search_grid=8, iterations=2)
        initial = full_gptq_quantize(transformed, hessian, global_scale=down_scale, search_grid=8)
        controls = dequantize(initial).to(device)
        control_actual = F.linear(apply_activation_rotation(eval_mid, h16), controls)
        control_nmse = _full_nmse(reference, control_actual, eval_route)
        rows.append({"expert": expert, "ridge_ratio": None, "arm": "down-h16-gptq", "evaluation_full_expert_nmse": control_nmse})
        for ridge in plan["candidate"]["ridge_ratio_grid"]:
            endpoint, history = optimize_identity_k4_endpoint(transformed, hessian, initial, sweeps=plan["candidate"]["sweeps"], ridge_ratio=ridge)
            stream = pack_identity_k4_stream(endpoint)
            candidate = dequantize(endpoint).to(device)
            actual = F.linear(apply_activation_rotation(eval_mid, h16), candidate)
            rows.append({"expert": expert, "ridge_ratio": ridge, "arm": "identity-k4-down-h16", "evaluation_full_expert_nmse": _full_nmse(reference, actual, eval_route), "codec": {"trellis_bpw": stream.trellis_bpw, "scale_bpw": stream.scale_bpw, "bit_exact": True, "history": [x.to_dict() for x in history]}})
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        torch.cuda.empty_cache()
    payload = {"schema": "glm53-identity-k4-ridge-tuning-raw.v1", "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "expert_slice": [args.expert_start_index, args.expert_end_index], "rows": rows, "elapsed_seconds": time.time() - started, "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
