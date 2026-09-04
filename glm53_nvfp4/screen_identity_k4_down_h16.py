"""Screen the lossless K4 endpoint optimizer with down-only H16."""
from __future__ import annotations

import argparse
import json
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
from .screen_p4_down_h16 import _weight_nmse
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
    parser.add_argument("--search-grid", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan["candidate"].get("ldlq") is not False:
        raise ValueError("plan must explicitly disable LDLQ")
    experts = plan["experts"][args.expert_start_index:args.expert_end_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit_spec, eval_spec = plan["sampling"]["hessian_fit"], plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=fit_spec["count"], sample_offset=fit_spec["offset"], sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=eval_spec["count"], sample_offset=eval_spec["offset"], sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    identity, h16 = torch.eye(16, device=device), hadamard16(device=device)
    rows = []
    started = time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        stem = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {name: source.get(f"{stem}.{name}.weight").to(device).float() for name in ("gate_proj", "up_proj", "down_proj")}
        shared_scale = refine_global_scale(weights["gate_proj"], weights["up_proj"], search_grid=args.search_grid, iterations=2)
        gate = _full_quant(weights["gate_proj"], fit_hidden, fit_route, identity, shared_scale, args.search_grid)
        up = _full_quant(weights["up_proj"], fit_hidden, fit_route, identity, shared_scale, args.search_grid)
        fit_middle, eval_middle = _middle(fit_hidden, gate, up), _middle(eval_hidden, gate, up)
        reference_output = F.linear(_middle(eval_hidden, weights["gate_proj"], weights["up_proj"]), weights["down_proj"])
        arms, histories = {}, {}
        for arm, rotation in (("identity-gptq", identity), ("down-h16-gptq", h16)):
            transformed = apply_weight_rotation(weights["down_proj"], rotation)
            carrier = apply_activation_rotation(fit_middle, rotation)
            scale = refine_global_scale(transformed, search_grid=args.search_grid, iterations=2)
            packed = full_gptq_quantize(transformed, route_weighted_hessian(carrier, fit_route), global_scale=scale, search_grid=args.search_grid)
            arms[arm] = dequantize(packed).to(device)
            if arm == "down-h16-gptq":
                optimized, history = optimize_identity_k4_endpoint(
                    transformed, route_weighted_hessian(carrier, fit_route), packed,
                    sweeps=plan["candidate"]["sweeps"],
                    ridge_ratio=plan["candidate"].get("ridge_ratio", 0.0),
                )
                stream = pack_identity_k4_stream(optimized)
                arms["identity-k4-down-h16"] = dequantize(optimized).to(device)
                histories["identity-k4-down-h16"] = [item.to_dict() for item in history]
        for arm, rotation in (("identity-gptq", identity), ("down-h16-gptq", h16), ("identity-k4-down-h16", h16)):
            actual = F.linear(apply_activation_rotation(eval_middle, rotation), arms[arm])
            effective = apply_weight_rotation(arms[arm], rotation.transpose(-1, -2))
            rows.append({
                "expert": expert,
                "arm": arm,
                "down_weight_nmse": _weight_nmse(weights["down_proj"], effective),
                "evaluation_full_expert_nmse": _full_nmse(reference_output, actual, eval_route),
                "stored_bpw": 4.5,
                "codec_receipt": ({"trellis_bpw": stream.trellis_bpw, "scale_bpw": stream.scale_bpw, "codebook_law": stream.codebook_law, "decode_endpoint_bit_exact": True, "history": histories[arm]} if arm == "identity-k4-down-h16" else None),
            })
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, gate, up, fit_middle, eval_middle, reference_output, arms
        torch.cuda.empty_cache()
    payload = {"schema": "glm53-identity-k4-down-h16-screen-raw.v1", "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "expert_slice": [args.expert_start_index, args.expert_end_index], "rows": rows, "elapsed_seconds": time.time() - started, "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
