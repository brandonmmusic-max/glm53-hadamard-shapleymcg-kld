"""Measure projection-output error for a frozen MCG alpha grid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .output_aware import output_nmse, route_weighted_hessian
from .screen_hessian_trellis_full_experts import _middle
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import quantize_trellis_mxf_gptq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    experts = [int(value) for value in args.experts.split(",")]
    if not set(experts).issubset(plan["fit_experts"]):
        raise ValueError("expert subset is outside frozen alpha-fit experts")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit_spec, eval_spec = plan["sampling"]["hessian_fit"], plan["sampling"]["alpha_selection"]
    fit_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=fit_spec["count"], sample_offset=fit_spec["offset"], data_role="fit", sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=eval_spec["count"], sample_offset=eval_spec["offset"], data_role="fit", sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    rows = []
    started = time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        fit_carrier = _qdq_e4m3_k32(fit_hidden, 1.0, "amax")
        eval_carrier = _qdq_e4m3_k32(eval_hidden, 1.0, "amax")
        hidden_hessian = route_weighted_hessian(fit_carrier, fit_route)
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {projection: source.get(f"{base}.{projection}.weight").to(device).float() for projection in plan["projections"]}
        fit_middle = _middle(fit_hidden, weights["gate_proj"], weights["up_proj"], quantize_hidden=False, quantize_output=False)
        eval_middle = _middle(eval_hidden, weights["gate_proj"], weights["up_proj"], quantize_hidden=False, quantize_output=False)
        fit_middle_carrier = _qdq_e4m3_k32(fit_middle, 1.0, "amax")
        eval_middle_carrier = _qdq_e4m3_k32(eval_middle, 1.0, "amax")
        for projection in plan["projections"]:
            source_samples, carrier_samples, routes, hessian = (
                (fit_hidden, fit_carrier, fit_route, hidden_hessian)
                if projection != "down_proj"
                else (fit_middle, fit_middle_carrier, fit_route, route_weighted_hessian(fit_middle_carrier, fit_route))
            )
            eval_source, eval_quantized, eval_routes = (
                (eval_hidden, eval_carrier, eval_route)
                if projection != "down_proj"
                else (eval_middle, eval_middle_carrier, eval_route)
            )
            for alpha in plan["alpha_grid"]:
                payload = quantize_trellis_mxf_gptq(weights[projection], hessian, bits=4, alphabet="e4m3", law="mcg", compander_scale=alpha, block_size=32, scale_refinement_iterations=2)
                rows.append({
                    "expert": expert,
                    "projection": projection,
                    "alpha": alpha,
                    "fit_output_nmse": output_nmse(weights[projection], payload.reconstruction, source_samples, carrier_samples, routes),
                    "evaluation_output_nmse": output_nmse(weights[projection], payload.reconstruction, eval_source, eval_quantized, eval_routes),
                    "weight_nmse": float(((payload.reconstruction - weights[projection]).double().square().sum() / weights[projection].double().square().sum()).item()),
                    "physical_bpw": payload.stored_bpw,
                })
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del fit_hidden, fit_route, eval_hidden, eval_route, weights
        torch.cuda.empty_cache()
    result = {"schema": "glm53-p8-mcg-projection-alpha-raw.v1", "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "experts": experts, "rows": rows, "elapsed_seconds": time.time() - started, "protected_roles_opened": [], "algorithm_exclusion": "no LDLQ or BlockLDLQ used"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
