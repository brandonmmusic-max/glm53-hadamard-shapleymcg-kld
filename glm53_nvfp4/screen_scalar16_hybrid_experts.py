"""Causal P8 screen of MCG, learned scalar16, their fit-selected hybrid, and GPTQ."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .block_gptq import full_gptq_quantize
from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import route_weighted_hessian
from .screen_hessian_trellis_full_experts import _full_nmse, _middle, _trellis
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import quantize_scalar16_mxf_gptq, validate_scalar16_codebook


ARMS = ("mcg", "scalar16")


def _scalar(weight: torch.Tensor, hessian: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return quantize_scalar16_mxf_gptq(
        weight,
        hessian,
        table,
        block_size=32,
        scale_refinement_iterations=2,
    ).reconstruction


def _gptq(weight: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return dequantize(full_gptq_quantize(weight, hessian, group_size=16)).to(weight.device)


def _weight_nmse(source: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(((candidate - source).double().square().sum() / source.double().square().sum().clamp_min(1e-30)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--experts", help="optional comma-separated plan subset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    experts = plan["experts"] if args.experts is None else [int(value) for value in args.experts.split(",")]
    if not set(experts).issubset(plan["experts"]):
        raise ValueError("expert subset is outside the frozen plan")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    tables_raw = load_file(str(args.codebooks), device="cpu")
    expected = {f"{projection}_codebook_e4m3" for projection in ("gate_proj", "up_proj", "down_proj")}
    if set(tables_raw) != expected:
        raise ValueError("scalar16 bundle must contain exactly three projection codebooks")
    tables = {projection: validate_scalar16_codebook(tables_raw[f"{projection}_codebook_e4m3"]).to(device) for projection in ("gate_proj", "up_proj", "down_proj")}
    fit_spec = plan["sampling"]["hessian_and_policy_fit"]
    eval_spec = plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=fit_spec["count"], sample_offset=fit_spec["offset"], data_role="fit", sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=eval_spec["count"], sample_offset=eval_spec["offset"], data_role="fit", sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    rows = []
    policies = []
    started = time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        fit_carrier = _qdq_e4m3_k32(fit_hidden, 1.0, "amax")
        hidden_hessian = route_weighted_hessian(fit_carrier, fit_route)
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {projection: source.get(f"{base}.{projection}.weight").to(device).float() for projection in ("gate_proj", "up_proj", "down_proj")}
        fc1 = {}
        for arm in ARMS:
            encoder = _trellis if arm == "mcg" else _scalar
            fc1[arm] = {
                projection: encoder(weights[projection], hidden_hessian, tables[projection]) if arm == "scalar16" else encoder(weights[projection], hidden_hessian)
                for projection in ("gate_proj", "up_proj")
            }
        reference_fit_middle = _middle(fit_hidden, weights["gate_proj"], weights["up_proj"], quantize_hidden=False, quantize_output=False)
        reference_eval_middle = _middle(eval_hidden, weights["gate_proj"], weights["up_proj"], quantize_hidden=False, quantize_output=False)
        combo_middle = {}
        combo_fit_nmse = {}
        for gate_arm, up_arm in itertools.product(ARMS, repeat=2):
            key = f"gate={gate_arm},up={up_arm}"
            combo_middle[key] = _middle(fit_hidden, fc1[gate_arm]["gate_proj"], fc1[up_arm]["up_proj"], quantize_hidden=True, quantize_output=True)
            combo_fit_nmse[key] = _full_nmse(reference_fit_middle, combo_middle[key], fit_route)
        selected_fc1 = min(combo_fit_nmse, key=combo_fit_nmse.get)
        gate_arm = selected_fc1.split(",")[0].split("=")[1]
        up_arm = selected_fc1.split(",")[1].split("=")[1]
        variants = {}
        for variant, chosen_gate, chosen_up in (
            ("mcg", "mcg", "mcg"),
            ("scalar16", "scalar16", "scalar16"),
            ("hybrid", gate_arm, up_arm),
        ):
            fit_middle = combo_middle[f"gate={chosen_gate},up={chosen_up}"]
            down_hessian = route_weighted_hessian(fit_middle, fit_route)
            if variant == "mcg":
                qdown = _trellis(weights["down_proj"], down_hessian)
                down_arm = "mcg"
            elif variant == "scalar16":
                qdown = _scalar(weights["down_proj"], down_hessian, tables["down_proj"])
                down_arm = "scalar16"
            else:
                candidates = {
                    "mcg": _trellis(weights["down_proj"], down_hessian),
                    "scalar16": _scalar(weights["down_proj"], down_hessian, tables["down_proj"]),
                }
                fit_reference = F.linear(reference_fit_middle.float(), weights["down_proj"].float())
                down_fit_nmse = {arm: _full_nmse(fit_reference, F.linear(fit_middle.float(), value.float()), fit_route) for arm, value in candidates.items()}
                down_arm = min(down_fit_nmse, key=down_fit_nmse.get)
                qdown = candidates[down_arm]
            variants[variant] = {"gate_proj": fc1[chosen_gate]["gate_proj"], "up_proj": fc1[chosen_up]["up_proj"], "down_proj": qdown, "policy": {"gate_proj": chosen_gate, "up_proj": chosen_up, "down_proj": down_arm}}
        gptq_gate = _gptq(weights["gate_proj"], hidden_hessian)
        gptq_up = _gptq(weights["up_proj"], hidden_hessian)
        gptq_fit_middle = _middle(fit_hidden, gptq_gate, gptq_up, quantize_hidden=True, quantize_output=True)
        gptq_down = _gptq(weights["down_proj"], route_weighted_hessian(gptq_fit_middle, fit_route))
        variants["gptq-nvfp4"] = {"gate_proj": gptq_gate, "up_proj": gptq_up, "down_proj": gptq_down, "policy": {"gate_proj": "gptq", "up_proj": "gptq", "down_proj": "gptq"}}
        fit_reference = F.linear(reference_fit_middle.float(), weights["down_proj"].float())
        eval_reference = F.linear(reference_eval_middle.float(), weights["down_proj"].float())
        for variant, quantized in variants.items():
            fit_middle = _middle(fit_hidden, quantized["gate_proj"], quantized["up_proj"], quantize_hidden=True, quantize_output=True)
            eval_middle = _middle(eval_hidden, quantized["gate_proj"], quantized["up_proj"], quantize_hidden=True, quantize_output=True)
            rows.append({
                "expert": expert,
                "variant": variant,
                "physical_bpw": 4.5 if variant == "gptq-nvfp4" else 4.25,
                "charged_bpw": 4.5,
                "policy": quantized["policy"],
                "full_expert_fit_nmse": _full_nmse(fit_reference, F.linear(fit_middle.float(), quantized["down_proj"].float()), fit_route),
                "full_expert_evaluation_nmse": _full_nmse(eval_reference, F.linear(eval_middle.float(), quantized["down_proj"].float()), eval_route),
                "projection_weight_nmse": {projection: _weight_nmse(weights[projection], quantized[projection]) for projection in weights},
            })
        policies.append({"expert": expert, "selected_fc1": selected_fc1, "hybrid": variants["hybrid"]["policy"], "fc1_fit_nmse": combo_fit_nmse})
        print(json.dumps({"expert": expert, "hybrid_policy": variants["hybrid"]["policy"]}), flush=True)
        del fit_hidden, fit_route, eval_hidden, eval_route, weights, fc1, combo_middle, variants
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-p8-scalar16-mcg-hybrid-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "experts": experts,
        "rows": rows,
        "policies": policies,
        "codebooks": {"path": str(args.codebooks), "sha256": sha256_file(args.codebooks)},
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
