"""Measure the W6A8 carrier on real REAP layer inputs, expert by expert.

This is a diagnosis/fit-role tool.  It compares the packed MXFP6 W6A8 kernel
against the exact decoded BF16 weights while holding the routed hidden states,
router weights, activation clamp, and expert identity fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .capture import LayerCapture


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().flatten()
    reference_f = reference.float().flatten()
    error = actual_f - reference_f
    reference_energy = reference_f.double().square().sum().item()
    return {
        "cosine": float(F.cosine_similarity(actual_f, reference_f, dim=0).item()),
        "nmse": float(error.double().square().sum().item() / reference_energy),
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
    }


def _qdq_e4m3_k32(
    values: torch.Tensor, global_scale: float, scale_policy: str
) -> torch.Tensor:
    """Torch mirror of the carrier's E4M3 + K32 UE8M0 quant/dequant."""
    if values.shape[-1] % 32:
        raise ValueError("E4M3 K32 quantization requires width divisible by 32")
    blocks = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    block_max = blocks.abs().amax(dim=-1)
    ratio = block_max * global_scale / 448.0
    exponent = torch.ceil(torch.log2(ratio.clamp(min=2.0**-127))).clamp(-127, 128)
    offsets = (0,) if scale_policy == "amax" else (0, -1, -2)
    if scale_policy not in {"amax", "mse-3"}:
        raise ValueError(f"unknown activation scale policy: {scale_policy}")
    candidates = []
    errors = []
    for offset in offsets:
        scale = torch.exp2((exponent + offset).clamp(-127, 128))
        scale = torch.where(block_max == 0, torch.zeros_like(scale), scale)
        output_scale = torch.where(scale == 0, torch.zeros_like(scale), global_scale / scale)
        quantized = (blocks * output_scale[..., None]).clamp(-448.0, 448.0)
        quantized = quantized.to(torch.float8_e4m3fn).float()
        candidate = quantized * scale[..., None] / global_scale
        candidates.append(candidate)
        errors.append((candidate - blocks).double().square().sum(dim=-1))
    winner = torch.stack(errors).argmin(dim=0)
    reconstructed = candidates[0]
    for index, candidate in enumerate(candidates[1:], start=1):
        reconstructed = torch.where((winner == index)[..., None], candidate, reconstructed)
    return reconstructed.reshape_as(values).to(torch.bfloat16)


def _projection(handle: safe_open, layer: int, expert: int, projection: str) -> dict[str, torch.Tensor]:
    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
    return {
        suffix: handle.get_tensor(f"{stem}.{suffix}").cuda()
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }


def _dense_projection(handle: safe_open, layer: int, expert: int, projection: str) -> torch.Tensor:
    name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    return handle.get_tensor(name).cuda()


def _run_one(
    plan: object,
    native: dict[str, dict[str, torch.Tensor]],
    dense: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    a1_gscale: float,
    a2_gscale: float,
    swiglu_limit: float,
    activation_scale_policy: str,
) -> dict[str, object]:
    from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod
    from b12x.moe import fused_moe

    # v79's loader swaps checkpoint [gate; up] to the kernel's [up; gate].
    w1 = torch.cat((native["up_proj"]["weight"], native["gate_proj"]["weight"]), dim=0)[None].contiguous()
    w1_scale = torch.cat(
        (native["up_proj"]["weight_scale"], native["gate_proj"]["weight_scale"]), dim=0
    )[None].contiguous()
    w2 = native["down_proj"]["weight"][None].contiguous()
    w2_scale = native["down_proj"]["weight_scale"][None].contiguous()
    ones = torch.ones(1, dtype=torch.float32, device="cuda")
    prepared = fused_moe.prepare_weights(
        plan=plan,
        w1_fp4=w1,
        w1_blockscale=w1_scale,
        w1_global_scale=ones,
        a1_gscale=torch.full_like(ones, a1_gscale),
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=ones.clone(),
        a2_gscale=torch.full_like(ones, a2_gscale),
        params_dtype=torch.bfloat16,
    )
    method = B12XFP6MoEMethod(prepared, plan)
    topk_ids = torch.zeros((hidden.shape[0], 1), dtype=torch.int32, device="cuda")
    topk_weights = route_weights[:, None].float()
    fused = method.apply(hidden, topk_weights, topk_ids)

    def exact_path(x: torch.Tensor, quantize_middle: bool) -> torch.Tensor:
        gate = F.linear(x, dense["gate_proj"])
        up = F.linear(x, dense["up_proj"])
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        middle = F.silu(gate.float()).to(torch.bfloat16) * up
        if quantize_middle:
            middle = _qdq_e4m3_k32(middle, a2_gscale, activation_scale_policy)
        output = F.linear(middle, dense["down_proj"])
        return (output.float() * route_weights[:, None]).to(torch.bfloat16)

    reference = exact_path(hidden, False)
    fc1_quantized = exact_path(
        _qdq_e4m3_k32(hidden, a1_gscale, activation_scale_policy), False
    )
    fc2_quantized = exact_path(hidden, True)
    both_quantized = exact_path(
        _qdq_e4m3_k32(hidden, a1_gscale, activation_scale_policy), True
    )
    torch.cuda.synchronize()
    result = _metrics(fused, reference)
    result["ablations"] = {
        "fc1_activation_quant_only_vs_exact": _metrics(fc1_quantized, reference),
        "fc2_activation_quant_only_vs_exact": _metrics(fc2_quantized, reference),
        "both_activation_quant_vs_exact": _metrics(both_quantized, reference),
        "fused_vs_both_activation_quant": _metrics(fused, both_quantized),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", required=True, help="comma-separated expert ids")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--a1-grid", default="1.0")
    parser.add_argument("--a2-grid", default="1.0")
    parser.add_argument("--scale-map", type=Path)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument(
        "--activation-scale-policy", choices=("amax", "mse-3"), default="amax"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    experts = [int(value) for value in args.experts.split(",")]
    a1_grid = [float(value) for value in args.a1_grid.split(",")]
    a2_grid = [float(value) for value in args.a2_grid.split(",")]
    scale_map = json.loads(args.scale_map.read_text()) if args.scale_map else None
    if scale_map is not None and "scales" in scale_map:
        scale_map = scale_map["scales"]
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        max_samples=args.samples,
        sample_offset=args.sample_offset,
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    device = torch.device("cuda")
    from b12x.integration.vllm.fp6_serving import get_fp6_moe_weight_plan

    plan = get_fp6_moe_weight_plan(
        source_format="mxfp6_e2m3",
        activation="silu",
        num_experts=1,
        hidden_size=4096,
        intermediate_size=2048,
    )
    cells: list[dict[str, object]] = []
    with safe_open(str(args.native), framework="pt", device="cpu") as native_handle, safe_open(
        str(args.dense), framework="pt", device="cpu"
    ) as dense_handle:
        for expert in experts:
            hidden_cpu, route_weights_cpu = capture.samples(expert)
            hidden = hidden_cpu.to(device)
            route_weights = route_weights_cpu.to(device)
            native = {
                projection: _projection(native_handle, args.layer, expert, projection)
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            dense = {
                projection: _dense_projection(dense_handle, args.layer, expert, projection)
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            expert_a1 = a1_grid
            expert_a2 = a2_grid
            if scale_map is not None:
                entry = scale_map[str(expert)]
                expert_a1 = [float(entry["a1_gscale"])]
                expert_a2 = [float(entry["a2_gscale"])]
            for a1_gscale in expert_a1:
                for a2_gscale in expert_a2:
                    metrics = _run_one(
                        plan,
                        native,
                        dense,
                        hidden,
                        route_weights,
                        a1_gscale,
                        a2_gscale,
                        args.swiglu_limit,
                        args.activation_scale_policy,
                    )
                    cell = {
                        "expert": expert,
                        "samples": int(hidden.shape[0]),
                        "a1_gscale": a1_gscale,
                        "a2_gscale": a2_gscale,
                        **metrics,
                    }
                    cells.append(cell)
                    print(json.dumps(cell, sort_keys=True), flush=True)
            del native, dense, hidden, route_weights
            torch.cuda.empty_cache()

    payload = {
        "schema": "glm53-trellis-mxf.w6a8-reap-canary.v1",
        "role": "fit",
        "sampling": "domain-balanced routes within each expert",
        "native": str(args.native),
        "dense": str(args.dense),
        "capture_root": str(args.capture_root),
        "roles": str(args.roles),
        "layer": args.layer,
        "experts": experts,
        "samples_per_expert": args.samples,
        "sample_offset": args.sample_offset,
        "swiglu_limit": args.swiglu_limit,
        "activation_scale_policy": args.activation_scale_policy,
        "a1_grid": a1_grid,
        "a2_grid": a2_grid,
        "scale_map": str(args.scale_map) if args.scale_map else None,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
