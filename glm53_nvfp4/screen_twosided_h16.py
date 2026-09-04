"""Evaluate fixed input/output H16 transforms on disjoint fit-only experts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import full_gptq_quantize, refine_global_scale
from .block_rotation import (
    apply_activation_rotation,
    apply_output_rotation,
    apply_weight_rotation,
    hadamard16,
)
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file


ARMS = ("identity", "input-h16", "output-h16", "two-sided-h16")


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_value = F.linear(hidden.float(), gate.float()).clamp(max=10.0)
    up_value = F.linear(hidden.float(), up.float()).clamp(-10.0, 10.0)
    return F.silu(gate_value) * up_value


def _full_nmse(
    reference: torch.Tensor,
    actual: torch.Tensor,
    route_weights: torch.Tensor,
) -> float:
    weights = route_weights.float().square()
    weights /= weights.sum().clamp_min(1e-30)
    numerator = (
        (actual.float() - reference.float()).double().square().mean(1)
        * weights.double()
    ).sum()
    denominator = (
        reference.float().double().square().mean(1) * weights.double()
    ).sum()
    return float((numerator / denominator.clamp_min(1e-30)).item())


def _transform(
    weight: torch.Tensor,
    input_rotation: torch.Tensor,
    output_rotation: torch.Tensor,
) -> torch.Tensor:
    return apply_output_rotation(
        apply_weight_rotation(weight, input_rotation), output_rotation
    )


def _effective_from_packed(
    packed,
    input_rotation: torch.Tensor,
    output_rotation: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    transformed = dequantize(packed).to(device)
    output_recovered = apply_output_rotation(
        transformed, output_rotation.transpose(-1, -2)
    )
    return apply_weight_rotation(
        output_recovered, input_rotation.transpose(-1, -2)
    )


def _quantize_pair(
    gate: torch.Tensor,
    up: torch.Tensor,
    carrier: torch.Tensor,
    route_weights: torch.Tensor,
    input_rotation: torch.Tensor,
    output_rotation: torch.Tensor,
    search_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_transformed = _transform(gate, input_rotation, output_rotation)
    up_transformed = _transform(up, input_rotation, output_rotation)
    scale = refine_global_scale(
        gate_transformed,
        up_transformed,
        search_grid=search_grid,
        iterations=2,
    )
    hessian = route_weighted_hessian(
        apply_activation_rotation(carrier, input_rotation), route_weights
    )
    rows = gate.shape[0]
    packed = full_gptq_quantize(
        torch.cat((gate_transformed, up_transformed)),
        hessian,
        global_scale=scale,
        search_grid=search_grid,
    )
    effective = _effective_from_packed(
        packed, input_rotation, output_rotation, gate.device
    )
    return effective[:rows], effective[rows:]


def _quantize_single(
    weight: torch.Tensor,
    carrier: torch.Tensor,
    route_weights: torch.Tensor,
    input_rotation: torch.Tensor,
    output_rotation: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    transformed = _transform(weight, input_rotation, output_rotation)
    scale = refine_global_scale(
        transformed, search_grid=search_grid, iterations=2
    )
    hessian = route_weighted_hessian(
        apply_activation_rotation(carrier, input_rotation), route_weights
    )
    packed = full_gptq_quantize(
        transformed,
        hessian,
        global_scale=scale,
        search_grid=search_grid,
    )
    return _effective_from_packed(
        packed, input_rotation, output_rotation, weight.device
    )


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
    if not experts:
        raise ValueError("empty expert slice")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit_spec = plan["sampling"]["hessian_fit"]
    eval_spec = plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=fit_spec["count"],
        sample_offset=fit_spec["offset"],
        sampling_strategy="domain-balanced",
    )
    eval_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=eval_spec["count"],
        sample_offset=eval_spec["offset"],
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(
        f"layers.{plan['layer']}."
    )[0]
    identity = torch.eye(16, device=device)
    h16 = hadamard16(device=device)
    arm_rotations = {
        "identity": (identity, identity),
        "input-h16": (h16, identity),
        "output-h16": (identity, h16),
        "two-sided-h16": (h16, h16),
    }
    rows = []
    started = time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden = fit_hidden_cpu.to(device)
        fit_route = fit_route_cpu.to(device)
        eval_hidden = eval_hidden_cpu.to(device)
        eval_route = eval_route_cpu.to(device)
        stem = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {
            projection: source.get(f"{stem}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        reference_middle = _middle(
            eval_hidden, weights["gate_proj"], weights["up_proj"]
        )
        reference_output = F.linear(reference_middle, weights["down_proj"])
        for arm_name in ARMS:
            input_rotation, output_rotation = arm_rotations[arm_name]
            gate, up = _quantize_pair(
                weights["gate_proj"],
                weights["up_proj"],
                fit_hidden,
                fit_route,
                input_rotation,
                output_rotation,
                args.search_grid,
            )
            fit_middle = _middle(fit_hidden, gate, up)
            eval_middle = _middle(eval_hidden, gate, up)
            down = _quantize_single(
                weights["down_proj"],
                fit_middle,
                fit_route,
                input_rotation,
                output_rotation,
                args.search_grid,
            )
            actual_output = F.linear(eval_middle, down)
            rows.append({
                "expert": expert,
                "arm": arm_name,
                "evaluation_full_expert_nmse": _full_nmse(
                    reference_output, actual_output, eval_route
                ),
            })
            del gate, up, down, fit_middle, eval_middle, actual_output
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, reference_middle, reference_output
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-two-sided-h16-screen-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "expert_slice": [args.expert_start_index, args.expert_end_index],
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "sha256": sha256_file(args.output)
    }, sort_keys=True))


if __name__ == "__main__":
    main()
