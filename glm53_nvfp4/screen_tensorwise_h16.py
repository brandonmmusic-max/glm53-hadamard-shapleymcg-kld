"""Fit and evaluate independent signed-H16 transforms per expert tensor."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import (
    block_hessian,
    full_gptq_quantize,
    gptq_quantize,
    refine_global_scale,
)
from .block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
    hadamard16,
    signed_hadamard16,
)
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import output_nmse, route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file


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


def _effective_from_packed(
    packed,
    rotation: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rotated = dequantize(packed).to(device)
    return apply_weight_rotation(rotated, rotation.transpose(-1, -2))


def _block_candidate(
    weight: torch.Tensor,
    source_samples: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor,
    rotation: torch.Tensor,
    search_grid: int,
) -> tuple[torch.Tensor, float, float]:
    rotated_weight = apply_weight_rotation(weight, rotation)
    rotated_carrier = apply_activation_rotation(carrier_samples, rotation)
    hessian = block_hessian(rotated_carrier, route_weights)
    scale = refine_global_scale(
        rotated_weight, search_grid=search_grid, iterations=2
    )
    packed = gptq_quantize(
        rotated_weight,
        hessian,
        global_scale=scale,
        search_grid=search_grid,
    )
    effective = _effective_from_packed(packed, rotation, weight.device)
    score = output_nmse(
        weight,
        effective,
        source_samples,
        carrier_samples,
        route_weights,
    )
    weight_nmse = float(
        (
            (effective.float() - weight.float()).double().square().sum()
            / weight.float().double().square().sum().clamp_min(1e-30)
        ).item()
    )
    return effective, score, weight_nmse


def _select_rotation(
    weight: torch.Tensor,
    source_samples: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor,
    rotations: list[tuple[str, torch.Tensor]],
    search_grid: int,
    selection_metric: str,
    minimum_margin_vs_fixed: float,
) -> tuple[str, torch.Tensor, list[dict]]:
    if selection_metric not in {"fit-output-nmse", "weight-nmse"}:
        raise ValueError(f"unknown selection metric {selection_metric}")
    rows = []
    for name, rotation in rotations:
        _, output_score, weight_score = _block_candidate(
            weight,
            source_samples,
            carrier_samples,
            route_weights,
            rotation,
            search_grid,
        )
        rows.append({
            "variant": name,
            "fit_output_nmse": output_score,
            "weight_nmse": weight_score,
        })
    score_key = "fit_output_nmse" if selection_metric == "fit-output-nmse" else "weight_nmse"
    winner = min(rows, key=lambda row: (row[score_key], row["variant"]))
    fixed = next(row for row in rows if row["variant"] == "h16-fixed")
    relative_gain = 1.0 - winner[score_key] / fixed[score_key]
    if relative_gain < minimum_margin_vs_fixed:
        winner = fixed
    return winner["variant"], dict(rotations)[winner["variant"]], rows


def _full_effective(
    weight: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor,
    rotation: torch.Tensor,
    global_scale: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    rotated_weight = apply_weight_rotation(weight, rotation)
    rotated_carrier = apply_activation_rotation(carrier_samples, rotation)
    hessian = route_weighted_hessian(rotated_carrier, route_weights)
    packed = full_gptq_quantize(
        rotated_weight,
        hessian,
        global_scale=global_scale,
        search_grid=search_grid,
    )
    return _effective_from_packed(packed, rotation, weight.device)


def _full_pair(
    gate: torch.Tensor,
    up: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor,
    gate_rotation: torch.Tensor,
    up_rotation: torch.Tensor,
    search_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_rotated = apply_weight_rotation(gate, gate_rotation)
    up_rotated = apply_weight_rotation(up, up_rotation)
    scale = refine_global_scale(
        gate_rotated,
        up_rotated,
        search_grid=search_grid,
        iterations=2,
    )
    if torch.equal(gate_rotation, up_rotation):
        carrier_rotated = apply_activation_rotation(carrier_samples, gate_rotation)
        hessian = route_weighted_hessian(carrier_rotated, route_weights)
        rows = gate.shape[0]
        packed = full_gptq_quantize(
            torch.cat((gate_rotated, up_rotated)),
            hessian,
            global_scale=scale,
            search_grid=search_grid,
        )
        effective = _effective_from_packed(packed, gate_rotation, gate.device)
        return effective[:rows], effective[rows:]
    return (
        _full_effective(
            gate, carrier_samples, route_weights, gate_rotation, scale, search_grid
        ),
        _full_effective(
            up, carrier_samples, route_weights, up_rotation, scale, search_grid
        ),
    )


def _variant_rotations(
    *,
    seed: int,
    layer: int,
    expert: int,
    projection: str,
    count: int,
    device: torch.device,
) -> list[tuple[str, torch.Tensor]]:
    rows = [("h16-fixed", hadamard16(device=device))]
    for index in range(1, count):
        identity = f"{seed}:l{layer}:e{expert}:{projection}:v{index}"
        rows.append((f"h16-sign-{index}", signed_hadamard16(identity, device=device)))
    return rows


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
    fit = plan["sampling"]["rotation_fit"]
    evaluation = plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=fit["count"],
        sample_offset=fit["offset"],
        sampling_strategy="domain-balanced",
    )
    eval_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=evaluation["count"],
        sample_offset=evaluation["offset"],
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(
        f"layers.{plan['layer']}."
    )[0]
    identity = torch.eye(16, device=device)
    fixed_h16 = hadamard16(device=device)
    selection_metric = plan["candidate"].get("selection_metric", "fit-output-nmse")
    minimum_margin = plan["candidate"].get("minimum_margin_vs_fixed", 0.0)
    result_rows = []
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
        selected = {}
        search_receipts = {}
        for projection in ("gate_proj", "up_proj"):
            variants = _variant_rotations(
                seed=plan["candidate"]["seed"],
                layer=plan["layer"],
                expert=expert,
                projection=projection,
                count=plan["candidate"]["variants_per_tensor"],
                device=device,
            )
            name, rotation, receipt = _select_rotation(
                weights[projection],
                fit_hidden,
                fit_hidden,
                fit_route,
                variants,
                args.search_grid,
                selection_metric,
                minimum_margin,
            )
            selected[projection] = (name, rotation)
            search_receipts[projection] = receipt

        arms = {}
        candidate_gate, candidate_up = _full_pair(
            weights["gate_proj"],
            weights["up_proj"],
            fit_hidden,
            fit_route,
            selected["gate_proj"][1],
            selected["up_proj"][1],
            args.search_grid,
        )
        identity_gate, identity_up = _full_pair(
            weights["gate_proj"], weights["up_proj"], fit_hidden, fit_route,
            identity, identity, args.search_grid,
        )
        fixed_gate, fixed_up = _full_pair(
            weights["gate_proj"], weights["up_proj"], fit_hidden, fit_route,
            fixed_h16, fixed_h16, args.search_grid,
        )
        arms["tensorwise-signed-h16"] = {"gate_proj": candidate_gate, "up_proj": candidate_up}
        arms["identity-gptq"] = {"gate_proj": identity_gate, "up_proj": identity_up}
        arms["fixed-h16-gptq"] = {"gate_proj": fixed_gate, "up_proj": fixed_up}

        fit_reference_middle = _middle(
            fit_hidden, weights["gate_proj"], weights["up_proj"]
        )
        eval_reference_middle = _middle(
            eval_hidden, weights["gate_proj"], weights["up_proj"]
        )
        for arm_name, arm in arms.items():
            arm["fit_middle"] = _middle(fit_hidden, arm["gate_proj"], arm["up_proj"])
            arm["eval_middle"] = _middle(eval_hidden, arm["gate_proj"], arm["up_proj"])

        down_variants = _variant_rotations(
            seed=plan["candidate"]["seed"],
            layer=plan["layer"],
            expert=expert,
            projection="down_proj",
            count=plan["candidate"]["variants_per_tensor"],
            device=device,
        )
        down_name, down_rotation, down_receipt = _select_rotation(
            weights["down_proj"],
            fit_reference_middle,
            arms["tensorwise-signed-h16"]["fit_middle"],
            fit_route,
            down_variants,
            args.search_grid,
            selection_metric,
            minimum_margin,
        )
        selected["down_proj"] = (down_name, down_rotation)
        search_receipts["down_proj"] = down_receipt
        for arm_name, rotation in (
            ("tensorwise-signed-h16", down_rotation),
            ("identity-gptq", identity),
            ("fixed-h16-gptq", fixed_h16),
        ):
            rotated = apply_weight_rotation(weights["down_proj"], rotation)
            scale = refine_global_scale(
                rotated, search_grid=args.search_grid, iterations=2
            )
            arms[arm_name]["down_proj"] = _full_effective(
                weights["down_proj"],
                arms[arm_name]["fit_middle"],
                fit_route,
                rotation,
                scale,
                args.search_grid,
            )

        reference_output = F.linear(
            eval_reference_middle, weights["down_proj"]
        )
        for arm_name, arm in arms.items():
            actual_output = F.linear(arm["eval_middle"], arm["down_proj"])
            result_rows.append({
                "expert": expert,
                "arm": arm_name,
                "evaluation_full_expert_nmse": _full_nmse(
                    reference_output, actual_output, eval_route
                ),
                "selected_variants": (
                    {projection: value[0] for projection, value in selected.items()}
                    if arm_name == "tensorwise-signed-h16" else None
                ),
                "fit_search": search_receipts if arm_name == "tensorwise-signed-h16" else None,
            })
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, arms
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-tensorwise-signed-h16-pilot-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "expert_slice": [args.expert_start_index, args.expert_end_index],
        "rows": result_rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
