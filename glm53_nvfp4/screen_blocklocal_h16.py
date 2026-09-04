"""Screen per-expert, per-tensor, per-physical-block signed H16 transforms."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import _best_scales, _prepare_inverse, full_gptq_quantize, refine_global_scale
from .block_rotation import apply_activation_rotation, apply_weight_rotation, hadamard16, signed_hadamard16, structured_hadamard16
from .capture import LayerCapture
from .modelopt import E2M1_LEVELS, dequantize
from .output_aware import route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(F.linear(hidden.float(), gate.float()).clamp(max=10.0)) * F.linear(
        hidden.float(), up.float()
    ).clamp(-10.0, 10.0)


def _full_nmse(reference: torch.Tensor, actual: torch.Tensor, route: torch.Tensor) -> float:
    weights = route.float().square()
    weights /= weights.sum().clamp_min(1e-30)
    numerator = ((actual.float() - reference.float()).double().square().mean(1) * weights.double()).sum()
    denominator = (reference.float().double().square().mean(1) * weights.double()).sum()
    return float((numerator / denominator.clamp_min(1e-30)).item())


def _variant_bank(identity: str, count: int, device: torch.device, family: str) -> list[torch.Tensor]:
    variants = [hadamard16(device=device)]
    if family == "signed-h16":
        variants.extend(
            signed_hadamard16(f"{identity}:v{index}", device=device)
            for index in range(1, count)
        )
    elif family == "structured-dph16":
        variants.extend(
            structured_hadamard16(f"bank-v1:{index}", device=device)
            for index in range(1, count)
        )
    else:
        raise ValueError(f"unknown transform family {family}")
    return variants


@torch.no_grad()
def _select_blocks(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    global_scale: torch.Tensor,
    variants: list[torch.Tensor],
    search_grid: int,
    method: str,
) -> tuple[torch.Tensor, list[int], list[float]]:
    blocks = weight.float().reshape(weight.shape[0], -1, 16)
    levels = E2M1_LEVELS.to(weight.device)
    mids = (levels[1:] + levels[:-1]) / 2
    score_rows = []
    for rotation in variants:
        transformed = blocks @ rotation
        rotated_hessian = torch.einsum(
            "ij,bjk,kl->bil", rotation.transpose(-1, -2), hessian, rotation
        )
        if method == "rtn-hessian":
            scales = _best_scales(transformed, global_scale, search_grid).float() * global_scale
            codes = torch.bucketize((transformed / scales[..., None]).abs(), mids)
            dequantized = levels[codes] * torch.sign(transformed) * scales[..., None]
        elif method == "block-gptq-hessian":
            hinv, permutation = _prepare_inverse(rotated_hessian, 0.01)
            work = torch.gather(
                transformed,
                2,
                permutation[None, :, :].expand(transformed.shape[0], -1, -1),
            ).clone()
            scales = _best_scales(work, global_scale, search_grid).float() * global_scale
            for column in range(16):
                current = work[:, :, column]
                codes = torch.bucketize((current / scales).abs(), mids)
                quantized = levels[codes] * torch.sign(current) * scales
                error = (current - quantized) / hinv[:, column, column][None, :]
                work[:, :, column:] -= error[:, :, None] * hinv[None, :, column, column:]
                work[:, :, column] = quantized
            inverse = torch.empty_like(permutation)
            inverse.scatter_(
                1,
                permutation,
                torch.arange(16, device=weight.device)[None, :].expand_as(permutation),
            )
            dequantized = torch.gather(
                work, 2, inverse[None, :, :].expand_as(work)
            )
        else:
            raise ValueError(f"unknown selection method {method}")
        effective_delta = dequantized @ rotation.transpose(-1, -2) - blocks
        score_rows.append(torch.einsum("obg,bgh,obh->b", effective_delta, hessian, effective_delta))
    scores = torch.stack(score_rows)
    selected = scores.argmin(0)
    rotations = torch.stack([variants[int(index)] for index in selected])
    gain = 1.0 - scores.gather(0, selected[None]).squeeze(0) / scores[0].clamp_min(1e-30)
    return rotations, selected.cpu().tolist(), gain.cpu().tolist()


def _effective(packed, rotation: torch.Tensor, device: torch.device) -> torch.Tensor:
    return apply_weight_rotation(
        dequantize(packed).to(device), rotation.transpose(-1, -2)
    )


def _full_quant(
    weight: torch.Tensor,
    carrier: torch.Tensor,
    route: torch.Tensor,
    rotation: torch.Tensor,
    global_scale: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    transformed = apply_weight_rotation(weight, rotation)
    hessian = route_weighted_hessian(apply_activation_rotation(carrier, rotation), route)
    packed = full_gptq_quantize(
        transformed, hessian, global_scale=global_scale, search_grid=search_grid
    )
    return _effective(packed, rotation, weight.device)


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
    fit_spec, eval_spec = plan["sampling"]["hessian_fit"], plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=fit_spec["count"], sample_offset=fit_spec["offset"], sampling_strategy="domain-balanced")
    eval_capture = LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=eval_spec["count"], sample_offset=eval_spec["offset"], sampling_strategy="domain-balanced")
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    identity = torch.eye(16, device=device)
    fixed = hadamard16(device=device)
    rows = []
    started = time.time()
    for expert in experts:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden, fit_route = fit_hidden_cpu.to(device), fit_route_cpu.to(device)
        eval_hidden, eval_route = eval_hidden_cpu.to(device), eval_route_cpu.to(device)
        stem = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {name: source.get(f"{stem}.{name}.weight").to(device).float() for name in ("gate_proj", "up_proj", "down_proj")}
        hidden_block_hessian = route_weighted_hessian(fit_hidden, fit_route).reshape(fit_hidden.shape[1] // 16, 16, fit_hidden.shape[1] // 16, 16)
        hidden_block_hessian = torch.stack([hidden_block_hessian[b, :, b, :] for b in range(hidden_block_hessian.shape[0])])
        family = plan["candidate"].get("transform_family", "signed-h16")
        selection_method = plan["candidate"].get("selection_method", "rtn-hessian")
        variants = {
            name: _variant_bank(f"l3:e{expert}:{name}", plan["candidate"]["variants_per_block"], device, family)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        fixed_gate = apply_weight_rotation(weights["gate_proj"], fixed)
        fixed_up = apply_weight_rotation(weights["up_proj"], fixed)
        shared_scale = refine_global_scale(fixed_gate, fixed_up, search_grid=args.search_grid, iterations=2)
        selected = {}
        receipts = {}
        for _ in range(plan["candidate"]["global_scale_refits"]):
            for name in ("gate_proj", "up_proj"):
                rotation, indexes, gains = _select_blocks(weights[name], hidden_block_hessian, shared_scale, variants[name], args.search_grid, selection_method)
                selected[name] = rotation
                receipts[name] = {"variant_indexes": indexes, "mean_local_gain_vs_fixed": float(torch.tensor(gains).mean())}
            shared_scale = refine_global_scale(
                apply_weight_rotation(weights["gate_proj"], selected["gate_proj"]),
                apply_weight_rotation(weights["up_proj"], selected["up_proj"]),
                search_grid=args.search_grid,
                iterations=2,
            )
        arms = {}
        for arm, gate_rotation, up_rotation in (
            ("identity", identity, identity),
            ("fixed-h16", fixed, fixed),
            ("blocklocal-h16", selected["gate_proj"], selected["up_proj"]),
        ):
            gate_transformed = apply_weight_rotation(weights["gate_proj"], gate_rotation)
            up_transformed = apply_weight_rotation(weights["up_proj"], up_rotation)
            arm_scale = refine_global_scale(gate_transformed, up_transformed, search_grid=args.search_grid, iterations=2)
            gate = _full_quant(weights["gate_proj"], fit_hidden, fit_route, gate_rotation, arm_scale, args.search_grid)
            up = _full_quant(weights["up_proj"], fit_hidden, fit_route, up_rotation, arm_scale, args.search_grid)
            arms[arm] = {"gate": gate, "up": up, "fit_middle": _middle(fit_hidden, gate, up), "eval_middle": _middle(eval_hidden, gate, up)}
        candidate_middle_hessian = route_weighted_hessian(arms["blocklocal-h16"]["fit_middle"], fit_route)
        candidate_block_hessian = candidate_middle_hessian.reshape(candidate_middle_hessian.shape[0] // 16, 16, candidate_middle_hessian.shape[1] // 16, 16)
        candidate_block_hessian = torch.stack([candidate_block_hessian[b, :, b, :] for b in range(candidate_block_hessian.shape[0])])
        down_fixed_transformed = apply_weight_rotation(weights["down_proj"], fixed)
        down_scale = refine_global_scale(down_fixed_transformed, search_grid=args.search_grid, iterations=2)
        down_rotation, indexes, gains = _select_blocks(weights["down_proj"], candidate_block_hessian, down_scale, variants["down_proj"], args.search_grid, selection_method)
        receipts["down_proj"] = {"variant_indexes": indexes, "mean_local_gain_vs_fixed": float(torch.tensor(gains).mean())}
        reference_middle = _middle(eval_hidden, weights["gate_proj"], weights["up_proj"])
        reference_output = F.linear(reference_middle, weights["down_proj"])
        for arm, rotation in (("identity", identity), ("fixed-h16", fixed), ("blocklocal-h16", down_rotation)):
            scale = refine_global_scale(apply_weight_rotation(weights["down_proj"], rotation), search_grid=args.search_grid, iterations=2)
            down = _full_quant(weights["down_proj"], arms[arm]["fit_middle"], fit_route, rotation, scale, args.search_grid)
            actual = F.linear(arms[arm]["eval_middle"], down)
            rows.append({"expert": expert, "arm": arm, "evaluation_full_expert_nmse": _full_nmse(reference_output, actual, eval_route), "selection_receipt": receipts if arm == "blocklocal-h16" else None})
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, arms, reference_middle, reference_output
        torch.cuda.empty_cache()
    payload = {"schema": "glm53-blocklocal-signed-h16-screen-raw.v1", "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "expert_slice": [args.expert_start_index, args.expert_end_index], "rows": rows, "elapsed_seconds": time.time() - started, "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
