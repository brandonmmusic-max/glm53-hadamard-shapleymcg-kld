"""Screen a learned per-block H16 bank on down_proj only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import refine_global_scale
from .block_rotation import apply_weight_rotation, hadamard16
from .capture import LayerCapture
from .output_aware import route_weighted_hessian
from .screen_blocklocal_h16 import _full_nmse, _full_quant, _middle, _select_blocks, _variant_bank
from .shard_index import IndexedCheckpoint, sha256_file


def _weight_nmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    numerator = (actual.double() - reference.double()).square().sum()
    denominator = reference.double().square().sum().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _block_diagonal(hessian: torch.Tensor) -> torch.Tensor:
    blocks = hessian.reshape(hessian.shape[0] // 16, 16, hessian.shape[1] // 16, 16)
    return torch.stack([blocks[index, :, index, :] for index in range(blocks.shape[0])])


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
        weights = {
            name: source.get(f"{stem}.{name}.weight").to(device).float()
            for name in ("gate_proj", "up_proj", "down_proj")
        }

        # Gate/up are quantized once and reused byte-for-byte by every down arm.
        gate_t = apply_weight_rotation(weights["gate_proj"], identity)
        up_t = apply_weight_rotation(weights["up_proj"], identity)
        shared_scale = refine_global_scale(gate_t, up_t, search_grid=args.search_grid, iterations=2)
        gate = _full_quant(weights["gate_proj"], fit_hidden, fit_route, identity, shared_scale, args.search_grid)
        up = _full_quant(weights["up_proj"], fit_hidden, fit_route, identity, shared_scale, args.search_grid)
        fit_middle = _middle(fit_hidden, gate, up)
        eval_middle = _middle(eval_hidden, gate, up)
        block_hessian = _block_diagonal(route_weighted_hessian(fit_middle, fit_route))

        variants = _variant_bank(
            f"l3:e{expert}:down_proj", plan["candidate"]["variants_per_block"],
            device, plan["candidate"]["transform_family"],
        )
        selected_scale = refine_global_scale(
            apply_weight_rotation(weights["down_proj"], fixed),
            search_grid=args.search_grid, iterations=2,
        )
        selected = None
        indexes = gains = None
        for _ in range(plan["candidate"]["global_scale_refits"]):
            selected, indexes, gains = _select_blocks(
                weights["down_proj"], block_hessian, selected_scale, variants,
                args.search_grid, plan["candidate"]["selection_method"],
            )
            selected_scale = refine_global_scale(
                apply_weight_rotation(weights["down_proj"], selected),
                search_grid=args.search_grid, iterations=2,
            )
        assert selected is not None and indexes is not None and gains is not None

        reference_middle = _middle(eval_hidden, weights["gate_proj"], weights["up_proj"])
        reference_output = F.linear(reference_middle, weights["down_proj"])
        for arm, rotation in (
            ("identity-down", identity),
            ("fixed-h16-down", fixed),
            ("blocklocal-h16-down", selected),
        ):
            scale = refine_global_scale(
                apply_weight_rotation(weights["down_proj"], rotation),
                search_grid=args.search_grid, iterations=2,
            )
            down = _full_quant(
                weights["down_proj"], fit_middle, fit_route, rotation, scale, args.search_grid
            )
            actual = F.linear(eval_middle, down)
            rows.append({
                "expert": expert,
                "arm": arm,
                "down_weight_nmse": _weight_nmse(weights["down_proj"], down),
                "evaluation_full_expert_nmse": _full_nmse(reference_output, actual, eval_route),
                "selection_receipt": ({
                    "variant_indexes": indexes,
                    "mean_local_gain_vs_fixed": float(torch.tensor(gains).mean()),
                    "selected_global_scale": float(selected_scale.item()),
                } if arm == "blocklocal-h16-down" else None),
            })
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, gate, up, fit_middle, eval_middle, block_hessian, reference_middle, reference_output
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-down-blocklocal-h16-screen-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "expert_slice": [args.expert_start_index, args.expert_end_index],
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
