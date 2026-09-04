"""Small GLM calibration screen for direct trellis-to-NVFP4 codebooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .block_gptq import mse_rtn_quantize, refine_global_scale
from .capture import LayerCapture
from .modelopt import PackedNVFP4, dequantize
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_nvfp4 import quantize_trellis_nvfp4, quantize_trellis_nvfp4_group


def _metric(source, reconstructed, inputs, route):
    error = reconstructed - source
    raw_num = error.square().sum().double()
    raw_den = source.square().sum().double()
    expected = torch.nn.functional.linear(inputs, source)
    actual = torch.nn.functional.linear(inputs, reconstructed)
    weights = route.float().square()
    weights /= weights.sum().clamp_min(1e-30)
    act_num = ((actual - expected).square().mean(1) * weights).sum().double()
    act_den = (expected.square().mean(1) * weights).sum().double()
    return {
        "weight_error": float(raw_num),
        "weight_energy": float(raw_den),
        "activation_error": float(act_num),
        "activation_energy": float(act_den),
        "weight_nmse": float(raw_num / raw_den),
        "activation_nmse": float(act_num / act_den),
    }


def _full_expert_metric(gate, up, down, qgate, qup, qdown, inputs, route):
    expected_middle = (
        torch.nn.functional.silu(
            torch.nn.functional.linear(inputs, gate).clamp(max=10.0)
        )
        * torch.nn.functional.linear(inputs, up).clamp(-10.0, 10.0)
    )
    actual_middle = (
        torch.nn.functional.silu(
            torch.nn.functional.linear(inputs, qgate).clamp(max=10.0)
        )
        * torch.nn.functional.linear(inputs, qup).clamp(-10.0, 10.0)
    )
    expected = torch.nn.functional.linear(expected_middle, down)
    actual = torch.nn.functional.linear(actual_middle, qdown)
    weights = route.float().square()
    weights /= weights.sum().clamp_min(1e-30)
    error = ((actual - expected).square().mean(1) * weights).sum().double()
    energy = (expected.square().mean(1) * weights).sum().double()
    return {
        "error": float(error),
        "energy": float(energy),
        "nmse": float(error / energy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--search-grid", type=int, default=12)
    parser.add_argument("--scale-refinement-iterations", type=int, default=2)
    parser.add_argument(
        "--data-role", choices=("fit", "selection", "confirmation"), default="fit"
    )
    parser.add_argument("--projections", choices=("gate-up", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME:BITS:mcg|mul1|sqg-normal|sqg-xor-cheb-t12:COMPANDER_SCALE",
    )
    args = parser.parse_args()
    candidates = []
    for specification in args.candidate:
        name, bits, law, scale = specification.split(":")
        candidates.append((name, int(bits), law, float(scale)))

    source = IndexedCheckpoint(args.source, args.source_index)
    carrier = IndexedCheckpoint(args.carrier)
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        args.max_samples,
        data_role=args.data_role,
    )
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    rows = []
    refinement_rows = []
    aggregate: dict[str, dict[str, float]] = {}
    full_expert_aggregate: dict[str, dict[str, float]] = {}
    for expert in range(args.expert_start, args.expert_end):
        hidden, route = capture.samples(expert)
        hidden = hidden.to(args.device).float()
        route = route.to(args.device)
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        tensors = {
            projection: f"{base}.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        weights = {
            projection: source.get(tensor).to(args.device).float()
            for projection, tensor in tensors.items()
        }
        gate, up, down = (weights[key] for key in ("gate_proj", "up_proj", "down_proj"))
        middle = (
            torch.nn.functional.silu(
                torch.nn.functional.linear(hidden, gate).clamp(max=10.0)
            )
            * torch.nn.functional.linear(hidden, up).clamp(-10.0, 10.0)
        )
        active_projections = (
            ("gate_proj", "up_proj", "down_proj")
            if args.projections == "all"
            else ("gate_proj", "up_proj")
        )
        variants: dict[str, dict[str, torch.Tensor]] = {"stock": {}}
        for projection in active_projections:
            tensor = tensors[projection]
            stock = PackedNVFP4(
                carrier.get(tensor),
                carrier.get(f"{tensor}_scale"),
                carrier.get(f"{tensor}_scale_2"),
            )
            variants["stock"][projection] = dequantize(stock).to(args.device)
        gate_up_scale = refine_global_scale(
            gate, up, search_grid=args.search_grid, iterations=2
        )
        variants["matched-rtn"] = {
            "gate_proj": dequantize(
                mse_rtn_quantize(gate, global_scale=gate_up_scale, search_grid=args.search_grid)
            ).to(args.device),
            "up_proj": dequantize(
                mse_rtn_quantize(up, global_scale=gate_up_scale, search_grid=args.search_grid)
            ).to(args.device),
        }
        if args.projections == "all":
            down_scale = refine_global_scale(
                down, search_grid=args.search_grid, iterations=2
            )
            variants["matched-rtn"]["down_proj"] = dequantize(
                mse_rtn_quantize(down, global_scale=down_scale, search_grid=args.search_grid)
            ).to(args.device)
        rates = {"stock": 4.5, "matched-rtn": 4.5}
        for name, bits, law, scale in candidates:
            gate_codec, up_codec = quantize_trellis_nvfp4_group(
                [gate, up],
                global_scale=gate_up_scale,
                bits=bits,
                codebook_law=law,
                compander_scale=scale,
                search_grid=args.search_grid,
                scale_refinement_iterations=args.scale_refinement_iterations,
            )
            variants[name] = {
                "gate_proj": dequantize(gate_codec.endpoint).to(args.device),
                "up_proj": dequantize(up_codec.endpoint).to(args.device),
            }
            if args.projections == "all":
                down_codec = quantize_trellis_nvfp4(
                    down,
                    global_scale=down_scale,
                    bits=bits,
                    codebook_law=law,
                    compander_scale=scale,
                    search_grid=args.search_grid,
                    scale_refinement_iterations=args.scale_refinement_iterations,
                )
                variants[name]["down_proj"] = dequantize(
                    down_codec.endpoint
                ).to(args.device)
            rates[name] = gate_codec.trellis_bpw + gate_codec.scale_bpw
            refinement_rows.append({
                "expert": expert,
                "variant": name,
                "iterations": gate_codec.scale_refinement_iterations,
                "gate_up_initial_mse": gate_codec.initial_reconstruction_mse,
                "gate_up_final_mse": gate_codec.final_reconstruction_mse,
                "down_initial_mse": (
                    down_codec.initial_reconstruction_mse
                    if args.projections == "all"
                    else None
                ),
                "down_final_mse": (
                    down_codec.final_reconstruction_mse
                    if args.projections == "all"
                    else None
                ),
            })
        for name, reconstructed_by_projection in variants.items():
            for projection in active_projections:
                inputs = middle if projection == "down_proj" else hidden
                metric = _metric(
                    weights[projection],
                    reconstructed_by_projection[projection],
                    inputs,
                    route,
                )
                rows.append(
                    {
                        "expert": expert,
                        "projection": projection,
                        "variant": name,
                        "stored_bpw": rates[name],
                        **metric,
                    }
                )
                totals = aggregate.setdefault(
                    name,
                    {
                        "weight_error": 0.0,
                        "weight_energy": 0.0,
                        "activation_error": 0.0,
                        "activation_energy": 0.0,
                    },
                )
                for key in totals:
                    totals[key] += metric[key]
            if args.projections == "all":
                functional = _full_expert_metric(
                    gate,
                    up,
                    down,
                    reconstructed_by_projection["gate_proj"],
                    reconstructed_by_projection["up_proj"],
                    reconstructed_by_projection["down_proj"],
                    hidden,
                    route,
                )
                rows.append(
                    {
                        "expert": expert,
                        "projection": "full_expert",
                        "variant": name,
                        "stored_bpw": rates[name],
                        **{f"functional_{key}": value for key, value in functional.items()},
                    }
                )
                totals = full_expert_aggregate.setdefault(
                    name, {"error": 0.0, "energy": 0.0}
                )
                totals["error"] += functional["error"]
                totals["energy"] += functional["energy"]
    for name, totals in aggregate.items():
        totals["weight_nmse"] = totals["weight_error"] / totals["weight_energy"]
        totals["activation_nmse"] = (
            totals["activation_error"] / totals["activation_energy"]
        )
    for totals in full_expert_aggregate.values():
        totals["nmse"] = totals["error"] / totals["energy"]
    payload = {
        "schema": "glm53-native-nvfp4-codebook-screen.v2",
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "max_samples": args.max_samples,
        "data_role": args.data_role,
        "projections": args.projections,
        "scale_refinement_iterations": args.scale_refinement_iterations,
        "ldlq": False,
        "rotation": False,
        "roles_sha256": sha256_file(args.roles),
        "source_index_sha256": sha256_file(args.source_index),
        "candidates": [
            {"name": name, "bits": bits, "law": law, "compander_scale": scale}
            for name, bits, law, scale in candidates
        ],
        "aggregate": aggregate,
        "full_expert_aggregate": full_expert_aggregate,
        "scale_refinement": refinement_rows,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
