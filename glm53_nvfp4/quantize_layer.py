"""Quantize a GLM routed-expert range from BF16 and write ModelOpt tensors plus a receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .block_gptq import (
    block_hessian,
    compare_full_packed_to_rtn,
    compare_packed_to_rtn,
    full_gptq_quantize,
    full_hessian,
    gptq_quantize,
    refine_global_scale,
)
from .block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
    hadamard16,
    load_layer_rotation,
    orthogonality_error,
    rotate_block_hessian,
)
from .capture import LayerCapture
from .modelopt import PackedNVFP4, choose_global_scale
from .shard_index import IndexedCheckpoint, sha256_file


def tensor_names(prefix: str, expert: int) -> dict[str, str]:
    base = f"{prefix}layers.{{layer}}.mlp.experts.{expert}.{{proj}}"
    return {proj: base.format(layer="{layer}", proj=proj) for proj in ("gate_proj", "up_proj", "down_proj")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--hessian-file", type=Path)
    parser.add_argument("--hessian-output", type=Path)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=288)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument(
        "--gptq-geometry",
        choices=("block16", "full"),
        default="block16",
        help="full reproduces Qwen's sequential full-Hessian GPTQ geometry",
    )
    parser.add_argument("--rotation", choices=("identity", "had16", "learned"), default="identity")
    parser.add_argument("--rotation-file", type=Path)
    parser.add_argument(
        "--rotation-scope",
        choices=("gate-up", "mid-only", "all"),
        default="gate-up",
        help="'all' reproduces the Qwen recipe with a separate down-input transform",
    )
    parser.add_argument("--projections", choices=("all", "gate-up"), default="all")
    args = parser.parse_args()
    if not (3 <= args.layer <= 44 and 0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid layer or expert range")

    started = time.time()
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    if (args.capture_root is None) == (args.hessian_file is None):
        raise ValueError("provide exactly one of --capture-root or --hessian-file")
    if args.gptq_geometry == "full" and args.hessian_file is not None:
        raise ValueError("full GPTQ currently requires routed samples, not block Hessian files")
    if args.gptq_geometry == "full" and args.hessian_output is not None:
        raise ValueError("full Hessians are per-expert scratch and are not persisted")
    capture = LayerCapture(args.capture_root, args.layer, args.roles, args.max_samples) if args.capture_root else None
    hessian_bundle = load_file(str(args.hessian_file), device="cpu") if args.hessian_file else None
    sample_counts = capture.counts() if capture else {
        expert: int(hessian_bundle[f"samples_{expert:03d}"].item())
        for expert in range(args.expert_start, args.expert_end)
    }
    output: dict[str, torch.Tensor] = {}
    hessian_output: dict[str, torch.Tensor] = {}
    metrics = {}
    source_files = set()
    prefix = checkpoint.expert_prefix(args.layer, args.expert_start).split(f"layers.{args.layer}.")[0]
    cuda_device = torch.device(args.device)
    if cuda_device.type != "cuda":
        raise ValueError("quantization requires a CUDA device")
    # torch 2.11 can reject reset_peak_memory_stats before the CUDA context exists.
    torch.cuda.set_device(cuda_device)
    torch.empty(0, device=cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    if args.rotation == "identity":
        rotation_in = rotation_mid = None
    elif args.rotation == "had16":
        fixed = hadamard16(device=cuda_device)
        rotation_in = fixed if args.rotation_scope in {"gate-up", "all"} else None
        rotation_mid = fixed if args.rotation_scope in {"mid-only", "all"} else None
    else:
        if args.rotation_file is None:
            raise ValueError("learned rotation requires --rotation-file")
        rotation_in = (
            load_layer_rotation(
                args.rotation_file, args.layer, kind="in", width=4096, device=cuda_device
            )
            if args.rotation_scope in {"gate-up", "all"}
            else None
        )
        rotation_mid = (
            load_layer_rotation(
                args.rotation_file,
                args.layer,
                kind="mid",
                width=2048,
                device=cuda_device,
            )
            if args.rotation_scope in {"mid-only", "all"}
            else None
        )

    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        names = {proj: f"{base}.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")}
        for name in names.values():
            source_files.add(checkpoint.weight_map[name])
        if capture is not None:
            hidden_cpu, route_cpu = capture.samples(expert)
            hidden = hidden_cpu.to(args.device)
            route = route_cpu.to(args.device)
            h_in = (
                block_hessian(hidden, route)
                if args.gptq_geometry == "block16"
                else None
            )
        else:
            hidden = route = None
            h_in = hessian_bundle[f"hessian_{expert:03d}"].to(args.device)
        if args.hessian_output is not None and h_in is not None:
            hessian_output[f"hessian_{expert:03d}"] = h_in.cpu().contiguous()
            hessian_output[f"samples_{expert:03d}"] = torch.tensor(sample_counts[expert], dtype=torch.int32)
        gate = checkpoint.get(names["gate_proj"]).to(args.device)
        up = checkpoint.get(names["up_proj"]).to(args.device)
        gate_basis = (
            gate if rotation_in is None else apply_weight_rotation(gate, rotation_in)
        )
        up_basis = up if rotation_in is None else apply_weight_rotation(up, rotation_in)
        if args.gptq_geometry == "full":
            hidden_basis = (
                hidden
                if rotation_in is None
                else apply_activation_rotation(hidden, rotation_in)
            )
            h_basis = full_hessian(hidden_basis, route)
        else:
            h_basis = (
                h_in
                if rotation_in is None
                else rotate_block_hessian(h_in, rotation_in)
            )
        shared = (
            refine_global_scale(
                gate_basis,
                up_basis,
                search_grid=args.search_grid,
                iterations=2,
            )
            if args.gptq_geometry == "full"
            else choose_global_scale(gate_basis, up_basis).to(args.device)
        )
        if args.gptq_geometry == "full":
            gate_rows = gate_basis.shape[0]
            stacked = torch.cat((gate_basis, up_basis), dim=0)
            packed_stacked = full_gptq_quantize(
                stacked,
                h_basis,
                global_scale=shared,
                search_grid=args.search_grid,
            )
            packed_gate = PackedNVFP4(
                packed_stacked.weight[:gate_rows].contiguous(),
                packed_stacked.weight_scale[:gate_rows].contiguous(),
                packed_stacked.weight_scale_2,
            )
            packed_up = PackedNVFP4(
                packed_stacked.weight[gate_rows:].contiguous(),
                packed_stacked.weight_scale[gate_rows:].contiguous(),
                packed_stacked.weight_scale_2.clone(),
            )
            gate_cmp = compare_full_packed_to_rtn(
                gate_basis, h_basis, packed_gate, shared, args.search_grid
            )
            up_cmp = compare_full_packed_to_rtn(
                up_basis, h_basis, packed_up, shared, args.search_grid
            )
        else:
            packed_gate = gptq_quantize(
                gate_basis,
                h_basis,
                global_scale=shared,
                search_grid=args.search_grid,
            )
            packed_up = gptq_quantize(
                up_basis,
                h_basis,
                global_scale=shared,
                search_grid=args.search_grid,
            )
            gate_cmp = compare_packed_to_rtn(
                gate_basis, h_basis, packed_gate, shared, args.search_grid
            )
            up_cmp = compare_packed_to_rtn(
                up_basis, h_basis, packed_up, shared, args.search_grid
            )

        # Match the official Glm5NextTextExperts activation exactly.
        packed_items = [("gate_proj", packed_gate), ("up_proj", packed_up)]
        down = h_mid = middle = None
        down_cmp = None
        if args.projections == "all":
            if hidden is None or route is None:
                raise ValueError("down projection quantization requires routed activation samples")
            down = checkpoint.get(names["down_proj"]).to(args.device)
            middle = F.silu(F.linear(hidden, gate).clamp(max=10.0)) * F.linear(hidden, up).clamp(-10.0, 10.0)
            down_basis = (
                down
                if rotation_mid is None
                else apply_weight_rotation(down, rotation_mid)
            )
            if args.gptq_geometry == "full":
                middle_basis = (
                    middle
                    if rotation_mid is None
                    else apply_activation_rotation(middle, rotation_mid)
                )
                h_mid_basis = full_hessian(middle_basis, route)
                down_global_scale = refine_global_scale(
                    down_basis, search_grid=args.search_grid, iterations=2
                )
                packed_down = full_gptq_quantize(
                    down_basis,
                    h_mid_basis,
                    global_scale=down_global_scale,
                    search_grid=args.search_grid,
                )
                down_cmp = compare_full_packed_to_rtn(
                    down_basis,
                    h_mid_basis,
                    packed_down,
                    packed_down.weight_scale_2,
                    args.search_grid,
                )
                h_mid = None
            else:
                h_mid = block_hessian(middle, route)
                h_mid_basis = (
                    h_mid
                    if rotation_mid is None
                    else rotate_block_hessian(h_mid, rotation_mid)
                )
                packed_down = gptq_quantize(
                    down_basis, h_mid_basis, search_grid=args.search_grid
                )
                down_cmp = compare_packed_to_rtn(
                    down_basis,
                    h_mid_basis,
                    packed_down,
                    packed_down.weight_scale_2,
                    args.search_grid,
                )
            packed_items.append(("down_proj", packed_down))
        for proj, packed in packed_items:
            stem = f"{base}.{proj}"
            output[f"{stem}.weight"] = packed.weight
            output[f"{stem}.weight_scale"] = packed.weight_scale
            output[f"{stem}.weight_scale_2"] = packed.weight_scale_2
        metrics[str(expert)] = {"samples": sample_counts[expert], "gate": gate_cmp, "up": up_cmp}
        if down_cmp is not None:
            metrics[str(expert)]["down"] = down_cmp
        del h_basis, gate, up, gate_basis, up_basis
        if h_in is not None:
            del h_in
        if args.gptq_geometry == "full":
            del hidden_basis, stacked, packed_stacked
        if hidden is not None:
            del hidden, route
        if down is not None:
            del h_mid_basis, down, down_basis, middle
            if h_mid is not None:
                del h_mid
            if args.gptq_geometry == "full":
                del middle_basis
        torch.cuda.empty_cache()
        ratios = {"gate": gate_cmp["ratio"], "up": up_cmp["ratio"]}
        if down_cmp is not None:
            ratios["down"] = down_cmp["ratio"]
        print(json.dumps({"layer": args.layer, "expert": expert, "ratios": ratios}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.output), metadata={
        "schema": "glm53-nvfp4-v5.modelopt-qwen-full-gptq-layer-chunk.v1" if args.gptq_geometry == "full" else ("glm53-nvfp4-v5.modelopt-qwen-exact-rotated-layer-chunk.v1" if rotation_mid is not None else ("glm53-nvfp4-v3.modelopt-rotated-layer-chunk.v1" if rotation_in is not None else "glm53-nvfp4-v2.modelopt-layer-chunk.v1")),
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "source_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
    })
    hessian_receipt = None
    if args.hessian_output is not None:
        args.hessian_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(hessian_output, str(args.hessian_output), metadata={
            "schema": "glm53-nvfp4-v3.fit-block-hessians.v1",
            "layer": str(args.layer),
            "expert_range": f"{args.expert_start}:{args.expert_end}",
            "roles_sha256": sha256_file(args.roles),
        })
        hessian_receipt = {"path": str(args.hessian_output), "bytes": args.hessian_output.stat().st_size, "sha256": sha256_file(args.hessian_output)}
    partial_capture_path = (
        args.capture_root
        / f"layers/layer-{args.layer:03d}/partial-fit-capture.json"
        if args.capture_root
        else None
    )
    receipt = {
        "schema": "glm53-nvfp4-v2.layer-chunk-receipt.v1",
        "layer": args.layer,
        "expert_start": args.expert_start,
        "expert_end": args.expert_end,
        "algorithm": {"format": "ModelOpt NVFP4 E2M1", "group_size": 16, "search_grid": args.search_grid, "global_scale_refit_iterations": 2 if args.gptq_geometry == "full" else 0, "percdamp": 0.01, "max_samples": args.max_samples, "route_power": 2, "control": "matched MSE-search-grid RTN", "gptq_geometry": args.gptq_geometry, "column_block": 128 if args.gptq_geometry == "full" else 16, "static_in_group_act_order": True, "gate_up_rows_concatenated": args.gptq_geometry == "full", "packed_adaptation": "one representable FP32 global scale per checkpoint tensor family; E4M3 block scales", "rotation": args.rotation, "rotation_scope": args.rotation_scope, "projections": args.projections, "rotation_file": str(args.rotation_file) if args.rotation_file else None, "orthogonality_max_abs": {"in": orthogonality_error(rotation_in) if rotation_in is not None else 0.0, "mid": orthogonality_error(rotation_mid) if rotation_mid is not None else 0.0}},
        "source_files": [{"path": name, "bytes": (args.source / name).stat().st_size, "sha256": sha256_file(args.source / name)} for name in sorted(source_files)],
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json") if args.capture_root else None,
        "partial_capture": (
            {
                "path": str(partial_capture_path),
                "sha256": sha256_file(partial_capture_path),
            }
            if partial_capture_path is not None and partial_capture_path.is_file()
            else None
        ),
        "hessian_input": {"path": str(args.hessian_file), "sha256": sha256_file(args.hessian_file)} if args.hessian_file else None,
        "hessian_output": hessian_receipt,
        "roles_sha256": sha256_file(args.roles),
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output), "tensors": len(output)},
        "metrics": metrics,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(cuda_device),
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "output_sha256": receipt["output"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
