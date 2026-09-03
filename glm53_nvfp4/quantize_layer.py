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

from .block_gptq import block_hessian, compare_packed_to_rtn, gptq_quantize
from .block_rotation import (
    apply_weight_rotation,
    hadamard16,
    load_layer_rotation,
    orthogonality_error,
    rotate_block_hessian,
)
from .capture import LayerCapture
from .modelopt import choose_global_scale
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
    parser.add_argument("--rotation", choices=("identity", "had16", "learned"), default="identity")
    parser.add_argument("--rotation-file", type=Path)
    parser.add_argument("--projections", choices=("all", "gate-up"), default="all")
    args = parser.parse_args()
    if not (3 <= args.layer <= 44 and 0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid layer or expert range")

    started = time.time()
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    if (args.capture_root is None) == (args.hessian_file is None):
        raise ValueError("provide exactly one of --capture-root or --hessian-file")
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
        rotation = None
    elif args.rotation == "had16":
        rotation = hadamard16(device=cuda_device)
    else:
        if args.rotation_file is None:
            raise ValueError("learned rotation requires --rotation-file")
        rotation = load_layer_rotation(args.rotation_file, args.layer, device=cuda_device)

    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        names = {proj: f"{base}.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")}
        for name in names.values():
            source_files.add(checkpoint.weight_map[name])
        if capture is not None:
            hidden_cpu, route_cpu = capture.samples(expert)
            hidden = hidden_cpu.to(args.device)
            route = route_cpu.to(args.device)
            h_in = block_hessian(hidden, route)
        else:
            hidden = route = None
            h_in = hessian_bundle[f"hessian_{expert:03d}"].to(args.device)
        if args.hessian_output is not None:
            hessian_output[f"hessian_{expert:03d}"] = h_in.cpu().contiguous()
            hessian_output[f"samples_{expert:03d}"] = torch.tensor(sample_counts[expert], dtype=torch.int32)
        gate = checkpoint.get(names["gate_proj"]).to(args.device)
        up = checkpoint.get(names["up_proj"]).to(args.device)
        if rotation is None:
            gate_basis, up_basis, h_basis = gate, up, h_in
        else:
            gate_basis = apply_weight_rotation(gate, rotation)
            up_basis = apply_weight_rotation(up, rotation)
            h_basis = rotate_block_hessian(h_in, rotation)
        shared = choose_global_scale(gate_basis, up_basis).to(args.device)
        packed_gate = gptq_quantize(gate_basis, h_basis, global_scale=shared, search_grid=args.search_grid)
        packed_up = gptq_quantize(up_basis, h_basis, global_scale=shared, search_grid=args.search_grid)
        gate_cmp = compare_packed_to_rtn(gate_basis, h_basis, packed_gate, shared, args.search_grid)
        up_cmp = compare_packed_to_rtn(up_basis, h_basis, packed_up, shared, args.search_grid)

        # Match the official Glm5NextTextExperts activation exactly.
        packed_items = [("gate_proj", packed_gate), ("up_proj", packed_up)]
        down = h_mid = middle = None
        down_cmp = None
        if args.projections == "all":
            if hidden is None or route is None:
                raise ValueError("down projection quantization requires routed activation samples")
            down = checkpoint.get(names["down_proj"]).to(args.device)
            middle = F.silu(F.linear(hidden, gate).clamp(max=10.0)) * F.linear(hidden, up).clamp(-10.0, 10.0)
            h_mid = block_hessian(middle, route)
            packed_down = gptq_quantize(down, h_mid, search_grid=args.search_grid)
            down_cmp = compare_packed_to_rtn(down, h_mid, packed_down, packed_down.weight_scale_2, args.search_grid)
            packed_items.append(("down_proj", packed_down))
        for proj, packed in packed_items:
            stem = f"{base}.{proj}"
            output[f"{stem}.weight"] = packed.weight
            output[f"{stem}.weight_scale"] = packed.weight_scale
            output[f"{stem}.weight_scale_2"] = packed.weight_scale_2
        metrics[str(expert)] = {"samples": sample_counts[expert], "gate": gate_cmp, "up": up_cmp}
        if down_cmp is not None:
            metrics[str(expert)]["down"] = down_cmp
        del h_in, h_basis, gate, up, gate_basis, up_basis
        if hidden is not None:
            del hidden, route
        if down is not None:
            del h_mid, down, middle
        torch.cuda.empty_cache()
        ratios = {"gate": gate_cmp["ratio"], "up": up_cmp["ratio"]}
        if down_cmp is not None:
            ratios["down"] = down_cmp["ratio"]
        print(json.dumps({"layer": args.layer, "expert": expert, "ratios": ratios}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.output), metadata={
        "schema": "glm53-nvfp4-v3.modelopt-rotated-layer-chunk.v1" if rotation is not None else "glm53-nvfp4-v2.modelopt-layer-chunk.v1",
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
    receipt = {
        "schema": "glm53-nvfp4-v2.layer-chunk-receipt.v1",
        "layer": args.layer,
        "expert_start": args.expert_start,
        "expert_end": args.expert_end,
        "algorithm": {"format": "ModelOpt NVFP4 E2M1", "group_size": 16, "search_grid": args.search_grid, "percdamp": 0.01, "max_samples": args.max_samples, "route_power": 2, "control": "matched MSE-search-grid RTN", "rotation": args.rotation, "rotation_scope": "routed gate/up input only", "projections": args.projections, "rotation_file": str(args.rotation_file) if args.rotation_file else None, "orthogonality_max_abs": orthogonality_error(rotation) if rotation is not None else 0.0},
        "source_files": [{"path": name, "bytes": (args.source / name).stat().st_size, "sha256": sha256_file(args.source / name)} for name in sorted(source_files)],
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json") if args.capture_root else None,
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
