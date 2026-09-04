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
    mr_gptq_quantize,
    mse_rtn_quantize,
    refine_global_scale,
)
from .block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
    hadamard,
    load_layer_rotation,
    orthogonality_error,
    rotate_block_hessian,
)
from .capture import LayerCapture
from .endpoint_codec import optimize_identity_k4_endpoint, pack_identity_k4_stream
from .modelopt import PackedNVFP4, choose_global_scale
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_nvfp4 import (
    decode_trellis_endpoint,
    quantize_trellis_nvfp4,
    quantize_trellis_nvfp4_group,
)


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
    parser.add_argument(
        "--sampling-strategy",
        choices=("stable", "domain-balanced"),
        default="stable",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument(
        "--quant-method",
        choices=("gptq", "mr-gptq", "fpquant-mr-gptq", "rtn", "trellis-nvfp4", "identity-k4"),
        default="gptq",
        help="packed solver; mr-gptq uses FP-Quant global static ActOrder and fixed precomputed group scales",
    )
    parser.add_argument(
        "--gptq-geometry",
        choices=("block16", "full"),
        default="block16",
        help="full reproduces Qwen's sequential full-Hessian GPTQ geometry",
    )
    parser.add_argument(
        "--rotation",
        choices=("identity", "had16", "had32", "had64", "learned"),
        default="identity",
    )
    parser.add_argument("--rotation-file", type=Path)
    parser.add_argument("--codec-output", type=Path)
    parser.add_argument("--trellis-bits", type=int, default=4)
    parser.add_argument(
        "--trellis-codebook",
        choices=("mcg", "mul1", "sqg-normal", "sqg-xor-cheb-t12"),
        default="mcg",
    )
    parser.add_argument(
        "--trellis-codebook-file",
        type=Path,
        help="safetensors file containing one uint8 codebook_e4m3 tensor",
    )
    parser.add_argument("--trellis-compander-scale", type=float, default=1.9)
    parser.add_argument("--trellis-scale-refinement-iterations", type=int, default=2)
    parser.add_argument("--endpoint-sweeps", type=int, default=1)
    parser.add_argument("--endpoint-ridge-ratio", type=float, default=3.0)
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
    if args.quant_method == "trellis-nvfp4" and args.rotation != "identity":
        raise ValueError("the direct native-NVFP4 trellis path forbids rotations")
    if (args.codec_output is None) != (args.quant_method != "trellis-nvfp4"):
        raise ValueError("trellis-nvfp4 requires --codec-output, other methods forbid it")
    if args.trellis_codebook_file is not None and args.quant_method != "trellis-nvfp4":
        raise ValueError("a trellis codebook file requires trellis-nvfp4")
    if args.trellis_scale_refinement_iterations < 0:
        raise ValueError("trellis scale refinement iterations must be nonnegative")
    if args.quant_method == "identity-k4" and (
        args.gptq_geometry != "full" or args.projections != "all"
    ):
        raise ValueError("identity-k4 requires full Hessian geometry and all projections")

    started = time.time()
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    if (args.capture_root is None) == (args.hessian_file is None):
        raise ValueError("provide exactly one of --capture-root or --hessian-file")
    if args.gptq_geometry == "full" and args.hessian_file is not None:
        raise ValueError("full GPTQ currently requires routed samples, not block Hessian files")
    if args.gptq_geometry == "full" and args.hessian_output is not None:
        raise ValueError("full Hessians are per-expert scratch and are not persisted")
    if args.quant_method in {"mr-gptq", "fpquant-mr-gptq"} and args.gptq_geometry != "full":
        raise ValueError("mr-gptq requires full-Hessian geometry")
    capture = (
        LayerCapture(
            args.capture_root,
            args.layer,
            args.roles,
            args.max_samples,
            sampling_strategy=args.sampling_strategy,
        )
        if args.capture_root
        else None
    )
    hessian_bundle = load_file(str(args.hessian_file), device="cpu") if args.hessian_file else None
    sample_counts = capture.counts() if capture else {
        expert: int(hessian_bundle[f"samples_{expert:03d}"].item())
        for expert in range(args.expert_start, args.expert_end)
    }
    output: dict[str, torch.Tensor] = {}
    codec_output: dict[str, torch.Tensor] = {}
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
    custom_codebook = None
    custom_codebook_sha256 = None
    if args.trellis_codebook_file is not None:
        codebook_bundle = load_file(str(args.trellis_codebook_file), device="cpu")
        if set(codebook_bundle) != {"codebook_e4m3"}:
            raise ValueError(
                "trellis codebook file must contain only codebook_e4m3"
            )
        custom_codebook = codebook_bundle["codebook_e4m3"]
        custom_codebook_sha256 = sha256_file(args.trellis_codebook_file)
    if args.rotation == "identity":
        rotation_in = rotation_mid = None
    elif args.rotation in {"had16", "had32", "had64"}:
        fixed = hadamard(int(args.rotation[3:]), device=cuda_device)
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
            choose_global_scale(gate_basis, up_basis).to(args.device)
            if args.quant_method == "fpquant-mr-gptq"
            else
            refine_global_scale(
                gate_basis,
                up_basis,
                search_grid=args.search_grid,
                iterations=2,
            )
            if args.gptq_geometry == "full" or args.quant_method == "trellis-nvfp4"
            else choose_global_scale(gate_basis, up_basis).to(args.device)
        )
        if args.quant_method == "trellis-nvfp4":
            gate_codec, up_codec = quantize_trellis_nvfp4_group(
                [gate_basis, up_basis],
                global_scale=shared,
                bits=args.trellis_bits,
                codebook_law=args.trellis_codebook,
                codebook_e4m3=custom_codebook,
                compander_scale=args.trellis_compander_scale,
                search_grid=args.search_grid,
                scale_refinement_iterations=args.trellis_scale_refinement_iterations,
            )
            packed_gate = gate_codec.endpoint
            packed_up = up_codec.endpoint
            comparator = (
                compare_full_packed_to_rtn
                if args.gptq_geometry == "full"
                else compare_packed_to_rtn
            )
            gate_cmp = comparator(
                gate_basis,
                h_basis,
                packed_gate,
                packed_gate.weight_scale_2,
                args.search_grid,
                candidate_method=args.quant_method,
            )
            up_cmp = comparator(
                up_basis,
                h_basis,
                packed_up,
                packed_up.weight_scale_2,
                args.search_grid,
                candidate_method=args.quant_method,
            )
        elif args.gptq_geometry == "full":
            gate_rows = gate_basis.shape[0]
            stacked = torch.cat((gate_basis, up_basis), dim=0)
            if args.quant_method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "identity-k4"}:
                full_quantizer = (
                    mr_gptq_quantize
                    if args.quant_method in {"mr-gptq", "fpquant-mr-gptq"}
                    else full_gptq_quantize
                )
                full_kwargs = {
                    "global_scale": shared,
                    "search_grid": args.search_grid,
                }
                if args.quant_method == "fpquant-mr-gptq":
                    full_kwargs["scale_observer"] = "fpquant-mse-l2.4"
                packed_stacked = full_quantizer(stacked, h_basis, **full_kwargs)
            else:
                packed_stacked = mse_rtn_quantize(
                    stacked,
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
                gate_basis, h_basis, packed_gate, packed_gate.weight_scale_2, args.search_grid,
                candidate_method=args.quant_method,
                rtn_scale_observer=("fpquant-mse-l2.4" if args.quant_method == "fpquant-mr-gptq" else "modelopt-mse46"),
            )
            up_cmp = compare_full_packed_to_rtn(
                up_basis, h_basis, packed_up, packed_up.weight_scale_2, args.search_grid,
                candidate_method=args.quant_method,
                rtn_scale_observer=("fpquant-mse-l2.4" if args.quant_method == "fpquant-mr-gptq" else "modelopt-mse46"),
            )
        else:
            quantizer = gptq_quantize if args.quant_method == "gptq" else mse_rtn_quantize
            quantizer_args = {"global_scale": shared, "search_grid": args.search_grid}
            packed_gate = (
                quantizer(gate_basis, h_basis, **quantizer_args)
                if args.quant_method == "gptq"
                else quantizer(gate_basis, **quantizer_args)
            )
            packed_up = (
                quantizer(up_basis, h_basis, **quantizer_args)
                if args.quant_method == "gptq"
                else quantizer(up_basis, **quantizer_args)
            )
            gate_cmp = compare_packed_to_rtn(
                gate_basis, h_basis, packed_gate, packed_gate.weight_scale_2, args.search_grid,
                candidate_method=args.quant_method,
            )
            up_cmp = compare_packed_to_rtn(
                up_basis, h_basis, packed_up, packed_up.weight_scale_2, args.search_grid,
                candidate_method=args.quant_method,
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
            if args.quant_method == "trellis-nvfp4":
                down_global_scale = refine_global_scale(
                    down_basis, search_grid=args.search_grid, iterations=2
                )
                down_codec = quantize_trellis_nvfp4(
                    down_basis,
                    global_scale=down_global_scale,
                    bits=args.trellis_bits,
                    codebook_law=args.trellis_codebook,
                    codebook_e4m3=custom_codebook,
                    compander_scale=args.trellis_compander_scale,
                    search_grid=args.search_grid,
                    scale_refinement_iterations=args.trellis_scale_refinement_iterations,
                )
                packed_down = down_codec.endpoint
                if args.gptq_geometry == "full":
                    middle_basis = middle
                    h_mid_basis = full_hessian(middle_basis, route)
                    down_cmp = compare_full_packed_to_rtn(
                        down_basis,
                        h_mid_basis,
                        packed_down,
                        packed_down.weight_scale_2,
                        args.search_grid,
                        candidate_method=args.quant_method,
                    )
                    h_mid = None
                else:
                    h_mid = block_hessian(middle, route)
                    h_mid_basis = h_mid
                    down_cmp = compare_packed_to_rtn(
                        down_basis,
                        h_mid_basis,
                        packed_down,
                        packed_down.weight_scale_2,
                        args.search_grid,
                        candidate_method=args.quant_method,
                    )
            elif args.gptq_geometry == "full":
                middle_basis = (
                    middle
                    if rotation_mid is None
                    else apply_activation_rotation(middle, rotation_mid)
                )
                h_mid_basis = full_hessian(middle_basis, route)
                down_global_scale = (
                    choose_global_scale(down_basis).to(args.device)
                    if args.quant_method == "fpquant-mr-gptq"
                    else refine_global_scale(
                        down_basis, search_grid=args.search_grid, iterations=2
                    )
                )
                if args.quant_method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "identity-k4"}:
                    full_quantizer = (
                        mr_gptq_quantize
                        if args.quant_method in {"mr-gptq", "fpquant-mr-gptq"}
                        else full_gptq_quantize
                    )
                    full_kwargs = {
                        "global_scale": down_global_scale,
                        "search_grid": args.search_grid,
                    }
                    if args.quant_method == "fpquant-mr-gptq":
                        full_kwargs["scale_observer"] = "fpquant-mse-l2.4"
                    packed_down = full_quantizer(
                        down_basis, h_mid_basis, **full_kwargs
                    )
                    endpoint_history = None
                    if args.quant_method == "identity-k4":
                        packed_down, endpoint_steps = optimize_identity_k4_endpoint(
                            down_basis,
                            h_mid_basis,
                            packed_down,
                            sweeps=args.endpoint_sweeps,
                            ridge_ratio=args.endpoint_ridge_ratio,
                        )
                        identity_stream = pack_identity_k4_stream(packed_down)
                        endpoint_history = [step.to_dict() for step in endpoint_steps]
                        if identity_stream.trellis_bpw != 4.0 or identity_stream.scale_bpw != 0.5:
                            raise RuntimeError("identity-K4 endpoint rate is not exactly 4.5 bpw")
                else:
                    packed_down = mse_rtn_quantize(
                        down_basis,
                        global_scale=down_global_scale,
                        search_grid=args.search_grid,
                    )
                down_cmp = compare_full_packed_to_rtn(
                    down_basis,
                    h_mid_basis,
                    packed_down,
                    packed_down.weight_scale_2,
                    args.search_grid,
                    candidate_method=args.quant_method,
                    rtn_scale_observer=("fpquant-mse-l2.4" if args.quant_method == "fpquant-mr-gptq" else "modelopt-mse46"),
                )
                h_mid = None
            else:
                h_mid = block_hessian(middle, route)
                h_mid_basis = (
                    h_mid
                    if rotation_mid is None
                    else rotate_block_hessian(h_mid, rotation_mid)
                )
                packed_down = (
                    gptq_quantize(
                        down_basis, h_mid_basis, search_grid=args.search_grid
                    )
                    if args.quant_method == "gptq"
                    else mse_rtn_quantize(
                        down_basis, search_grid=args.search_grid
                    )
                )
                down_cmp = compare_packed_to_rtn(
                    down_basis,
                    h_mid_basis,
                    packed_down,
                    packed_down.weight_scale_2,
                    args.search_grid,
                    candidate_method=args.quant_method,
                )
            packed_items.append(("down_proj", packed_down))
        for proj, packed in packed_items:
            stem = f"{base}.{proj}"
            output[f"{stem}.weight"] = packed.weight
            output[f"{stem}.weight_scale"] = packed.weight_scale
            output[f"{stem}.weight_scale_2"] = packed.weight_scale_2
        if args.quant_method == "trellis-nvfp4":
            codecs = {
                "gate_proj": gate_codec,
                "up_proj": up_codec,
                **({"down_proj": down_codec} if args.projections == "all" else {}),
            }
            for proj, codec in codecs.items():
                stem = f"{base}.{proj}"
                decoded = decode_trellis_endpoint(codec)
                closure = (
                    torch.equal(decoded.weight, codec.endpoint.weight)
                    and torch.equal(
                        decoded.weight_scale, codec.endpoint.weight_scale
                    )
                    and torch.equal(
                        decoded.weight_scale_2, codec.endpoint.weight_scale_2
                    )
                )
                if not closure:
                    raise RuntimeError(f"trellis endpoint closure failed for {stem}")
                codec_output[f"{stem}.trellis"] = codec.trellis
                codec_output[f"{stem}.weight_scale"] = codec.endpoint.weight_scale
                codec_output[f"{stem}.weight_scale_2"] = codec.endpoint.weight_scale_2
            codec_output.setdefault("codec.codebook_e4m3", gate_codec.codebook_e4m3)
        metrics[str(expert)] = {"samples": sample_counts[expert], "gate": gate_cmp, "up": up_cmp}
        if args.quant_method == "trellis-nvfp4":
            metrics[str(expert)]["codec"] = {
                "trellis_bpw": gate_codec.trellis_bpw,
                "scale_bpw": gate_codec.scale_bpw,
                "stored_bpw_excluding_global_scalar": gate_codec.trellis_bpw + gate_codec.scale_bpw,
                "codebook_sha256": gate_codec.codebook_sha256,
                "codebook_law": gate_codec.codebook_law,
                "scale_refinement_iterations": gate_codec.scale_refinement_iterations,
                "initial_reconstruction_mse": gate_codec.initial_reconstruction_mse,
                "final_reconstruction_mse": gate_codec.final_reconstruction_mse,
                "endpoint_closure": True,
            }
        if down_cmp is not None:
            metrics[str(expert)]["down"] = down_cmp
            if args.quant_method == "identity-k4":
                metrics[str(expert)]["identity_k4"] = {
                    "bits": 4,
                    "scale_bpw": 0.5,
                    "stored_bpw": 4.5,
                    "ridge_ratio": args.endpoint_ridge_ratio,
                    "sweeps": args.endpoint_sweeps,
                    "reference_stream_endpoint_closure": True,
                    "runtime": "direct native E2M1 endpoint; no decode prologue required",
                    "ldlq": False,
                    "history": endpoint_history,
                }
        del h_basis, gate, up, gate_basis, up_basis
        if h_in is not None:
            del h_in
        if args.gptq_geometry == "full":
            del hidden_basis
            if args.quant_method != "trellis-nvfp4":
                del stacked, packed_stacked
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
        "schema": "glm53-nvfp4-v11.modelopt-rotated-layer-chunk.v1",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "source_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
    })
    codec_receipt = None
    if args.codec_output is not None:
        args.codec_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            codec_output,
            str(args.codec_output),
            metadata={
                "schema": f"glm53-native-nvfp4-trellis-k{args.trellis_bits}.v1",
                "layer": str(args.layer),
                "expert_range": f"{args.expert_start}:{args.expert_end}",
                "endpoint": "E2M1 plus UE4M3 group-16 scale",
            },
        )
        codec_receipt = {
            "path": str(args.codec_output),
            "bytes": args.codec_output.stat().st_size,
            "sha256": sha256_file(args.codec_output),
            "tensors": len(codec_output),
        }
        encoded_weights = sum(
            tensor.numel() * 2
            for name, tensor in output.items()
            if name.endswith(".weight")
        )
        logical_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in codec_output.values()
        )
        codec_receipt.update(
            {
                "encoded_weight_values": encoded_weights,
                "logical_tensor_bytes": logical_tensor_bytes,
                "container_overhead_bytes": (
                    args.codec_output.stat().st_size - logical_tensor_bytes
                ),
                "logical_stored_bpw_including_shared_lut": (
                    logical_tensor_bytes * 8 / encoded_weights
                ),
                "file_bpw_including_container": (
                    args.codec_output.stat().st_size * 8 / encoded_weights
                ),
                "shared_codebook_bytes": int(
                    codec_output["codec.codebook_e4m3"].numel()
                    * codec_output["codec.codebook_e4m3"].element_size()
                ),
                "selector_bytes": 0,
                "endpoint_closure": True,
            }
        )
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
        "algorithm": {"format": "ModelOpt NVFP4 E2M1", "group_size": 16, "search_grid": (100 if args.quant_method == "fpquant-mr-gptq" else args.search_grid), "scale_observer": ("fpquant-mse-l2.4-shrink0.8" if args.quant_method == "fpquant-mr-gptq" else "modelopt-mse46"), "global_scale_refit_iterations": (args.trellis_scale_refinement_iterations if args.quant_method == "trellis-nvfp4" else 0 if args.quant_method == "fpquant-mr-gptq" else 2 if args.gptq_geometry == "full" else 0), "percdamp": 0.01 if args.quant_method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "identity-k4"} else None, "max_samples": args.max_samples, "sampling_strategy": args.sampling_strategy, "route_power": 2, "control": ("matched FP-Quant L2.4-observer RTN" if args.quant_method == "fpquant-mr-gptq" else "matched MSE-search-grid RTN"), "quant_method": args.quant_method, "gptq_geometry": args.gptq_geometry if args.quant_method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "identity-k4"} else None, "column_block": (128 if args.gptq_geometry == "full" else 16) if args.quant_method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "identity-k4"} else None, "activation_order": ("global-preserve-original-group" if args.quant_method in {"mr-gptq", "fpquant-mr-gptq"} else "within-native-group" if args.quant_method in {"gptq", "identity-k4"} else "none"), "group_scale_timing": ("precomputed-before-error-feedback" if args.quant_method in {"mr-gptq", "fpquant-mr-gptq"} else "output-aware coordinate-refit after GPTQ" if args.quant_method == "identity-k4" else "searched-after-prior-error-feedback" if args.quant_method == "gptq" else "candidate-specific alternating Viterbi/fixed-code UE4M3 refit" if args.quant_method == "trellis-nvfp4" else "precomputed"), "gate_up_rows_concatenated": args.gptq_geometry == "full" and args.quant_method != "trellis-nvfp4", "packed_adaptation": "one representable FP32 global scale per checkpoint tensor family; E4M3 block scales", "rotation": args.rotation, "rotation_scope": args.rotation_scope, "projections": args.projections, "rotation_file": str(args.rotation_file) if args.rotation_file else None, "trellis": ({"bits": args.trellis_bits, "codebook_law": ("learned-e2m1" if custom_codebook is not None else args.trellis_codebook), "codebook_file": str(args.trellis_codebook_file) if args.trellis_codebook_file else None, "codebook_file_sha256": custom_codebook_sha256, "compander_scale": args.trellis_compander_scale, "tail_biting": True, "ldlq": False, "runtime_weight_decode_dtype": "E2M1", "dense_weight_reconstruction": False, "mma_count": 1, "mma_count_status": "design invariant; runtime qualification pending"} if args.quant_method == "trellis-nvfp4" else {"bits": 4, "codebook_law": "identity-low4", "runtime_table_bytes": 0, "ldlq": False, "runtime": "direct native endpoint", "endpoint_sweeps": args.endpoint_sweeps, "ridge_ratio": args.endpoint_ridge_ratio} if args.quant_method == "identity-k4" else None), "orthogonality_max_abs": {"in": orthogonality_error(rotation_in) if rotation_in is not None else 0.0, "mid": orthogonality_error(rotation_mid) if rotation_mid is not None else 0.0}},
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
        "codec_output": codec_receipt,
        "metrics": metrics,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(cuda_device),
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "output_sha256": receipt["output"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
