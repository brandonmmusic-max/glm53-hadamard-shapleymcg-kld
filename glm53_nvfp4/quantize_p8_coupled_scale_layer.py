"""Encode one GLM-5.3-Flash layer in the coupled H512/H128 P8 basis.

This is a preparation entrypoint only.  It preserves the existing K4 MCG
trellis plus UE8M0/32 weight payload and carries EXL3-derived FP16 scale
metadata in a new fail-closed schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .capture import LayerCapture
from .output_aware import route_weighted_hessian
from .p8_coupled_scale import (
    COUPLED_ACTIVATION,
    COUPLED_BOUNDARY,
    COUPLED_CAST_ORDER,
    COUPLED_CHUNK_SCHEMA,
    COUPLED_FC1_INTERLEAVE,
    COUPLED_QUANTIZED_DOWN_ORDER,
    COUPLED_QUANTIZED_INPUT_ORDER,
    COUPLED_SIGN_DRAW,
    COUPLED_SIGN_GENERATOR,
    COUPLED_TP_SLICE,
    COUPLED_TRANSFORM_ID,
    COUPLED_TRANSFORM_SHA256,
    SUPPORTED_LAYERS,
    CoupledScaleSet,
    coupled_input_carrier,
    coupled_middle_carrier,
    encode_coupled_scale_weights,
    load_exact_exl3_scales,
    _tensor_sha256,
)
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, pack_ue8m0, quantize_trellis_mxf_gptq


def _quantize(weight: torch.Tensor, hessian: torch.Tensor):
    return quantize_trellis_mxf_gptq(
        weight,
        hessian,
        bits=4,
        alphabet="e4m3",
        law="mcg",
        compander_scale=2.0,
        block_size=32,
        scale_refinement_iterations=2,
    )


def _validate_design(
    args: argparse.Namespace,
) -> tuple[dict, list[int], CoupledScaleSet]:
    design = json.loads(args.design.read_text())
    if (
        design.get("schema") != "glm53-p8.coupled-scale-preparation.v2"
        or design.get("decision_before_result") is not True
        or design.get("boundary") != COUPLED_BOUNDARY
        or design.get("layers") != list(SUPPORTED_LAYERS)
        or design.get("bits") != 4
        or design.get("weight_payload_bpw") != 4.25
        or design.get("ldlq") is not False
        or design.get("activation") != COUPLED_ACTIVATION
        or design.get("cast_order") != COUPLED_CAST_ORDER
        or design.get("quantized_input_order") != COUPLED_QUANTIZED_INPUT_ORDER
        or design.get("quantized_down_order") != COUPLED_QUANTIZED_DOWN_ORDER
        or design.get("transform_id") != COUPLED_TRANSFORM_ID
        or design.get("encoder_transform_sha256") != COUPLED_TRANSFORM_SHA256
        or design.get("sign_generator") != COUPLED_SIGN_GENERATOR
        or design.get("sign_draw") != COUPLED_SIGN_DRAW
        or design.get("fc1_interleave") != COUPLED_FC1_INTERLEAVE
        or design.get("tp_slice") != COUPLED_TP_SLICE
        or design.get("data_policy", {}).get("protected_roles_opened") != []
        or args.layer not in SUPPORTED_LAYERS
    ):
        raise RuntimeError("invalid coupled-scale P8 preparation design")
    paths = {
        "encoder": Path(__file__),
        "coupled_reference": Path(__file__).with_name("p8_coupled_scale.py"),
        "trellis_codec": Path(__file__).with_name("trellis_mxf.py"),
        "activation_quantizer": Path(__file__).with_name("canary_mxfp6_reap.py"),
        "encoder_transform": (
            Path(__file__).resolve().parents[1]
            / "experiments/p8-coupled-transform-draw0-silu10-v1.json"
        ),
        "source_index": args.source_index,
        "fit_roles": args.roles,
        "capture_manifest": args.capture_root / "capture-manifest.json",
    }
    for name, path in paths.items():
        expected = design.get("inputs", {}).get(name, {})
        if expected.get("sha256") != sha256_file(path):
            raise RuntimeError(f"{name} differs from the coupled-scale design")
    scale_input = (
        design.get("inputs", {}).get("exl3_scale_sources", {}).get(str(args.layer), {})
    )
    scales = load_exact_exl3_scales(
        args.exl3_scales,
        layer=args.layer,
        expected_experts=288,
        expected_hidden=4096,
        expected_intermediate=2048,
    )
    if scale_input.get("sha256") != scales.source_sha256:
        raise RuntimeError("exl3_scale_source differs from the coupled-scale design")
    draws = design.get("intermediate_draws_by_layer", {}).get(str(args.layer))
    if not isinstance(draws, list) or len(draws) != 288:
        raise RuntimeError("design must freeze one intermediate draw for every expert")
    normalized = [int(value) for value in draws]
    if any(value != COUPLED_SIGN_DRAW for value in normalized):
        raise RuntimeError("V2 candidate requires preregistered draw zero for every expert")
    return design, normalized, scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--exl3-scales", type=Path, required=True)
    parser.add_argument("--codec-output", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True, choices=SUPPORTED_LAYERS)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.dense_output is not None:
        raise ValueError(
            "--dense-output is prohibited in this campaign: decoded BF16 "
            "carriers cost 14,495,514,624 bytes per layer and are never needed "
            "for the native P8 sidecar product"
        )
    outputs = [args.codec_output, args.receipt]
    if args.dense_output is not None:
        outputs.append(args.dense_output)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not 0 <= args.expert_start < args.expert_end <= 288:
        raise ValueError("invalid expert range")
    design, intermediate_draws, scales = _validate_design(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("coupled P8 encoding requires CUDA")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    source = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        max_samples=args.samples,
        sample_offset=0,
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    codec_tensors: dict[str, torch.Tensor] = {
        "coupled.gate_up_suh_fp16": scales.gate_up_suh,
        "coupled.gate_svh_fp16": scales.gate_svh[
            args.expert_start : args.expert_end
        ].contiguous(),
        "coupled.up_svh_fp16": scales.up_svh[
            args.expert_start : args.expert_end
        ].contiguous(),
        "coupled.down_suh_fp16": scales.down_suh[
            args.expert_start : args.expert_end
        ].contiguous(),
        "coupled.down_svh_fp16": scales.down_svh,
        "coupled.intermediate_draw_u8": torch.tensor(
            intermediate_draws[args.expert_start : args.expert_end], dtype=torch.uint8
        ),
    }
    dense_tensors: dict[str, torch.Tensor] = {}
    metrics: list[dict[str, object]] = []
    codebook = None
    logical_elements = 0

    for expert in range(args.expert_start, args.expert_end):
        intermediate_draw = intermediate_draws[expert]
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device)
        route = route_cpu.to(device)
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        original = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        scale_device = type(scales)(
            gate_up_suh=scales.gate_up_suh.to(device),
            gate_svh=scales.gate_svh[expert : expert + 1].to(device),
            up_svh=scales.up_svh[expert : expert + 1].to(device),
            down_suh=scales.down_suh[expert : expert + 1].to(device),
            down_svh=scales.down_svh.to(device),
            source_path=scales.source_path,
            source_sha256=scales.source_sha256,
            source_metadata=scales.source_metadata,
            tensor_hashes=scales.tensor_hashes,
        )
        transformed = encode_coupled_scale_weights(
            original["gate_proj"],
            original["up_proj"],
            original["down_proj"],
            scale_device,
            expert=0,
            intermediate_draw=intermediate_draw,
        )
        carrier = coupled_input_carrier(
            hidden, scale_device.gate_up_suh, quantize=True
        )
        hidden_hessian = route_weighted_hessian(carrier, route)
        payloads = {
            "gate_proj": _quantize(transformed[0], hidden_hessian),
            "up_proj": _quantize(transformed[1], hidden_hessian),
        }
        down_carrier = coupled_middle_carrier(
            carrier,
            payloads["gate_proj"].reconstruction,
            payloads["up_proj"].reconstruction,
            scale_device,
            expert=0,
            intermediate_draw=intermediate_draw,
            quantize=True,
        )
        payloads["down_proj"] = _quantize(
            transformed[2], route_weighted_hessian(down_carrier, route)
        )
        for projection, payload in payloads.items():
            target = transformed[("gate_proj", "up_proj", "down_proj").index(projection)]
            scale_codes = pack_ue8m0(payload.scales)
            decoded = decode_trellis_mxf(
                payload.trellis,
                payload.codebook_e4m3,
                scale_codes,
                bits=4,
                block_size=32,
                rows=target.shape[0],
                width=target.shape[1],
                device=device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"codec closure failed for {base}.{projection}")
            codec_tensors[f"{base}.{projection}.trellis"] = payload.trellis.cpu().contiguous()
            codec_tensors[f"{base}.{projection}.scale_ue8m0"] = scale_codes.cpu().contiguous()
            if args.dense_output is not None:
                dense_tensors[f"{base}.{projection}.weight"] = (
                    decoded.to(torch.bfloat16).cpu().contiguous()
                )
            if codebook is None:
                codebook = payload.codebook_e4m3.cpu().contiguous()
            elif not torch.equal(codebook, payload.codebook_e4m3.cpu()):
                raise RuntimeError("procedural MCG codebook changed within chunk")
            delta = decoded - target
            metrics.append(
                {
                    "expert": expert,
                    "projection": projection,
                    "transformed_weight_nmse": float(
                        (delta.double().square().sum() / target.double().square().sum()).item()
                    ),
                }
            )
            logical_elements += target.numel()
        del hidden, route, original, scale_device, transformed, carrier
        del hidden_hessian, payloads, down_carrier
        torch.cuda.empty_cache()
        print(json.dumps({"layer": args.layer, "expert": expert, "completed": True}), flush=True)

    assert codebook is not None
    weight_names = [
        name for name in codec_tensors if name.endswith((".trellis", ".scale_ue8m0"))
    ]
    weight_payload_bytes = sum(
        codec_tensors[name].numel() * codec_tensors[name].element_size()
        for name in weight_names
    )
    weight_payload_bpw = weight_payload_bytes * 8.0 / logical_elements
    if not math.isclose(weight_payload_bpw, 4.25, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"weight payload is {weight_payload_bpw} bpw, expected 4.25")
    metadata_bytes = sum(
        value.numel() * value.element_size()
        for name, value in codec_tensors.items()
        if name.startswith("coupled.")
    )
    metadata_bpw = metadata_bytes * 8.0 / logical_elements
    metadata = {
        "schema": COUPLED_CHUNK_SCHEMA,
        "role": "physical-codec",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "bits": "4",
        "alphabet": "e4m3",
        "block_size": "32",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": COUPLED_BOUNDARY,
        "activation": COUPLED_ACTIVATION,
        "cast_order": COUPLED_CAST_ORDER,
        "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER,
        "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
        "transform_id": COUPLED_TRANSFORM_ID,
        "encoder_transform_sha256": COUPLED_TRANSFORM_SHA256,
        "sign_generator": COUPLED_SIGN_GENERATOR,
        "sign_draw": str(COUPLED_SIGN_DRAW),
        "sign_pre_axis": "1",
        "sign_post_axis": "2",
        "fc1_interleave": COUPLED_FC1_INTERLEAVE,
        "tp_slice": COUPLED_TP_SLICE,
        "ldlq": "false",
        "encoder": "gptq-feedback-static-in-group-act-order",
        "fc1_trellis_slot_order": "gate-up",
        "fc1_scale_plane_order": "up-gate",
        "coupled_scale_order": "gate_svh-up_svh-down_suh",
        "design_sha256": sha256_file(args.design),
        "exl3_scale_source_sha256": scales.source_sha256,
        "codebook_sha256": hashlib.sha256(codebook.numpy().tobytes()).hexdigest(),
    }
    args.codec_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(codec_tensors, str(args.codec_output), metadata=metadata)
    if args.dense_output is not None:
        args.dense_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            dense_tensors,
            str(args.dense_output),
            metadata={**metadata, "role": "decoded-pseudoquant-carrier"},
        )
    outputs = {
        "codec": {
            "path": str(args.codec_output.resolve()),
            "bytes": args.codec_output.stat().st_size,
            "sha256": sha256_file(args.codec_output),
        }
    }
    if args.dense_output is not None:
        outputs["dense"] = {
            "path": str(args.dense_output.resolve()),
            "bytes": args.dense_output.stat().st_size,
            "sha256": sha256_file(args.dense_output),
        }
    receipt = {
        "schema": "glm53-hessian-trellis-p8-coupled-scale-layer-chunk-receipt.v1",
        "command": sys.argv,
        "design_sha256": sha256_file(args.design),
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "calibration": {
            "role": "fit",
            "sampling": "domain-balanced routes per expert",
            "samples": args.samples,
            "fc1_carrier": COUPLED_QUANTIZED_INPUT_ORDER,
            "down_carrier": COUPLED_QUANTIZED_DOWN_ORDER,
            "causal_down_hessian": True,
        },
        "algorithm": {
            "weight_payload_bpw": weight_payload_bpw,
            "chunk_metadata_bytes": metadata_bytes,
            "chunk_metadata_bpw": metadata_bpw,
            "boundary": COUPLED_BOUNDARY,
            "intermediate_draws": intermediate_draws[
                args.expert_start : args.expert_end
            ],
            "error_feedback": "GPTQ-style full-Hessian between native trellis groups",
            "activation_order": "static within native 16-column group",
            "scale_refit_iterations": 2,
            "ldlq": False,
        },
        "scale_source": {
            "path": str(scales.source_path),
            "sha256": scales.source_sha256,
            "metadata": scales.source_metadata,
            "aggregate_tensor_hashes": {
                name: _tensor_sha256(tensor)
                for name, tensor in scales.tensors().items()
            },
        },
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "logical_elements": logical_elements,
        "projection_metrics": metrics,
        "outputs": outputs,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "protected_roles_opened": [],
        "runtime_status": "prepared codec only; native runtime closure not performed",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
