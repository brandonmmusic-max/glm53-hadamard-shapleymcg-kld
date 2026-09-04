"""Fit-role routed expert screen for the H-ALPHABET trellis hypothesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .block_gptq import refine_global_scale
from .capture import LayerCapture
from .modelopt import dequantize
from .screen_trellis_codebooks import _full_expert_metric, _metric
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import quantize_scalar_mxf, quantize_trellis_mxf
from .trellis_mxf import decode_trellis_mxf, pack_ue8m0
from .trellis_nvfp4 import refine_rtn_nvfp4_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sampling-strategy", choices=("stable", "domain-balanced"), default="domain-balanced")
    parser.add_argument("--bits", type=int, choices=(3, 4), required=True)
    parser.add_argument("--alphabet", choices=("e2m1", "e2m3", "e3m2", "e4m3"), required=True)
    parser.add_argument("--block-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--law", choices=("mcg", "sqg-xor-cheb-t12"), required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--scale-refinement-iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        args.max_samples,
        data_role="fit",
        sample_offset=args.sample_offset,
        sampling_strategy=args.sampling_strategy,
    )
    hidden, route = capture.samples(args.expert)
    hidden, route = hidden.to(args.device).float(), route.to(args.device)
    prefix = source.expert_prefix(args.layer, args.expert).split(f"layers.{args.layer}.")[0]
    base = f"{prefix}layers.{args.layer}.mlp.experts.{args.expert}"
    weights = {
        projection: source.get(f"{base}.{projection}.weight").to(args.device).float()
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    middle = (
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, weights["gate_proj"]).clamp(max=10.0))
        * torch.nn.functional.linear(hidden, weights["up_proj"]).clamp(-10.0, 10.0)
    )
    trellis = {}
    scalar = {}
    payloads = {}
    for projection, weight in weights.items():
        payload = quantize_trellis_mxf(
            weight,
            bits=args.bits,
            alphabet=args.alphabet,
            law=args.law,
            compander_scale=args.alpha,
            block_size=args.block_size,
            scale_refinement_iterations=args.scale_refinement_iterations,
        )
        payloads[projection] = payload
        trellis[projection] = payload.reconstruction
        scalar[projection] = quantize_scalar_mxf(
            weight,
            alphabet=args.alphabet,
            block_size=args.block_size,
            scale_refinement_iterations=args.scale_refinement_iterations,
        )[0]
    gate_up_scale = refine_global_scale(weights["gate_proj"], weights["up_proj"], search_grid=12, iterations=2)
    down_scale = refine_global_scale(weights["down_proj"], search_grid=12, iterations=2)
    nv_gate, nv_up = refine_rtn_nvfp4_group(
        [weights["gate_proj"], weights["up_proj"]],
        global_scale=gate_up_scale,
        scale_refinement_iterations=2,
    )
    nv_down = refine_rtn_nvfp4_group(
        [weights["down_proj"]], global_scale=down_scale, scale_refinement_iterations=2
    )[0]
    variants = {
        "trellis": trellis,
        "matched-scalar": scalar,
        "matched-nvfp4": {
            "gate_proj": dequantize(nv_gate).to(args.device),
            "up_proj": dequantize(nv_up).to(args.device),
            "down_proj": dequantize(nv_down).to(args.device),
        },
    }
    rows = []
    full = {}
    for variant, qweights in variants.items():
        for projection, inputs in (("gate_proj", hidden), ("up_proj", hidden), ("down_proj", middle)):
            rows.append({"variant": variant, "projection": projection, **_metric(weights[projection], qweights[projection], inputs, route)})
        full[variant] = _full_expert_metric(
            weights["gate_proj"], weights["up_proj"], weights["down_proj"],
            qweights["gate_proj"], qweights["up_proj"], qweights["down_proj"], hidden, route,
        )
    trellis_error = full["trellis"]["error"]
    result = {
        "schema": "glm53-trellis-alphabet-screen.v1",
        "status": "fit-only development screen",
        "data_role": "fit",
        "candidate_adaptation_on_selection_or_confirmation": False,
        "layer": args.layer,
        "expert": args.expert,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "sampling_strategy": args.sampling_strategy,
        "bits": args.bits,
        "alphabet": args.alphabet,
        "block_size": args.block_size,
        "scale_format": "UE8M0 pseudoquant power-of-two",
        "law": args.law,
        "alpha": args.alpha,
        "stored_bpw": payloads["gate_proj"].stored_bpw,
        "native_compute_family": "mxf4nvf4" if args.alphabet == "e2m1" and args.block_size == 16 else ("mxf8f6f4" if args.block_size == 32 else "no documented one-MMA block16 path"),
        "aggregate": {
            name: {
                **metric,
                "nmse": metric["error"] / metric["energy"],
                "trellis_reduction_percent": (metric["error"] - trellis_error) / metric["error"] * 100,
            }
            for name, metric in full.items()
        },
        "codebook_sha256": sha256_file(args.output) if False else None,
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "source_index_sha256": sha256_file(args.source_index),
        "rows": rows,
    }
    # Hash the shared state LUT without writing a second artifact.
    import hashlib
    result["codebook_sha256"] = hashlib.sha256(payloads["gate_proj"].codebook_e4m3.numpy().tobytes()).hexdigest()
    if args.artifact_output is not None:
        artifact_tensors = {"codec.codebook_e4m3": payloads["gate_proj"].codebook_e4m3}
        for projection, payload in payloads.items():
            stem = f"{base}.{projection}"
            artifact_tensors[f"{stem}.trellis"] = payload.trellis
            artifact_tensors[f"{stem}.scale_ue8m0"] = pack_ue8m0(payload.scales)
            decoded = decode_trellis_mxf(
                payload.trellis,
                payload.codebook_e4m3,
                artifact_tensors[f"{stem}.scale_ue8m0"],
                bits=args.bits,
                block_size=args.block_size,
                rows=weights[projection].shape[0],
                width=weights[projection].shape[1],
                device=args.device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"frozen MXF closure failed for {projection}")
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            artifact_tensors,
            str(args.artifact_output),
            metadata={
                "schema": "glm53-trellis-mxf-frozen.v1",
                "layer": str(args.layer),
                "expert": str(args.expert),
                "bits": str(args.bits),
                "alphabet": args.alphabet,
                "block_size": str(args.block_size),
                "law": args.law,
                "alpha": str(args.alpha),
                "selection_role": "fit",
                "sampling_strategy": args.sampling_strategy,
                "stored_bpw": str(payloads["gate_proj"].stored_bpw),
            },
        )
        result["frozen_artifact"] = {
            "path": str(args.artifact_output),
            "bytes": args.artifact_output.stat().st_size,
            "sha256": sha256_file(args.artifact_output),
            "closure": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"], "stored_bpw": result["stored_bpw"]}, sort_keys=True))


if __name__ == "__main__":
    main()
