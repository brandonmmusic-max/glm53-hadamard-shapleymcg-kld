"""Build one dense pseudoquant plus compressed P8 layer chunk from REAP Hessians."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .output_aware import route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import (
    decode_trellis_mxf,
    pack_ue8m0,
    quantize_trellis_mxf_gptq,
)


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax")
    gate_output = F.linear(carrier.float(), gate.float()).clamp(max=10.0)
    up_output = F.linear(carrier.float(), up.float()).clamp(-10.0, 10.0)
    return _qdq_e4m3_k32(F.silu(gate_output) * up_output, 1.0, "amax")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--codec-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.dense_output, args.codec_output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not (0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid expert range")
    device = torch.device(args.device)
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
    dense_tensors: dict[str, torch.Tensor] = {}
    codec_tensors: dict[str, torch.Tensor] = {}
    rows = []
    codebook = None
    logical_elements = 0
    for expert in range(args.expert_start, args.expert_end):
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device)
        route = route_cpu.to(device)
        hidden_hessian = route_weighted_hessian(
            _qdq_e4m3_k32(hidden, 1.0, "amax"), route
        )
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        weights = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        payloads = {
            "gate_proj": _quantize(weights["gate_proj"], hidden_hessian),
            "up_proj": _quantize(weights["up_proj"], hidden_hessian),
        }
        middle = _middle(
            hidden,
            payloads["gate_proj"].reconstruction,
            payloads["up_proj"].reconstruction,
        )
        payloads["down_proj"] = _quantize(
            weights["down_proj"], route_weighted_hessian(middle, route)
        )
        for projection, payload in payloads.items():
            name = f"{base}.{projection}.weight"
            scale_codes = pack_ue8m0(payload.scales)
            decoded = decode_trellis_mxf(
                payload.trellis,
                payload.codebook_e4m3,
                scale_codes,
                bits=payload.bits,
                block_size=payload.block_size,
                rows=weights[projection].shape[0],
                width=weights[projection].shape[1],
                device=device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"codec closure failed for {name}")
            dense_tensors[name] = decoded.to(torch.bfloat16).cpu().contiguous()
            codec_tensors[f"{base}.{projection}.trellis"] = payload.trellis
            codec_tensors[f"{base}.{projection}.scale_ue8m0"] = scale_codes
            if codebook is None:
                codebook = payload.codebook_e4m3
                codec_tensors["codec.codebook_e4m3"] = codebook
            elif not torch.equal(codebook, payload.codebook_e4m3):
                raise RuntimeError("procedural MCG codebook changed within chunk")
            delta = decoded - weights[projection]
            rows.append({
                "expert": expert,
                "projection": projection,
                "weight_nmse": float(
                    (delta.double().square().sum() / weights[projection].double().square().sum()).item()
                ),
            })
            logical_elements += weights[projection].numel()
        if expert % 4 == 3:
            print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del hidden, route, hidden_hessian, weights, payloads, middle
        torch.cuda.empty_cache()

    metadata = {
        "schema": "glm53-hessian-trellis-p8-layer-chunk.v1",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "bits": "4",
        "alphabet": "e4m3",
        "block_size": "32",
        "scale": "UE8M0",
        "law": "mcg",
        "alpha": "2.0",
        "encoder": "gptq-feedback-static-in-group-act-order",
    }
    args.dense_output.parent.mkdir(parents=True, exist_ok=True)
    args.codec_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dense_tensors, str(args.dense_output), metadata=metadata)
    save_file(codec_tensors, str(args.codec_output), metadata=metadata)
    assert codebook is not None
    receipt = {
        "schema": "glm53-hessian-trellis-p8-layer-chunk-receipt.v1",
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "calibration": {
            "role": "fit",
            "sampling": "domain-balanced routes per expert",
            "samples": args.samples,
            "activation_carrier": "E4M3 K32 UE8M0 amax",
            "causal_down_hessian": True,
        },
        "algorithm": {
            "bits": 4,
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0",
            "law": "procedural MCG",
            "alpha": 2.0,
            "error_feedback": "GPTQ-style full-Hessian between native trellis groups",
            "activation_order": "static within native 16-column group",
            "scale_refit_iterations": 2,
            "physical_bpw": 4.25,
            "ldlq": False,
        },
        "codebook_sha256": hashlib.sha256(codebook.numpy().tobytes()).hexdigest(),
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "logical_elements": logical_elements,
        "projection_metrics": rows,
        "dense_output": {"path": str(args.dense_output), "bytes": args.dense_output.stat().st_size, "sha256": sha256_file(args.dense_output)},
        "codec_output": {"path": str(args.codec_output), "bytes": args.codec_output.stat().st_size, "sha256": sha256_file(args.codec_output)},
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.time() - started,
        "runtime_status": "dense BF16 pseudoquant overlay only; P8 prologue not yet implemented",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
