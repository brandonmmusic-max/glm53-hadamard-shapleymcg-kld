"""Build an identity-boundary procedural-MCG P8 layer payload without LDLQ."""
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


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate_out = F.linear(carrier, gate).clamp(max=10.0)
    up_out = F.linear(carrier, up).clamp(-10.0, 10.0)
    return _qdq_e4m3_k32(F.silu(gate_out) * up_out, 1.0, "amax").float()


def _nmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(
        ((actual.float() - reference.float()).double().square().sum()
         / reference.float().double().square().sum().clamp_min(1e-30)).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.dense_output, args.codec_output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    plan = json.loads(args.plan.read_text())
    if plan.get("algorithm_exclusion") != "LDLQ and BlockLDLQ are excluded":
        raise RuntimeError("identity build requires the frozen no-LDLQ boundary")
    if args.layer != plan["layer"] or not (0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("requested range is outside the frozen plan")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    source = IndexedCheckpoint(args.source, args.source_index)
    sample = plan["calibration"]
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        max_samples=sample["count"],
        sample_offset=sample["offset"],
        data_role=sample["role"],
        sampling_strategy=sample["strategy"],
    )
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    dense_tensors: dict[str, torch.Tensor] = {}
    codec_tensors: dict[str, torch.Tensor] = {}
    metrics, codebook = [], None
    logical_elements = 0

    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        original = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden, route = hidden_cpu.to(device).float(), route_cpu.to(device).float()
        carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
        hessian = route_weighted_hessian(carrier, route)
        payloads = {
            "gate_proj": _quantize(original["gate_proj"], hessian),
            "up_proj": _quantize(original["up_proj"], hessian),
        }
        middle = _middle(
            hidden,
            payloads["gate_proj"].reconstruction,
            payloads["up_proj"].reconstruction,
        )
        payloads["down_proj"] = _quantize(
            original["down_proj"], route_weighted_hessian(middle, route)
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
                rows=original[projection].shape[0],
                width=original[projection].shape[1],
                device=device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"codec closure failed for {name}")
            dense_tensors[name] = decoded.to(torch.bfloat16).cpu().contiguous()
            codec_tensors[f"{base}.{projection}.trellis"] = payload.trellis.contiguous()
            codec_tensors[f"{base}.{projection}.scale_ue8m0"] = scale_codes.contiguous()
            if codebook is None:
                codebook = payload.codebook_e4m3.contiguous()
                codec_tensors["codec.codebook_e4m3"] = codebook
            elif not torch.equal(codebook, payload.codebook_e4m3):
                raise RuntimeError("procedural codebook changed within the chunk")
            metrics.append({
                "expert": expert,
                "projection": projection,
                "weight_nmse": _nmse(original[projection], decoded),
            })
            logical_elements += original[projection].numel()
        if expert % 4 == 3:
            print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del original, hidden, route, carrier, hessian, payloads, middle
        torch.cuda.empty_cache()

    assert codebook is not None
    metadata = {
        "schema": "glm53-p8-identity-mcg-layer-chunk.v1",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "bits": "4",
        "alphabet": "e4m3",
        "block_size": "32",
        "scale": "ue8m0",
        "law": "procedural-mcg-alpha2",
        "boundary": "identity",
        "ldlq": "false",
    }
    args.dense_output.parent.mkdir(parents=True, exist_ok=True)
    args.codec_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dense_tensors, str(args.dense_output), metadata={**metadata, "role": "decoded-pseudoquant-carrier"})
    save_file(codec_tensors, str(args.codec_output), metadata={**metadata, "role": "physical-codec"})
    receipt = {
        "schema": "glm53-p8-identity-mcg-layer-chunk-receipt.v1",
        "plan_sha256": sha256_file(args.plan),
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "calibration": sample,
        "algorithm": {
            "rate": "K4", "alphabet": "E4M3", "block_size": 32,
            "scale": "UE8M0 per 32 weights", "law": "procedural MCG alpha 2.0",
            "encoder": "Viterbi plus full-Hessian GPTQ-style inter-group error feedback",
            "boundary": "identity", "physical_bpw": 4.25, "ldlq": False,
        },
        "codebook_sha256": hashlib.sha256(codebook.numpy().tobytes()).hexdigest(),
        "logical_elements": logical_elements,
        "projection_metrics": metrics,
        "outputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in (("dense", args.dense_output), ("codec", args.codec_output))
        },
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
