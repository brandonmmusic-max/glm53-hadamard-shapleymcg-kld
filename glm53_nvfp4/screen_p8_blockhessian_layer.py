"""Screen one layer with the stored 16x16 Hessians and the no-LDLQ P8 encoder."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from .block_gptq import gptq_quantize
from .modelopt import dequantize
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import quantize_trellis_mxf_gptq


def _weighted_error(weight: torch.Tensor, reconstructed: torch.Tensor, hessian: torch.Tensor) -> float:
    delta = (reconstructed.float() - weight.float()).reshape(weight.shape[0], -1, 16)
    numerator = torch.einsum("rbi,bij,rbj->", delta.double(), hessian.double(), delta.double())
    source = weight.float().reshape(weight.shape[0], -1, 16)
    denominator = torch.einsum("rbi,bij,rbj->", source.double(), hessian.double(), source.double())
    return float((numerator / denominator.clamp_min(1e-30)).item())


def _block_diagonal(hessian: torch.Tensor) -> torch.Tensor:
    groups, width, _ = hessian.shape
    result = torch.zeros((groups * width, groups * width), dtype=hessian.dtype, device=hessian.device)
    view = result.view(groups, width, groups, width)
    indices = torch.arange(groups, device=hessian.device)
    view[indices, :, indices, :] = hessian
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--hessian-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if args.layer not in plan["layers"]:
        raise ValueError("layer is outside the frozen sweep")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    hessians = load_file(str(args.hessian_file), device="cpu")
    prefix = source.expert_prefix(args.layer, plan["experts_per_layer"][0]).split(
        f"layers.{args.layer}."
    )[0]
    rows = []
    started = time.time()
    for expert in plan["experts_per_layer"]:
        hessian = hessians[f"hessian_{expert:03d}"].to(device).float()
        full_hessian = _block_diagonal(hessian)
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        projection_values = {}
        for projection in plan["projections"]:
            weight = source.get(f"{base}.{projection}.weight").to(device).float()
            candidate = quantize_trellis_mxf_gptq(
                weight,
                full_hessian,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            control = dequantize(gptq_quantize(weight, hessian, group_size=16)).to(device)
            candidate_error = _weighted_error(weight, candidate, hessian)
            control_error = _weighted_error(weight, control, hessian)
            projection_values[projection] = {
                "candidate_error": candidate_error,
                "control_error": control_error,
                "log_ratio": math.log(candidate_error / control_error),
            }
            del weight, candidate, control
        rows.append({
            "expert": expert,
            "projections": projection_values,
            "log_ratio_gmean": sum(row["log_ratio"] for row in projection_values.values()) / len(projection_values),
        })
        print(json.dumps({"layer": args.layer, "expert": expert, "completed": True}), flush=True)
        del hessian, full_hessian
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-p8-blockhessian-layer-raw.v1",
        "plan_sha256": sha256_file(args.plan),
        "layer": args.layer,
        "hessian_file": {"path": str(args.hessian_file), "sha256": sha256_file(args.hessian_file)},
        "source_index_sha256": sha256_file(args.source_index),
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
