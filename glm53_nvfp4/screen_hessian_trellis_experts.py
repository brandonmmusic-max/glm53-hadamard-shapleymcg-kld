"""Run the frozen 16-expert P8 Hessian-aware trellis projection screen."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .block_gptq import full_gptq_quantize
from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import output_nmse, route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import (
    quantize_scalar_mxf,
    quantize_trellis_mxf,
    quantize_trellis_mxf_gptq,
)


def _weight_nmse(source: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(
        (
            (candidate.float() - source.float()).double().square().sum()
            / source.float().double().square().sum().clamp_min(1e-30)
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit = plan["sampling"]["hessian_fit"]
    evaluation = plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=fit["count"],
        sample_offset=fit["offset"],
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    evaluation_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=evaluation["count"],
        sample_offset=evaluation["offset"],
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(
        f"layers.{plan['layer']}."
    )[0]
    rows = []
    started = time.time()
    for expert in plan["experts"]:
        fit_source_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_source_cpu, eval_route_cpu = evaluation_capture.samples(expert)
        fit_source = fit_source_cpu.to(device)
        fit_route = fit_route_cpu.to(device)
        eval_source = eval_source_cpu.to(device)
        eval_route = eval_route_cpu.to(device)
        fit_carrier = _qdq_e4m3_k32(fit_source, 1.0, "amax")
        eval_carrier = _qdq_e4m3_k32(eval_source, 1.0, "amax")
        hessian = route_weighted_hessian(fit_carrier, fit_route)
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        for projection in plan["projections"]:
            weight = source.get(f"{base}.{projection}.weight").to(device).float()
            scalar = quantize_scalar_mxf(weight, alphabet="e4m3", block_size=32)[0]
            plain = quantize_trellis_mxf(
                weight,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            candidate = quantize_trellis_mxf_gptq(
                weight,
                hessian,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            control = dequantize(
                full_gptq_quantize(weight, hessian, group_size=16)
            ).to(device)
            for variant, reconstructed, bpw in (
                # This unrestricted 253-level diagnostic needs one E4M3 byte
                # per weight.  It is an 8.25-bpw upper bound, not a K4 arm.
                ("scalar-e4m3-k32", scalar, 8.25),
                ("plain-mcg-k4", plain, 4.25),
                ("hessian-mcg-k4", candidate, 4.25),
                ("gptq-nvfp4", control, 4.5),
            ):
                rows.append({
                    "expert": expert,
                    "projection": projection,
                    "variant": variant,
                    "stored_bpw": bpw,
                    "weight_nmse": _weight_nmse(weight, reconstructed),
                    "fit_output_nmse": output_nmse(
                        weight, reconstructed, fit_source, fit_carrier, fit_route
                    ),
                    "evaluation_output_nmse": output_nmse(
                        weight, reconstructed, eval_source, eval_carrier, eval_route
                    ),
                })
            del weight, scalar, plain, candidate, control
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del fit_source, fit_route, eval_source, eval_route, fit_carrier, eval_carrier, hessian
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-hessian-trellis-16expert-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
