"""Real-REAP projection screen for the output-aware native P8 encoder."""
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
from .output_aware import (
    output_aware_weight_target,
    output_nmse,
    route_weighted_hessian,
)
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import (
    quantize_scalar_mxf,
    quantize_trellis_mxf,
    quantize_trellis_mxf_gptq,
)


def _weight_nmse(source: torch.Tensor, candidate: torch.Tensor) -> float:
    error = (candidate.float() - source.float()).double().square().sum()
    energy = source.float().double().square().sum()
    return float((error / energy).item())


def _score(
    source_weight: torch.Tensor,
    candidate: torch.Tensor,
    source_samples: torch.Tensor,
    carrier_samples: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict[str, float]:
    return {
        "weight_nmse": _weight_nmse(source_weight, candidate),
        "output_nmse": output_nmse(
            source_weight,
            candidate,
            source_samples,
            carrier_samples,
            route_weights,
        ),
    }


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
    fit_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=plan["sampling"]["fit"]["count"],
        sample_offset=plan["sampling"]["fit"]["offset"],
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    tuning_spec = plan["sampling"].get("tuning")
    tuning_capture = (
        LayerCapture(
            args.capture_root,
            plan["layer"],
            args.roles,
            max_samples=tuning_spec["count"],
            sample_offset=tuning_spec["offset"],
            data_role="fit",
            sampling_strategy="domain-balanced",
        )
        if tuning_spec is not None
        else None
    )
    validation_capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=plan["sampling"]["validation"]["count"],
        sample_offset=plan["sampling"]["validation"]["offset"],
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(
        f"layers.{plan['layer']}."
    )[0]
    started = time.time()
    rows: list[dict[str, object]] = []
    selections: dict[str, dict[str, float]] = {}
    for expert in plan["experts"]:
        fit_source_cpu, fit_route_cpu = fit_capture.samples(expert)
        tune_source_cpu, tune_route_cpu = (
            tuning_capture.samples(expert)
            if tuning_capture is not None
            else validation_capture.samples(expert)
        )
        val_source_cpu, val_route_cpu = validation_capture.samples(expert)
        fit_source = fit_source_cpu.to(device)
        val_source = val_source_cpu.to(device)
        fit_route = fit_route_cpu.to(device)
        tune_source = tune_source_cpu.to(device)
        tune_route = tune_route_cpu.to(device)
        val_route = val_route_cpu.to(device)
        fit_carrier = _qdq_e4m3_k32(fit_source, 1.0, "amax")
        tune_carrier = _qdq_e4m3_k32(tune_source, 1.0, "amax")
        val_carrier = _qdq_e4m3_k32(val_source, 1.0, "amax")
        hessian = route_weighted_hessian(fit_carrier, fit_route)
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        for projection in plan["projections"]:
            name = f"{base}.{projection}.weight"
            weight = source.get(name).to(device).float()
            fit_grid = []
            for ridge_ratio in plan["candidate"]["fit_grid"]["ridge_ratio"]:
                for correction_scale in plan["candidate"]["fit_grid"]["correction_scale"]:
                    target, fit_receipt = output_aware_weight_target(
                        weight,
                        fit_source,
                        fit_carrier,
                        fit_route,
                        ridge_ratio=ridge_ratio,
                        correction_scale=correction_scale,
                    )
                    fit_grid.append({
                        "ridge_ratio": ridge_ratio,
                        "correction_scale": correction_scale,
                        "unquantized_fit_output_nmse": output_nmse(
                            weight, target, fit_source, fit_carrier, fit_route
                        ),
                        "unquantized_tuning_output_nmse": output_nmse(
                            weight, target, tune_source, tune_carrier, tune_route
                        ),
                        "fit": fit_receipt.to_dict(),
                    })
                    del target
            selection_metric = (
                "unquantized_tuning_output_nmse"
                if tuning_capture is not None
                else "unquantized_fit_output_nmse"
            )
            selected = min(fit_grid, key=lambda row: row[selection_metric])
            selections[f"{expert}:{projection}"] = {
                "ridge_ratio": float(selected["ridge_ratio"]),
                "correction_scale": float(selected["correction_scale"]),
            }
            target, fit_receipt = output_aware_weight_target(
                weight,
                fit_source,
                fit_carrier,
                fit_route,
                ridge_ratio=selected["ridge_ratio"],
                correction_scale=selected["correction_scale"],
            )

            scalar, _ = quantize_scalar_mxf(
                weight, alphabet="e4m3", block_size=32
            )
            plain = quantize_trellis_mxf(
                weight,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            hessian_only = quantize_trellis_mxf_gptq(
                weight,
                hessian,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            output_aware = quantize_trellis_mxf_gptq(
                target,
                hessian,
                bits=4,
                alphabet="e4m3",
                law="mcg",
                compander_scale=2.0,
                block_size=32,
                scale_refinement_iterations=2,
            ).reconstruction
            nvfp4 = dequantize(
                full_gptq_quantize(weight, hessian, group_size=16)
            ).to(device)
            variants = {
                "bf16-weight-p8-carrier": weight,
                "scalar-e4m3-k32": scalar,
                "plain-mcg-k4": plain,
                "hessian-mcg-k4": hessian_only,
                "output-aware-hessian-mcg-k4": output_aware,
                "gptq-nvfp4": nvfp4,
            }
            for variant, candidate in variants.items():
                rows.append({
                    "expert": expert,
                    "projection": projection,
                    "variant": variant,
                    "stored_bpw": 4.5 if variant == "gptq-nvfp4" else (16.0 if variant == "bf16-weight-p8-carrier" else 4.25),
                    "fit": _score(weight, candidate, fit_source, fit_carrier, fit_route),
                    "tuning": _score(weight, candidate, tune_source, tune_carrier, tune_route),
                    "validation": _score(weight, candidate, val_source, val_carrier, val_route),
                })
            rows.append({
                "expert": expert,
                "projection": projection,
                "variant": "selected-output-aware-target-unquantized",
                "stored_bpw": 32.0,
                "fit_grid": fit_grid,
                "selected": selections[f"{expert}:{projection}"],
                "fit_receipt": fit_receipt.to_dict(),
                "fit": _score(weight, target, fit_source, fit_carrier, fit_route),
                "tuning": _score(weight, target, tune_source, tune_carrier, tune_route),
                "validation": _score(weight, target, val_source, val_carrier, val_route),
            })
            print(json.dumps(rows[-2], sort_keys=True), flush=True)
            del weight, target, scalar, plain, hessian_only, output_aware, nvfp4
            torch.cuda.empty_cache()

    payload = {
        "schema": "glm53-output-aware-trellis-screen-result.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "selections": selections,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
