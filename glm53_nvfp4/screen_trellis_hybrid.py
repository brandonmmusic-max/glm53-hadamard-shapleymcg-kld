"""Fit-only screen for a genuinely combined K4 SQG/MCG/MUL1 tile codec."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .block_gptq import refine_global_scale
from .capture import LayerCapture
from .modelopt import PackedNVFP4, dequantize
from .screen_trellis_codebooks import _full_expert_metric, _metric
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_nvfp4 import (
    decode_hybrid_trellis_endpoint,
    hybridize_trellis_tiles,
    quantize_trellis_nvfp4,
    quantize_trellis_nvfp4_group,
    refine_rtn_nvfp4_group,
)


def _assert_hybrid_closure(payload) -> None:
    decoded = decode_hybrid_trellis_endpoint(payload)
    if not (
        torch.equal(decoded.weight, payload.endpoint.weight)
        and torch.equal(decoded.weight_scale, payload.endpoint.weight_scale)
        and torch.equal(decoded.weight_scale_2, payload.endpoint.weight_scale_2)
    ):
        raise RuntimeError("hybrid trellis endpoint closure failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--carrier-index", type=Path)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="persist the fit-frozen hybrid endpoint, trellises, and selectors",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--search-grid", type=int, default=12)
    parser.add_argument("--scale-refinement-iterations", type=int, default=2)
    parser.add_argument("--selector-sweeps", type=int, default=2)
    parser.add_argument("--selector-phases", type=int, default=16)
    parser.add_argument("--selector-isolated-regularization", type=float, default=0.0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("stable", "domain-balanced"),
        default="stable",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-role", choices=("fit", "selection", "confirmation"), default="fit"
    )
    parser.add_argument(
        "--law",
        action="append",
        required=True,
        help="NAME:mcg|mul1|sqg-normal|sqg-xor-cheb-t12:ALPHA",
    )
    args = parser.parse_args()
    if not (0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid expert range")
    specifications = []
    for item in args.law:
        name, law, alpha = item.split(":")
        specifications.append((name, law, float(alpha)))
    if not 2 <= len(specifications) <= 4:
        raise ValueError("hybrid requires two through four laws")

    source = IndexedCheckpoint(args.source, args.source_index)
    carrier = IndexedCheckpoint(args.carrier, args.carrier_index)
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        args.max_samples,
        data_role=args.data_role,
        sample_offset=args.sample_offset,
        sampling_strategy=args.sampling_strategy,
    )
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    rows = []
    aggregate: dict[str, dict[str, float]] = {}
    tile_receipts = []
    artifact_tensors: dict[str, torch.Tensor] = {}
    for expert in range(args.expert_start, args.expert_end):
        hidden, route = capture.samples(expert)
        hidden = hidden.to(args.device).float()
        route = route.to(args.device)
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        names = {
            projection: f"{base}.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        gate, up, down = (
            source.get(names[projection]).to(args.device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        middle = (
            torch.nn.functional.silu(
                torch.nn.functional.linear(hidden, gate).clamp(max=10.0)
            )
            * torch.nn.functional.linear(hidden, up).clamp(-10.0, 10.0)
        )
        gate_up_scale = refine_global_scale(
            gate, up, search_grid=args.search_grid, iterations=2
        )
        down_scale = refine_global_scale(
            down, search_grid=args.search_grid, iterations=2
        )
        matched_gate, matched_up = refine_rtn_nvfp4_group(
            [gate, up],
            global_scale=gate_up_scale,
            search_grid=args.search_grid,
            scale_refinement_iterations=args.scale_refinement_iterations,
        )
        matched_down = refine_rtn_nvfp4_group(
            [down],
            global_scale=down_scale,
            search_grid=args.search_grid,
            scale_refinement_iterations=args.scale_refinement_iterations,
        )[0]
        reconstructed: dict[str, dict[str, torch.Tensor]] = {
            "matched-rtn": {
                "gate_proj": dequantize(matched_gate).to(args.device),
                "up_proj": dequantize(matched_up).to(args.device),
                "down_proj": dequantize(matched_down).to(args.device),
            },
            "stock": {},
        }
        for projection, name in names.items():
            reconstructed["stock"][projection] = dequantize(
                PackedNVFP4(
                    carrier.get(name),
                    carrier.get(f"{name}_scale"),
                    carrier.get(f"{name}_scale_2"),
                )
            ).to(args.device)

        per_law_codecs = {}
        for label, law, alpha in specifications:
            gate_codec, up_codec = quantize_trellis_nvfp4_group(
                [gate, up],
                global_scale=gate_up_scale,
                bits=4,
                codebook_law=law,
                compander_scale=alpha,
                search_grid=args.search_grid,
                scale_refinement_iterations=args.scale_refinement_iterations,
                refine_global_scale=False,
            )
            down_codec = quantize_trellis_nvfp4(
                down,
                global_scale=down_scale,
                bits=4,
                codebook_law=law,
                compander_scale=alpha,
                search_grid=args.search_grid,
                scale_refinement_iterations=args.scale_refinement_iterations,
                refine_global_scale=False,
            )
            per_law_codecs[label] = {
                "gate_proj": gate_codec,
                "up_proj": up_codec,
                "down_proj": down_codec,
            }
            reconstructed[label] = {
                projection: dequantize(codec.endpoint).to(args.device)
                for projection, codec in per_law_codecs[label].items()
            }

        hybrid_payloads = {}
        hybrid_diagnostics = {}
        for projection, weight, inputs in (
            ("gate_proj", gate, hidden),
            ("up_proj", up, hidden),
            ("down_proj", down, middle),
        ):
            payload, diagnostics = hybridize_trellis_tiles(
                weight,
                [per_law_codecs[label][projection] for label, _law, _alpha in specifications],
                inputs,
                route,
                selector_sweeps=args.selector_sweeps,
                selector_phases=args.selector_phases,
                selector_isolated_regularization=args.selector_isolated_regularization,
            )
            _assert_hybrid_closure(payload)
            hybrid_payloads[projection] = payload
            hybrid_diagnostics[projection] = diagnostics
            if args.artifact_output is not None:
                stem = f"{base}.{projection}"
                artifact_tensors[f"{stem}.weight"] = payload.endpoint.weight
                artifact_tensors[f"{stem}.weight_scale"] = payload.endpoint.weight_scale
                artifact_tensors[f"{stem}.weight_scale_2"] = payload.endpoint.weight_scale_2
                artifact_tensors[f"{stem}.trellis"] = payload.trellis
                artifact_tensors[f"{stem}.selectors_2bit"] = payload.selectors
                existing = artifact_tensors.get("codec.codebooks_e4m3")
                if existing is not None and not torch.equal(
                    existing, payload.codebooks_e4m3
                ):
                    raise RuntimeError("hybrid projections disagree on shared codebooks")
                artifact_tensors["codec.codebooks_e4m3"] = payload.codebooks_e4m3
        reconstructed["hybrid"] = {
            projection: dequantize(payload.endpoint).to(args.device)
            for projection, payload in hybrid_payloads.items()
        }
        tile_receipts.append(
            {
                "expert": expert,
                "diagnostics": hybrid_diagnostics,
                "endpoint_closure": True,
                "selector_bytes": sum(
                    payload.selectors.numel() for payload in hybrid_payloads.values()
                ),
            }
        )

        weights = {"gate_proj": gate, "up_proj": up, "down_proj": down}
        for variant, qweights in reconstructed.items():
            for projection, inputs in (
                ("gate_proj", hidden),
                ("up_proj", hidden),
                ("down_proj", middle),
            ):
                metric = _metric(
                    weights[projection], qweights[projection], inputs, route
                )
                rows.append(
                    {
                        "expert": expert,
                        "variant": variant,
                        "projection": projection,
                        **metric,
                    }
                )
            functional = _full_expert_metric(
                gate,
                up,
                down,
                qweights["gate_proj"],
                qweights["up_proj"],
                qweights["down_proj"],
                hidden,
                route,
            )
            rows.append(
                {
                    "expert": expert,
                    "variant": variant,
                    "projection": "full_expert",
                    **{f"functional_{key}": value for key, value in functional.items()},
                }
            )
            totals = aggregate.setdefault(variant, {"error": 0.0, "energy": 0.0})
            totals["error"] += functional["error"]
            totals["energy"] += functional["energy"]
        del gate, up, down, hidden, middle
        torch.cuda.empty_cache()

    for totals in aggregate.values():
        totals["nmse"] = totals["error"] / totals["energy"]
    control_error = aggregate["matched-rtn"]["error"]
    for totals in aggregate.values():
        totals["relative_error_reduction_vs_matched_rtn_percent"] = (
            (control_error - totals["error"]) / control_error * 100
        )
    artifact_receipt = None
    if args.artifact_output is not None:
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            artifact_tensors,
            str(args.artifact_output),
            metadata={
                "schema": "glm53-native-nvfp4-trilaw-tile-hybrid.v1",
                "layer": str(args.layer),
                "expert_range": f"{args.expert_start}:{args.expert_end}",
                "data_role": args.data_role,
                "sample_offset": str(args.sample_offset),
                "sampling_strategy": args.sampling_strategy,
                "bits": "4",
                "selector_sweeps": str(args.selector_sweeps),
                "selector_phases": str(args.selector_phases),
                "selector_isolated_regularization": str(args.selector_isolated_regularization),
                "selector_bits_per_16x16_tile": "2",
                "laws_json": json.dumps(specifications, separators=(",", ":")),
                "endpoint": "E2M1 plus UE4M3-per-16 plus FP32-global",
            },
        )
        artifact_receipt = {
            "path": str(args.artifact_output),
            "bytes": args.artifact_output.stat().st_size,
            "sha256": sha256_file(args.artifact_output),
            "tensor_count": len(artifact_tensors),
        }
    payload = {
        "schema": "glm53-native-nvfp4-trilaw-tile-hybrid-screen.v1",
        "status": "fit-only optimistic development bound" if args.data_role == "fit" else "controlled role evaluation",
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "data_role": args.data_role,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "sampling_strategy": args.sampling_strategy,
        "selector_sweeps": args.selector_sweeps,
        "selector_phases": args.selector_phases,
        "selector_isolated_regularization": args.selector_isolated_regularization,
        "bits": 4,
        "selector_granularity": "one 2-bit selector per 16x16 tile",
        "selector_bpw": 2 / 256,
        "laws": [
            {"label": label, "law": law, "alpha": alpha}
            for label, law, alpha in specifications
        ],
        "selection_objective": "routed activation-weighted isolated tile output SSE on the declared data role",
        "ldlq": False,
        "rotation": False,
        "dense_weight_reconstruction": False,
        "intended_runtime_mma_count": 1,
        "runtime_qualification": "Not tested",
        "frozen_artifact": artifact_receipt,
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "source_index_sha256": sha256_file(args.source_index),
        "aggregate": aggregate,
        "tile_receipts": tile_receipts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
