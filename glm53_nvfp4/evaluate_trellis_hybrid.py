"""Evaluate a fit-frozen K4 hybrid artifact on a disjoint routed-data role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .block_gptq import refine_global_scale
from .capture import LayerCapture
from .modelopt import PackedNVFP4, dequantize
from .screen_trellis_codebooks import _full_expert_metric, _metric
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_nvfp4 import (
    HybridTrellisNVFP4,
    decode_hybrid_trellis_endpoint,
    refine_rtn_nvfp4_group,
)


def _bootstrap(rows: list[dict[str, float]], replicates: int, seed: int) -> dict[str, object]:
    count = len(rows)
    if count < 16:
        raise ValueError(
            f"expert bootstrap requires at least 16 independent experts, got {count}"
        )
    candidate_error = torch.tensor([row["candidate_error"] for row in rows], dtype=torch.float64)
    control_error = torch.tensor([row["control_error"] for row in rows], dtype=torch.float64)
    energy = torch.tensor([row["energy"] for row in rows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(count, (replicates, count), generator=generator)
    c = candidate_error[indices].sum(1) / energy[indices].sum(1)
    r = control_error[indices].sum(1) / energy[indices].sum(1)
    difference = c - r
    lower, upper = torch.quantile(difference, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "unit": "expert",
        "replicates": replicates,
        "seed": seed,
        "candidate_minus_control_nmse_ci95": [float(lower), float(upper)],
        "upper_below_zero": bool(upper < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--carrier-index", type=Path)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--search-grid", type=int, default=12)
    parser.add_argument("--scale-refinement-iterations", type=int, default=2)
    parser.add_argument("--practical-gate-percent", type=float, default=10.0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("stable", "domain-balanced"),
        default="stable",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-role", choices=("fit", "selection", "confirmation"), required=True
    )
    args = parser.parse_args()
    if args.practical_gate_percent < 10.0:
        raise ValueError("practical gate may not be weakened below the preregistered 10 percent")

    with safe_open(str(args.artifact), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata.get("schema") != "glm53-native-nvfp4-trilaw-tile-hybrid.v1":
        raise ValueError("unsupported frozen hybrid artifact schema")
    if metadata.get("data_role") != "fit":
        raise ValueError("candidate artifact was not frozen from fit-role data")
    layer = int(metadata["layer"])
    expert_start, expert_end = map(int, metadata["expert_range"].split(":"))
    specifications = json.loads(metadata["laws_json"])
    laws = tuple(item[1] for item in specifications)
    frozen = load_file(str(args.artifact), device="cpu")
    codebooks = frozen["codec.codebooks_e4m3"]

    source = IndexedCheckpoint(args.source, args.source_index)
    carrier = IndexedCheckpoint(args.carrier, args.carrier_index)
    capture = LayerCapture(
        args.capture_root,
        layer,
        args.roles,
        args.max_samples,
        data_role=args.data_role,
        sample_offset=args.sample_offset,
        sampling_strategy=args.sampling_strategy,
    )
    prefix = source.expert_prefix(layer, expert_start).split(f"layers.{layer}.")[0]
    rows = []
    aggregate = {
        name: {"error": 0.0, "energy": 0.0}
        for name in ("hybrid", "matched-rtn", "stock")
    }
    bootstrap_rows = []
    for expert in range(expert_start, expert_end):
        hidden, route = capture.samples(expert)
        hidden = hidden.to(args.device).float()
        route = route.to(args.device)
        base = f"{prefix}layers.{layer}.mlp.experts.{expert}"
        names = {
            projection: f"{base}.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        source_weights = {
            projection: source.get(name).to(args.device).float()
            for projection, name in names.items()
        }
        gate, up, down = (
            source_weights[projection]
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        middle = (
            torch.nn.functional.silu(
                torch.nn.functional.linear(hidden, gate).clamp(max=10.0)
            )
            * torch.nn.functional.linear(hidden, up).clamp(-10.0, 10.0)
        )
        reconstructed: dict[str, dict[str, torch.Tensor]] = {
            "hybrid": {},
            "stock": {},
        }
        for projection, name in names.items():
            stem = f"{base}.{projection}"
            endpoint = PackedNVFP4(
                frozen[f"{stem}.weight"],
                frozen[f"{stem}.weight_scale"],
                frozen[f"{stem}.weight_scale_2"],
            )
            width = endpoint.weight.shape[1] * 2
            hybrid = HybridTrellisNVFP4(
                endpoint=endpoint,
                trellis=frozen[f"{stem}.trellis"],
                selectors=frozen[f"{stem}.selectors_2bit"],
                selector_shape=(width // 16, endpoint.weight.shape[0] // 16),
                bits=4,
                codebooks_e4m3=codebooks,
                codebook_laws=laws,
            )
            decoded = decode_hybrid_trellis_endpoint(hybrid)
            if not (
                torch.equal(decoded.weight, endpoint.weight)
                and torch.equal(decoded.weight_scale, endpoint.weight_scale)
                and torch.equal(decoded.weight_scale_2, endpoint.weight_scale_2)
            ):
                raise RuntimeError(f"frozen hybrid closure failed for {stem}")
            reconstructed["hybrid"][projection] = dequantize(endpoint).to(args.device)
            reconstructed["stock"][projection] = dequantize(
                PackedNVFP4(
                    carrier.get(name),
                    carrier.get(f"{name}_scale"),
                    carrier.get(f"{name}_scale_2"),
                )
            ).to(args.device)
        gate_up_scale = refine_global_scale(
            gate, up, search_grid=args.search_grid, iterations=2
        )
        down_scale = refine_global_scale(down, search_grid=args.search_grid, iterations=2)
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
        reconstructed["matched-rtn"] = {
            "gate_proj": dequantize(matched_gate).to(args.device),
            "up_proj": dequantize(matched_up).to(args.device),
            "down_proj": dequantize(matched_down).to(args.device),
        }
        functional_by_variant = {}
        for variant, qweights in reconstructed.items():
            for projection, inputs in (
                ("gate_proj", hidden),
                ("up_proj", hidden),
                ("down_proj", middle),
            ):
                metric = _metric(
                    source_weights[projection], qweights[projection], inputs, route
                )
                rows.append(
                    {"expert": expert, "variant": variant, "projection": projection, **metric}
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
            functional_by_variant[variant] = functional
            rows.append(
                {
                    "expert": expert,
                    "variant": variant,
                    "projection": "full_expert",
                    **{f"functional_{key}": value for key, value in functional.items()},
                }
            )
            aggregate[variant]["error"] += functional["error"]
            aggregate[variant]["energy"] += functional["energy"]
        bootstrap_rows.append(
            {
                "candidate_error": functional_by_variant["hybrid"]["error"],
                "control_error": functional_by_variant["matched-rtn"]["error"],
                "energy": functional_by_variant["matched-rtn"]["energy"],
            }
        )
        del gate, up, down, hidden, middle
        torch.cuda.empty_cache()

    control_error = aggregate["matched-rtn"]["error"]
    for totals in aggregate.values():
        totals["nmse"] = totals["error"] / totals["energy"]
        totals["relative_error_reduction_vs_matched_rtn_percent"] = (
            (control_error - totals["error"]) / control_error * 100
        )
    uncertainty = (
        _bootstrap(bootstrap_rows, 10000, 5304)
        if len(bootstrap_rows) >= 16
        else {
            "status": "not-computed",
            "unit": "expert",
            "expert_count": len(bootstrap_rows),
            "reason": "expert-bootstrap inference requires at least 16 experts",
        }
    )
    effect_percent = aggregate["hybrid"][
        "relative_error_reduction_vs_matched_rtn_percent"
    ]
    decision = "pass" if effect_percent >= args.practical_gate_percent else "fail"
    result = {
        "schema": "glm53-native-nvfp4-trilaw-tile-hybrid-evaluation.v1",
        "data_role": args.data_role,
        "candidate_adaptation_on_evaluation_role": False,
        "artifact": {
            "path": str(args.artifact),
            "bytes": args.artifact.stat().st_size,
            "sha256": sha256_file(args.artifact),
        },
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "source_index_sha256": sha256_file(args.source_index),
        "layer": layer,
        "expert_range": [expert_start, expert_end],
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "sampling_strategy": args.sampling_strategy,
        "aggregate": aggregate,
        "uncertainty": uncertainty,
        "practical_gate_percent": args.practical_gate_percent,
        "decision_rule": "candidate aggregate relative error reduction must meet or exceed the preregistered practical gate; expert-bootstrap inference requires at least 16 experts",
        "decision": decision,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate, "uncertainty": uncertainty, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
