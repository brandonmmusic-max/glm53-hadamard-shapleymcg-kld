"""Evaluate one fit-frozen trellis-MXF artifact on a declared routed-data role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .block_gptq import refine_global_scale
from .capture import LayerCapture
from .modelopt import dequantize
from .screen_trellis_codebooks import _full_expert_metric, _metric
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, quantize_scalar_mxf
from .trellis_nvfp4 import refine_rtn_nvfp4_group


def _bootstrap_experts(
    rows: list[dict[str, float]], replicates: int = 10000, seed: int = 5304
) -> dict[str, object]:
    """Bootstrap aggregate candidate-minus-control NMSE over expert units."""
    if len(rows) < 16:
        return {
            "status": "not-computed",
            "unit": "expert",
            "expert_count": len(rows),
            "reason": "expert-bootstrap inference requires at least 16 experts",
        }
    candidate = torch.tensor(
        [row["candidate_error"] for row in rows], dtype=torch.float64
    )
    control = torch.tensor(
        [row["control_error"] for row in rows], dtype=torch.float64
    )
    energy = torch.tensor([row["energy"] for row in rows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(rows), (replicates, len(rows)), generator=generator)
    delta = (
        candidate[indices].sum(1) / energy[indices].sum(1)
        - control[indices].sum(1) / energy[indices].sum(1)
    )
    import numpy as np
    from scipy.stats import norm

    bootstrap = delta.numpy()
    observed = float(candidate.sum() / energy.sum() - control.sum() / energy.sum())
    z0 = norm.ppf(np.clip(np.mean(bootstrap < observed), 1e-9, 1 - 1e-9))
    jackknife = []
    for index in range(len(rows)):
        keep = torch.arange(len(rows)) != index
        jackknife.append(
            float(candidate[keep].sum() / energy[keep].sum())
            - float(control[keep].sum() / energy[keep].sum())
        )
    jackknife_array = np.asarray(jackknife)
    center = jackknife_array.mean()
    numerator = np.sum((center - jackknife_array) ** 3)
    denominator = 6 * np.sum((center - jackknife_array) ** 2) ** 1.5
    acceleration = numerator / denominator if denominator > 0 else 0.0
    bounds = []
    for target in (0.025, 0.975):
        z = norm.ppf(target)
        adjusted = norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        bounds.append(float(np.quantile(bootstrap, np.clip(adjusted, 0, 1))))
    lower, upper = bounds
    return {
        "status": "computed",
        "unit": "expert",
        "expert_count": len(rows),
        "replicates": replicates,
        "seed": seed,
        "candidate_minus_control_nmse_ci95_bca": [float(lower), float(upper)],
        "upper_below_zero": bool(upper < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sampling-strategy", choices=("stable", "domain-balanced"), default="stable")
    parser.add_argument("--data-role", choices=("fit", "selection", "confirmation"), required=True)
    parser.add_argument("--practical-gate-percent", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.practical_gate_percent < 10.0:
        raise ValueError("practical gate may not be weakened below the preregistered 10 percent")

    with safe_open(str(args.artifact), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("schema") != "glm53-trellis-mxf-frozen.v1":
        raise ValueError("unsupported frozen trellis-MXF artifact")
    if metadata.get("selection_role") != "fit":
        raise ValueError("trellis-MXF candidate was not selected on fit data")
    layer = int(metadata["layer"])
    expert = int(metadata["expert"])
    bits = int(metadata["bits"])
    alphabet = metadata["alphabet"]
    block_size = int(metadata["block_size"])
    frozen = load_file(str(args.artifact), device="cpu")
    codebook = frozen["codec.codebook_e4m3"]

    source = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root,
        layer,
        args.roles,
        args.max_samples,
        data_role=args.data_role,
        sample_offset=args.sample_offset,
        sampling_strategy=args.sampling_strategy,
    )
    hidden, route = capture.samples(expert)
    hidden, route = hidden.to(args.device).float(), route.to(args.device)
    prefix = source.expert_prefix(layer, expert).split(f"layers.{layer}.")[0]
    base = f"{prefix}layers.{layer}.mlp.experts.{expert}"
    weights = {
        projection: source.get(f"{base}.{projection}.weight").to(args.device).float()
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    middle = (
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, weights["gate_proj"]).clamp(max=10.0))
        * torch.nn.functional.linear(hidden, weights["up_proj"]).clamp(-10.0, 10.0)
    )
    candidate = {}
    scalar = {}
    for projection, weight in weights.items():
        stem = f"{base}.{projection}"
        candidate[projection] = decode_trellis_mxf(
            frozen[f"{stem}.trellis"],
            codebook,
            frozen[f"{stem}.scale_ue8m0"],
            bits=bits,
            block_size=block_size,
            rows=weight.shape[0],
            width=weight.shape[1],
            device=args.device,
        )
        scalar[projection] = quantize_scalar_mxf(
            weight, alphabet=alphabet, block_size=block_size
        )[0]
    gate_up_scale = refine_global_scale(weights["gate_proj"], weights["up_proj"], search_grid=12, iterations=2)
    down_scale = refine_global_scale(weights["down_proj"], search_grid=12, iterations=2)
    nv_gate, nv_up = refine_rtn_nvfp4_group(
        [weights["gate_proj"], weights["up_proj"]], global_scale=gate_up_scale, scale_refinement_iterations=2
    )
    nv_down = refine_rtn_nvfp4_group(
        [weights["down_proj"]], global_scale=down_scale, scale_refinement_iterations=2
    )[0]
    variants = {
        "trellis": candidate,
        "matched-nvfp4": {
            "gate_proj": dequantize(nv_gate).to(args.device),
            "up_proj": dequantize(nv_up).to(args.device),
            "down_proj": dequantize(nv_down).to(args.device),
        },
        "scalar-same-alphabet-upper-bound": scalar,
    }
    rows = []
    aggregate = {}
    for variant, qweights in variants.items():
        for projection, inputs in (("gate_proj", hidden), ("up_proj", hidden), ("down_proj", middle)):
            rows.append({"variant": variant, "projection": projection, **_metric(weights[projection], qweights[projection], inputs, route)})
        metric = _full_expert_metric(
            weights["gate_proj"], weights["up_proj"], weights["down_proj"],
            qweights["gate_proj"], qweights["up_proj"], qweights["down_proj"], hidden, route,
        )
        aggregate[variant] = {**metric, "nmse": metric["error"] / metric["energy"]}
    control_error = aggregate["matched-nvfp4"]["error"]
    effect = (control_error - aggregate["trellis"]["error"]) / control_error * 100
    aggregate["trellis"]["relative_error_reduction_vs_matched_nvfp4_percent"] = effect
    result = {
        "schema": "glm53-trellis-mxf-frozen-evaluation.v1",
        "data_role": args.data_role,
        "candidate_adaptation_on_evaluation_role": False,
        "artifact": {"path": str(args.artifact), "bytes": args.artifact.stat().st_size, "sha256": sha256_file(args.artifact)},
        "layer": layer,
        "expert": expert,
        "bits": bits,
        "alphabet": alphabet,
        "block_size": block_size,
        "stored_bpw": float(metadata["stored_bpw"]),
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "sampling_strategy": args.sampling_strategy,
        "aggregate": aggregate,
        "primary_effect_percent": effect,
        "uncertainty": _bootstrap_experts(
            [
                {
                    "candidate_error": aggregate["trellis"]["error"],
                    "control_error": aggregate["matched-nvfp4"]["error"],
                    "energy": aggregate["matched-nvfp4"]["energy"],
                }
            ]
        ),
        "practical_gate_percent": args.practical_gate_percent,
        "decision": "pass" if effect >= args.practical_gate_percent else "fail",
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "source_index_sha256": sha256_file(args.source_index),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "effect_percent": effect, "decision": result["decision"], "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
