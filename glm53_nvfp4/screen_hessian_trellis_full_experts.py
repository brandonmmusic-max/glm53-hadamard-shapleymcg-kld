"""Causal gate/up/down P8 pseudoquant screen over the frozen expert panel."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .block_gptq import full_gptq_quantize
from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import output_nmse, route_weighted_hessian
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import quantize_trellis_mxf_gptq


def _middle(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    quantize_hidden: bool,
    quantize_output: bool,
) -> torch.Tensor:
    x = _qdq_e4m3_k32(hidden, 1.0, "amax") if quantize_hidden else hidden
    gate_output = F.linear(x.float(), gate.float()).clamp(max=10.0)
    up_output = F.linear(x.float(), up.float()).clamp(-10.0, 10.0)
    result = F.silu(gate_output) * up_output
    return _qdq_e4m3_k32(result, 1.0, "amax") if quantize_output else result


def _full_nmse(
    source: torch.Tensor,
    candidate: torch.Tensor,
    route_weights: torch.Tensor,
) -> float:
    weights = route_weights.float().square()
    weights /= weights.sum().clamp_min(1e-30)
    numerator = ((candidate.float() - source.float()).double().square().mean(1) * weights.double()).sum()
    denominator = (source.float().double().square().mean(1) * weights.double()).sum()
    return float((numerator / denominator.clamp_min(1e-30)).item())


def _trellis(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    t12_codebook_e4m3: torch.Tensor | None = None,
) -> torch.Tensor:
    return quantize_trellis_mxf_gptq(
        weight,
        hessian,
        bits=4,
        alphabet="e4m3",
        law="mcg",
        compander_scale=2.0,
        block_size=32,
        scale_refinement_iterations=2,
        t12_codebook_e4m3=t12_codebook_e4m3,
    ).reconstruction


def _gptq(weight: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return dequantize(full_gptq_quantize(weight, hessian, group_size=16)).to(weight.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t12-codebook", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    t12_codebook = None
    if args.t12_codebook is not None:
        bundle = load_file(str(args.t12_codebook), device="cpu")
        if set(bundle) != {"codebook_t12_e4m3"}:
            raise ValueError("learned law file must contain only codebook_t12_e4m3")
        t12_codebook = bundle["codebook_t12_e4m3"]
    candidate_variant = (
        "learned-t12-k4" if t12_codebook is not None else "hessian-mcg-k4"
    )
    fit_spec = plan["sampling"]["hessian_fit"]
    eval_spec = plan["sampling"]["evaluation"]
    fit_capture = LayerCapture(
        args.capture_root, plan["layer"], args.roles,
        max_samples=fit_spec["count"], sample_offset=fit_spec["offset"],
        data_role="fit", sampling_strategy="domain-balanced",
    )
    eval_capture = LayerCapture(
        args.capture_root, plan["layer"], args.roles,
        max_samples=eval_spec["count"], sample_offset=eval_spec["offset"],
        data_role="fit", sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(
        f"layers.{plan['layer']}."
    )[0]
    rows = []
    started = time.time()
    for expert in plan["experts"]:
        fit_hidden_cpu, fit_route_cpu = fit_capture.samples(expert)
        eval_hidden_cpu, eval_route_cpu = eval_capture.samples(expert)
        fit_hidden = fit_hidden_cpu.to(device)
        fit_route = fit_route_cpu.to(device)
        eval_hidden = eval_hidden_cpu.to(device)
        eval_route = eval_route_cpu.to(device)
        hidden_hessian = route_weighted_hessian(
            _qdq_e4m3_k32(fit_hidden, 1.0, "amax"), fit_route
        )
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        weights = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        variants = {}
        encoders = [
            (
                candidate_variant,
                lambda weight, hessian: _trellis(
                    weight, hessian, t12_codebook
                ),
            ),
            ("gptq-nvfp4", _gptq),
        ]
        if t12_codebook is not None:
            encoders.append(("hessian-mcg-k4", _trellis))
        for variant, encoder in encoders:
            qgate = encoder(weights["gate_proj"], hidden_hessian)
            qup = encoder(weights["up_proj"], hidden_hessian)
            fit_middle = _middle(
                fit_hidden, qgate, qup, quantize_hidden=True, quantize_output=True
            )
            down_hessian = route_weighted_hessian(fit_middle, fit_route)
            qdown = encoder(weights["down_proj"], down_hessian)
            variants[variant] = {
                "gate_proj": qgate,
                "up_proj": qup,
                "down_proj": qdown,
            }

        fit_reference_middle = _middle(
            fit_hidden, weights["gate_proj"], weights["up_proj"],
            quantize_hidden=False, quantize_output=False,
        )
        eval_reference_middle = _middle(
            eval_hidden, weights["gate_proj"], weights["up_proj"],
            quantize_hidden=False, quantize_output=False,
        )
        reference = F.linear(eval_reference_middle, weights["down_proj"])
        for variant, quantized in variants.items():
            fit_candidate_middle = _middle(
                fit_hidden, quantized["gate_proj"], quantized["up_proj"],
                quantize_hidden=True, quantize_output=True,
            )
            eval_candidate_middle = _middle(
                eval_hidden, quantized["gate_proj"], quantized["up_proj"],
                quantize_hidden=True, quantize_output=True,
            )
            actual = F.linear(
                eval_candidate_middle.float(), quantized["down_proj"].float()
            )
            rows.append({
                "expert": expert,
                "variant": variant,
                "physical_bpw": 4.5 if variant == "gptq-nvfp4" else 4.25,
                "charged_bpw": 4.5,
                "full_expert_evaluation_nmse": _full_nmse(reference, actual, eval_route),
                "projection_weight_nmse": {
                    projection: float(
                        ((quantized[projection] - weights[projection]).double().square().sum()
                        / weights[projection].double().square().sum()).item()
                    )
                    for projection in weights
                },
                "down_fit_output_nmse": output_nmse(
                    weights["down_proj"], quantized["down_proj"],
                    fit_reference_middle, fit_candidate_middle, fit_route,
                ),
                "down_evaluation_output_nmse": output_nmse(
                    weights["down_proj"], quantized["down_proj"],
                    eval_reference_middle, eval_candidate_middle, eval_route,
                ),
            })
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del weights, variants, hidden_hessian, fit_hidden, fit_route, eval_hidden, eval_route
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-hessian-trellis-full-16expert-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "rows": rows,
        "t12_codebook": (
            {"path": str(args.t12_codebook), "sha256": sha256_file(args.t12_codebook)}
            if args.t12_codebook is not None
            else None
        ),
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
