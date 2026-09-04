"""Screen scaled-H128 procedural-MCG K4 against GPTQ NVFP4 experts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_gptq import full_gptq_quantize
from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .modelopt import dequantize
from .output_aware import route_weighted_hessian
from .p8_h128 import effective_uncoupled_weights, hadamard128_last, transform_uncoupled_weights
from .screen_p8_scaled_h128 import _diagonal, _middle
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, pack_ue8m0, quantize_trellis_mxf_gptq


def _weight_nmse(source: torch.Tensor, actual: torch.Tensor) -> float:
    return float(((actual.float() - source.float()).double().square().sum() / source.float().double().square().sum().clamp_min(1e-30)).item())


def _output_nmse(reference: torch.Tensor, actual: torch.Tensor, route: torch.Tensor) -> float:
    error = (actual.float() - reference.float()) * route[:, None]
    target = reference.float() * route[:, None]
    return float((error.double().square().sum() / target.double().square().sum().clamp_min(1e-30)).item())


def _trellis(weight: torch.Tensor, hessian: torch.Tensor):
    return quantize_trellis_mxf_gptq(weight, hessian, bits=4, alphabet="e4m3", law="mcg", compander_scale=2.0, block_size=32, scale_refinement_iterations=2)


def _candidate_forward(hidden: torch.Tensor, weights: dict[str, torch.Tensor], diagonal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate = hadamard128_last(F.linear(carrier, weights["gate_proj"]))
    up = hadamard128_last(F.linear(carrier, weights["up_proj"]))
    gate = gate.clamp(max=10.0)
    up = up.clamp(-10.0, 10.0)
    middle_rotated = hadamard128_last(F.silu(gate) * up * diagonal)
    middle_carrier = _qdq_e4m3_k32(middle_rotated, 1.0, "amax").float()
    return F.linear(middle_carrier, weights["down_proj"]), middle_carrier


def _control_forward(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate = F.linear(carrier, weights["gate_proj"]).clamp(max=10.0)
    up = F.linear(carrier, weights["up_proj"]).clamp(-10.0, 10.0)
    middle = _qdq_e4m3_k32(F.silu(gate) * up, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"]), middle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    experts = [int(value) for value in args.experts.split(",")]
    if not experts or not set(experts) <= set(plan["experts"]):
        raise ValueError("requested experts are outside the frozen plan")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    captures = {
        name: LayerCapture(args.capture_root, plan["layer"], args.roles, max_samples=spec["count"], sample_offset=spec["offset"], data_role=spec["role"], sampling_strategy=spec["strategy"])
        for name, spec in plan["sampling"].items()
    }
    prefix = source.expert_prefix(plan["layer"], experts[0]).split(f"layers.{plan['layer']}.")[0]
    rows = []
    started = time.time()
    for expert in experts:
        base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
        original = {p: source.get(f"{base}.{p}.weight").to(device).float() for p in ("gate_proj", "up_proj", "down_proj")}
        cal_hidden_cpu, cal_route_cpu = captures["calibration"].samples(expert)
        cal_hidden = cal_hidden_cpu.to(device).float()
        cal_route = cal_route_cpu.to(device).float()
        cal_carrier = _qdq_e4m3_k32(cal_hidden, 1.0, "amax").float()
        cal_middle = _middle(cal_carrier, original["gate_proj"], original["up_proj"])
        diagonal = _diagonal(
            "balance", 0.5,
            cal_middle.double().square().mean(0).sqrt().float(),
            original["down_proj"].double().square().mean(0).sqrt().float(),
            (0.25, 4.0),
        )
        transformed_tuple = transform_uncoupled_weights(original["gate_proj"], original["up_proj"], original["down_proj"], down_scale=diagonal)
        transformed = dict(zip(("gate_proj", "up_proj", "down_proj"), transformed_tuple))
        hidden_hessian = route_weighted_hessian(cal_carrier, cal_route)
        candidate_payload = {
            "gate_proj": _trellis(transformed["gate_proj"], hidden_hessian),
            "up_proj": _trellis(transformed["up_proj"], hidden_hessian),
        }
        candidate_weights = {p: candidate_payload[p].reconstruction for p in ("gate_proj", "up_proj")}
        _, candidate_middle = _candidate_forward(cal_hidden, {**candidate_weights, "down_proj": transformed["down_proj"]}, diagonal)
        down_hessian = route_weighted_hessian(candidate_middle, cal_route)
        candidate_payload["down_proj"] = _trellis(transformed["down_proj"], down_hessian)
        candidate_weights["down_proj"] = candidate_payload["down_proj"].reconstruction
        control_payload = {
            "gate_proj": full_gptq_quantize(original["gate_proj"], hidden_hessian, group_size=16),
            "up_proj": full_gptq_quantize(original["up_proj"], hidden_hessian, group_size=16),
        }
        control_weights = {p: dequantize(control_payload[p]).to(device).float() for p in ("gate_proj", "up_proj")}
        _, control_middle = _control_forward(cal_hidden, {**control_weights, "down_proj": original["down_proj"]})
        control_payload["down_proj"] = full_gptq_quantize(original["down_proj"], route_weighted_hessian(control_middle, cal_route), group_size=16)
        control_weights["down_proj"] = dequantize(control_payload["down_proj"]).to(device).float()
        effective = dict(zip(("gate_proj", "up_proj", "down_proj"), effective_uncoupled_weights(candidate_weights["gate_proj"], candidate_weights["up_proj"], candidate_weights["down_proj"], down_scale=diagonal)))
        projection_metrics = {
            p: {
                "candidate_effective_weight_nmse": _weight_nmse(original[p], effective[p]),
                "control_weight_nmse": _weight_nmse(original[p], control_weights[p]),
            }
            for p in ("gate_proj", "up_proj", "down_proj")
        }
        bit_exact = True
        for projection, payload in candidate_payload.items():
            decoded = decode_trellis_mxf(payload.trellis, payload.codebook_e4m3, pack_ue8m0(payload.scales), bits=payload.bits, block_size=payload.block_size, rows=transformed[projection].shape[0], width=transformed[projection].shape[1], device=device)
            bit_exact = bit_exact and torch.equal(decoded, payload.reconstruction)
        eval_hidden_cpu, eval_route_cpu = captures["evaluation"].samples(expert)
        eval_hidden = eval_hidden_cpu.to(device).float()
        eval_route = eval_route_cpu.to(device).float()
        reference = F.linear(_middle(eval_hidden, original["gate_proj"], original["up_proj"]), original["down_proj"])
        candidate_output, _ = _candidate_forward(eval_hidden, candidate_weights, diagonal)
        control_output, _ = _control_forward(eval_hidden, control_weights)
        row = {
            "expert": expert,
            "projection_weight_nmse": projection_metrics,
            "candidate_full_output_nmse": _output_nmse(reference, candidate_output, eval_route),
            "control_full_output_nmse": _output_nmse(reference, control_output, eval_route),
            "candidate_codec_decode_bit_exact": bit_exact,
            "candidate_physical_bpw": plan["candidate"]["physical_bpw"],
            "control_physical_bpw": plan["control"]["physical_bpw"],
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del original, cal_hidden, cal_route, cal_carrier, cal_middle, diagonal, transformed, hidden_hessian, candidate_payload, candidate_weights, down_hessian, control_payload, control_weights, control_middle, effective, eval_hidden, eval_route, reference, candidate_output, control_output
        torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-screen-raw.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "experts": experts,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
