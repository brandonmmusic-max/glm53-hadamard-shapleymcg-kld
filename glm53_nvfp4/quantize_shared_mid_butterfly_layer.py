"""Encode one GLM layer's down weights in a shared learned SO(16) basis.

Only the down-projection ModelOpt tensors are written.  Gate/up remain the
carrier's native NVFP4 tensors, while the Humming runtime applies the shared
middle rotation after SwiGLU and before W2 input quantization.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .block_gptq import compare_full_packed_to_rtn, full_gptq_quantize, full_hessian, refine_global_scale
from .block_rotation import apply_activation_rotation, apply_weight_rotation, butterfly16, load_layer_rotation, orthogonality_error
from .capture import LayerCapture
from .modelopt import dequantize
from .shard_index import IndexedCheckpoint, sha256_file


def _relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    numerator = (actual.double() - reference.double()).square().sum()
    denominator = reference.double().square().sum().clamp_min(1e-30)
    return float(torch.sqrt(numerator / denominator))


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_value = F.linear(hidden.float(), gate.float()).clamp(max=10.0)
    up_value = F.linear(hidden.float(), up.float()).clamp(-10.0, 10.0)
    return F.silu(gate_value) * up_value


def _rotation_metadata(path: Path, layer: int) -> tuple[float, dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != {f"layer_{layer:03d}_mid"}:
            raise ValueError("rotation file must contain exactly one layer mid matrix")
    if metadata.get("schema") != "glm53-rotation-v8.shared-butterfly16.v1":
        raise ValueError("unexpected rotation schema")
    if metadata.get("layer") != str(layer) or metadata.get("scope") != "mid-only":
        raise ValueError("rotation metadata does not match layer/scope")
    return float(metadata["angle_pi"]), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--rotation-file", type=Path, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "glm53-rotation-v8.shared-mid-butterfly-search-plan.v1":
        raise RuntimeError("unexpected plan schema")
    if plan.get("algorithm_exclusion") != "LDLQ and BlockLDLQ are excluded":
        raise RuntimeError("no-LDLQ boundary is absent")
    layer = int(plan["layer"])
    if not 0 <= args.expert_start < args.expert_end <= int(plan["experts"]):
        raise ValueError("expert range is outside plan")
    inputs = plan["inputs"]
    checks = {
        "source_index": args.source_index,
        "capture_manifest": args.capture_root / "capture-manifest.json",
        "calibration_roles": args.roles,
        "materializer": Path(__file__),
    }
    for name, path in checks.items():
        if sha256_file(path) != inputs[name]["sha256"]:
            raise RuntimeError(f"{name} hash differs from plan")

    angle_pi, _ = _rotation_metadata(args.rotation_file, layer)
    allowed = {float(item["angle_pi"]) for item in plan["search"]["coarse_grid"]}
    if angle_pi not in allowed:
        raise ValueError(f"rotation angle {angle_pi} is outside sealed coarse grid")
    rotation = load_layer_rotation(
        args.rotation_file, layer, kind="mid", width=2048, device=args.device
    )
    expected_rotation = butterfly16(torch.pi * angle_pi, device=args.device)
    if not torch.equal(rotation, expected_rotation):
        raise RuntimeError("rotation payload is not the deterministic sealed butterfly")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("quantization requires CUDA")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    calibration = plan["calibration"]
    capture = LayerCapture(
        args.capture_root,
        layer,
        args.roles,
        max_samples=int(calibration["samples_per_expert"]),
        sampling_strategy=calibration["sampling_strategy"],
        data_role=calibration["role"],
    )
    prefix = checkpoint.expert_prefix(layer, args.expert_start).split(
        f"layers.{layer}."
    )[0]
    output: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict] = {}
    source_files: set[str] = set()
    logical_elements = 0
    started = time.time()

    for expert in range(args.expert_start, args.expert_end):
        base = f"{prefix}layers.{layer}.mlp.experts.{expert}"
        names = {
            projection: f"{base}.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        source_files.update(checkpoint.weight_map[name] for name in names.values())
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device)
        route = route_cpu.to(device).float()
        gate = checkpoint.get(names["gate_proj"]).to(device)
        up = checkpoint.get(names["up_proj"]).to(device)
        down = checkpoint.get(names["down_proj"]).to(device)
        middle = _middle(hidden, gate, up)
        middle_basis = apply_activation_rotation(middle, rotation)
        down_basis = apply_weight_rotation(down, rotation)
        hessian = full_hessian(middle_basis, route)
        global_scale = refine_global_scale(
            down_basis,
            search_grid=int(plan["encoder"]["search_grid"]),
            iterations=int(plan["encoder"]["global_scale_refits"]),
        )
        packed = full_gptq_quantize(
            down_basis,
            hessian,
            global_scale=global_scale,
            search_grid=int(plan["encoder"]["search_grid"]),
        )
        stem = f"{base}.down_proj"
        output[f"{stem}.weight"] = packed.weight.cpu().contiguous()
        output[f"{stem}.weight_scale"] = packed.weight_scale.cpu().contiguous()
        output[f"{stem}.weight_scale_2"] = packed.weight_scale_2.cpu().contiguous()

        decoded = dequantize(packed).to(device).float()
        middle_bf16 = middle.to(torch.bfloat16)
        native_input = apply_activation_rotation(
            middle_bf16, rotation.to(torch.bfloat16)
        ).float()
        native_output = F.linear(native_input, decoded)
        effective = apply_weight_rotation(decoded, rotation.transpose(-1, -2))
        overlay_output = F.linear(middle_bf16.float(), effective)
        reference_output = F.linear(middle.float(), down.float())
        comparison = compare_full_packed_to_rtn(
            down_basis,
            hessian,
            packed,
            packed.weight_scale_2,
            int(plan["encoder"]["search_grid"]),
            candidate_method="shared-mid-butterfly-full-gptq",
        )
        metrics[str(expert)] = {
            "samples": int(hidden.shape[0]),
            "down": comparison,
            "full_expert_output_relative_l2": _relative_l2(
                reference_output, native_output
            ),
            "runtime_bf16_vs_fp32_effective_relative_l2": _relative_l2(
                overlay_output, native_output
            ),
        }
        logical_elements += int(down.numel())
        print(
            json.dumps(
                {
                    "layer": layer,
                    "expert": expert,
                    "ratio": comparison["ratio"],
                    "runtime_closure": metrics[str(expert)][
                        "runtime_bf16_vs_fp32_effective_relative_l2"
                    ],
                }
            ),
            flush=True,
        )
        del hidden, route, gate, up, down, middle, middle_basis, down_basis
        del hessian, packed, decoded, native_input, native_output, effective
        del overlay_output, reference_output
        torch.cuda.empty_cache()

    metadata = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-down-chunk.v1",
        "format": "pt",
        "layer": str(layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "angle_pi": format(angle_pi, ".17g"),
        "rotation_scope": "mid-only",
        "quantizer": "full-Hessian GPTQ ModelOpt NVFP4",
        "ldlq": "false",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.output), metadata=metadata)
    weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in output.items()
        if name.endswith(".weight")
    )
    scale_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in output.items()
        if name.endswith((".weight_scale", ".weight_scale_2"))
    )
    receipt = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-down-chunk-receipt.v1",
        "plan_sha256": sha256_file(args.plan),
        "layer": layer,
        "expert_range": [args.expert_start, args.expert_end],
        "angle_pi": angle_pi,
        "rotation": {
            "path": str(args.rotation_file.resolve()),
            "sha256": sha256_file(args.rotation_file),
            "table_bytes": int(rotation.numel() * rotation.element_size()),
            "fp32_gram_max_abs": orthogonality_error(rotation),
            "runtime_bf16_gram_max_abs": orthogonality_error(
                rotation.to(torch.bfloat16).float()
            ),
        },
        "algorithm": {
            "quant_method": "gptq",
            "gptq_geometry": "full",
            "rotation_scope": "mid-only",
        },
        "encoder": plan["encoder"],
        "logical_elements": logical_elements,
        "physical": {
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "payload_bytes": weight_bytes + scale_bytes,
            "payload_bpw": 8.0 * (weight_bytes + scale_bytes) / logical_elements,
        },
        "metrics": metrics,
        "source_files": [
            {"path": name, "sha256": sha256_file(args.source / name)}
            for name in sorted(source_files)
        ],
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in checks.items()
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensors": len(output),
        },
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "protected_roles_opened": [],
        "ldlq": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": receipt["output"], "physical": receipt["physical"]}, sort_keys=True))


if __name__ == "__main__":
    main()
