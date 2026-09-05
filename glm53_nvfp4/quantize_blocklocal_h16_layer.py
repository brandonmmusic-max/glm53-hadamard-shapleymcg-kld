"""Materialize per-expert, per-projection, per-K16 structured-H16 NVFP4.

The physical payload stores ``Q(W R_b)`` and one uint8 bank index for every
physical input block.  A separate BF16 chunk stores the algebraically exact
pseudoquant map ``Q(W R_b) R_b.T``.  The BF16 chunk is for end-to-end KLD
linkage only; it is not a compact or native-NVFP4 serving representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .block_gptq import full_gptq_quantize, refine_global_scale
from .block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
    hadamard16,
    structured_hadamard16,
)
from .capture import LayerCapture
from .modelopt import PackedNVFP4, dequantize
from .output_aware import route_weighted_hessian
from .screen_blocklocal_h16 import _select_blocks
from .shard_index import IndexedCheckpoint, sha256_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def descriptor_bank(
    count: int, device: torch.device, *, include_identity: bool = False
) -> list[torch.Tensor]:
    if not 1 <= count <= 256:
        raise ValueError("descriptor bank must contain 1..256 transforms")
    if include_identity:
        if count < 2:
            raise ValueError("identity-inclusive bank requires at least two transforms")
        return [torch.eye(16, device=device), hadamard16(device=device)] + [
            structured_hadamard16(f"bank-v1:{index}", device=device)
            for index in range(2, count)
        ]
    return [hadamard16(device=device)] + [
        structured_hadamard16(f"bank-v1:{index}", device=device)
        for index in range(1, count)
    ]


def block_diagonal_hessian(
    samples: torch.Tensor, route: torch.Tensor
) -> torch.Tensor:
    full = route_weighted_hessian(samples, route)
    if full.shape[0] % 16:
        raise ValueError("carrier width must be divisible by 16")
    blocks = full.reshape(full.shape[0] // 16, 16, full.shape[1] // 16, 16)
    return torch.stack([blocks[index, :, index, :] for index in range(blocks.shape[0])])


def quantize_projection(
    weight: torch.Tensor,
    carrier: torch.Tensor,
    route: torch.Tensor,
    rotation: torch.Tensor,
    global_scale: torch.Tensor,
    search_grid: int,
) -> tuple[PackedNVFP4, torch.Tensor]:
    transformed = apply_weight_rotation(weight, rotation)
    rotated_carrier = apply_activation_rotation(carrier, rotation)
    packed = full_gptq_quantize(
        transformed,
        route_weighted_hessian(rotated_carrier, route),
        global_scale=global_scale,
        search_grid=search_grid,
    )
    effective = apply_weight_rotation(
        dequantize(packed).to(weight.device), rotation.transpose(-1, -2)
    )
    return packed, effective


def _middle(hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_value = F.linear(hidden.float(), gate.float()).clamp(max=10.0)
    up_value = F.linear(hidden.float(), up.float()).clamp(-10.0, 10.0)
    return F.silu(gate_value) * up_value


def _nmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(
        (
            (actual.float() - reference.float()).double().square().sum()
            / reference.float().double().square().sum().clamp_min(1e-30)
        ).item()
    )


def _payload_tensors(
    base: str,
    projection: str,
    packed: PackedNVFP4,
    indexes: list[int],
) -> dict[str, torch.Tensor]:
    index_tensor = torch.tensor(indexes, dtype=torch.uint8)
    if index_tensor.numel() != packed.weight_scale.shape[1]:
        raise RuntimeError(f"descriptor/block mismatch for {base}.{projection}")
    return {
        f"{base}.{projection}.weight": packed.weight.contiguous(),
        f"{base}.{projection}.weight_scale": packed.weight_scale.contiguous(),
        f"{base}.{projection}.weight_scale_2": packed.weight_scale_2.contiguous(),
        f"{base}.{projection}.rotation_index": index_tensor.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--dense-output", type=Path, required=True)
    parser.add_argument("--physical-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.dense_output, args.physical_output, args.receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    plan = json.loads(args.plan.read_text())
    if plan.get("schema") not in {
        "glm53-rotation-v6.blocklocal-h16-layer3-plan.v1",
        "glm53-rotation-v7.selective-h16-layer3-plan.v1",
    }:
        raise RuntimeError("unexpected materialization plan schema")
    if plan.get("algorithm_exclusion") != "LDLQ and BlockLDLQ are excluded":
        raise RuntimeError("plan does not preserve the no-LDLQ boundary")
    if plan.get("layer") != 3 or not (
        0 <= args.expert_start < args.expert_end <= plan["experts"]
    ):
        raise ValueError("requested expert range is outside the frozen layer-3 plan")
    expected = plan["inputs"]
    checks = {
        "source_index": args.source_index,
        "capture_manifest": args.capture_root / "capture-manifest.json",
        "roles": args.roles,
    }
    for name, path in checks.items():
        if sha256_file(path) != expected[name]["sha256"]:
            raise RuntimeError(f"{name} hash differs from frozen plan")
    if sha256_file(Path(__file__)) != expected["materializer"]["sha256"]:
        raise RuntimeError("materializer source hash differs from frozen plan")
    selector_path = Path(__file__).with_name("screen_blocklocal_h16.py")
    if sha256_file(selector_path) != expected["selector"]["sha256"]:
        raise RuntimeError("selector source hash differs from frozen plan")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    fit = plan["calibration"]
    capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=fit["count"],
        sample_offset=fit["offset"],
        data_role=fit["role"],
        sampling_strategy=fit["strategy"],
    )
    include_identity = bool(plan["candidate"].get("include_identity", False))
    bank = descriptor_bank(
        plan["candidate"]["variants_per_block"],
        device,
        include_identity=include_identity,
    )
    prefix = source.expert_prefix(plan["layer"], args.expert_start).split(
        f"layers.{plan['layer']}."
    )[0]
    dense_tensors: dict[str, torch.Tensor] = {}
    physical_tensors: dict[str, torch.Tensor] = {}
    metrics: list[dict[str, object]] = []
    logical_elements = 0
    descriptor_bytes = 0
    started = time.time()

    for expert in range(args.expert_start, args.expert_end):
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device).float()
        route = route_cpu.to(device).float()
        base = f"{prefix}layers.3.mlp.experts.{expert}"
        weights = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in PROJECTIONS
        }

        hidden_blocks = block_diagonal_hessian(hidden, route)
        fixed = bank[0]
        shared_scale = refine_global_scale(
            apply_weight_rotation(weights["gate_proj"], fixed),
            apply_weight_rotation(weights["up_proj"], fixed),
            search_grid=plan["candidate"]["search_grid"],
            iterations=2,
        )
        selected: dict[str, torch.Tensor] = {}
        selected_indexes: dict[str, list[int]] = {}
        local_gains: dict[str, list[float]] = {}
        for _ in range(plan["candidate"]["global_scale_refits"]):
            for projection in ("gate_proj", "up_proj"):
                rotation, indexes, gains = _select_blocks(
                    weights[projection],
                    hidden_blocks,
                    shared_scale,
                    bank,
                    plan["candidate"]["search_grid"],
                    plan["candidate"]["selection_method"],
                )
                selected[projection] = rotation
                selected_indexes[projection] = indexes
                local_gains[projection] = gains
            shared_scale = refine_global_scale(
                apply_weight_rotation(weights["gate_proj"], selected["gate_proj"]),
                apply_weight_rotation(weights["up_proj"], selected["up_proj"]),
                search_grid=plan["candidate"]["search_grid"],
                iterations=2,
            )

        effective: dict[str, torch.Tensor] = {}
        for projection in ("gate_proj", "up_proj"):
            packed, dense = quantize_projection(
                weights[projection],
                hidden,
                route,
                selected[projection],
                shared_scale,
                plan["candidate"]["search_grid"],
            )
            effective[projection] = dense
            dense_tensors[f"{base}.{projection}.weight"] = dense.to(torch.bfloat16).cpu().contiguous()
            physical_tensors.update(
                _payload_tensors(base, projection, packed, selected_indexes[projection])
            )

        candidate_middle = _middle(
            hidden, effective["gate_proj"], effective["up_proj"]
        )
        middle_blocks = block_diagonal_hessian(candidate_middle, route)
        down_scale = refine_global_scale(
            apply_weight_rotation(weights["down_proj"], fixed),
            search_grid=plan["candidate"]["search_grid"],
            iterations=2,
        )
        down_rotation, down_indexes, down_gains = _select_blocks(
            weights["down_proj"],
            middle_blocks,
            down_scale,
            bank,
            plan["candidate"]["search_grid"],
            plan["candidate"]["selection_method"],
        )
        down_scale = refine_global_scale(
            apply_weight_rotation(weights["down_proj"], down_rotation),
            search_grid=plan["candidate"]["search_grid"],
            iterations=2,
        )
        down_packed, down_dense = quantize_projection(
            weights["down_proj"],
            candidate_middle,
            route,
            down_rotation,
            down_scale,
            plan["candidate"]["search_grid"],
        )
        selected["down_proj"] = down_rotation
        selected_indexes["down_proj"] = down_indexes
        local_gains["down_proj"] = down_gains
        effective["down_proj"] = down_dense
        dense_tensors[f"{base}.down_proj.weight"] = down_dense.to(torch.bfloat16).cpu().contiguous()
        physical_tensors.update(
            _payload_tensors(base, "down_proj", down_packed, down_indexes)
        )

        reference_output = F.linear(
            _middle(hidden, weights["gate_proj"], weights["up_proj"]),
            weights["down_proj"],
        )
        actual_output = F.linear(candidate_middle, down_dense)
        projection_metrics = {
            projection: {
                "effective_weight_nmse": _nmse(weights[projection], effective[projection]),
                "mean_local_gain_vs_fixed": float(
                    torch.tensor(local_gains[projection], dtype=torch.float64).mean()
                ),
                "nonzero_descriptor_fraction": float(
                    (torch.tensor(selected_indexes[projection]) != 0).float().mean()
                ),
            }
            for projection in PROJECTIONS
        }
        metrics.append(
            {
                "expert": expert,
                "fit_full_expert_output_nmse": _nmse(reference_output, actual_output),
                "projections": projection_metrics,
            }
        )
        for projection in PROJECTIONS:
            logical_elements += weights[projection].numel()
            descriptor_bytes += len(selected_indexes[projection])
        print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del hidden, route, weights, selected, effective, candidate_middle
        del reference_output, actual_output, hidden_blocks, middle_blocks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    common = {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-chunk.v1",
        "layer": "3",
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "transform": "per-expert-per-projection-per-K16 D-P-H16",
        "descriptor_bank": (
            "identity plus procedural structured_hadamard16 bank-v1, 255 entries"
            if include_identity
            else "procedural structured_hadamard16 bank-v1, 256 entries"
        ),
        "format": "pt",
        "weight_format": "ModelOpt NVFP4 E2M1/E4M3-per-16/FP32-global",
        "ldlq": "false",
    }
    args.dense_output.parent.mkdir(parents=True, exist_ok=True)
    args.physical_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dense_tensors,
        str(args.dense_output),
        metadata={**common, "role": "inverse-rotated-BF16-pseudoquant"},
    )
    save_file(
        physical_tensors,
        str(args.physical_output),
        metadata={**common, "role": "physical-packed-NVFP4-plus-transform-index"},
    )
    packed_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in physical_tensors.items()
        if name.endswith(".weight")
    )
    scale_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in physical_tensors.items()
        if name.endswith((".weight_scale", ".weight_scale_2"))
    )
    physical_bytes = packed_weight_bytes + scale_bytes + descriptor_bytes
    receipt = {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-chunk-receipt.v1",
        "plan_sha256": sha256_file(args.plan),
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "source_code": {
            "materializer_sha256": sha256_file(Path(__file__)),
            "screen_selector_sha256": sha256_file(
                Path(__file__).with_name("screen_blocklocal_h16.py")
            ),
        },
        "layer": 3,
        "expert_range": [args.expert_start, args.expert_end],
        "logical_elements": logical_elements,
        "physical_bytes": physical_bytes,
        "physical_bpw": 8.0 * physical_bytes / logical_elements,
        "descriptor_bytes": descriptor_bytes,
        "descriptor_bank_runtime_bytes": 0,
        "descriptor_law_sha256": hashlib.sha256(
            (
                b"identity+structured_hadamard16:bank-v1"
                if include_identity
                else b"structured_hadamard16:bank-v1"
            )
        ).hexdigest(),
        "include_identity": include_identity,
        "metrics": metrics,
        "outputs": {
            "dense": {
                "path": str(args.dense_output.resolve()),
                "bytes": args.dense_output.stat().st_size,
                "sha256": sha256_file(args.dense_output),
            },
            "physical": {
                "path": str(args.physical_output.resolve()),
                "bytes": args.physical_output.stat().st_size,
                "sha256": sha256_file(args.physical_output),
            },
        },
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "runtime_status": (
            "physical codec is materialized but projection-specific transform prologue "
            "is not implemented; dense output is pseudoquant linkage only"
        ),
        "protected_roles_opened": [],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "physical_bpw": receipt["physical_bpw"]}, sort_keys=True))


if __name__ == "__main__":
    main()
