"""Build a rotated, physical P8 layer from REAP Hessian calibration.

Gate/up retain the ordinary K4 procedural-MCG P8 representation.  The
post-SwiGLU intermediate is transformed by one scalar-angle 16-lane
butterfly and W2 is encoded in the matching basis.  The dense output, when
requested, is the exact decoded *rotated-basis* payload expected by the P8
pseudoquant reference; it is not an inverse-rotated BF16 shortcut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .block_rotation import (
    apply_shared_butterfly16_staged,
    apply_weight_rotation,
    butterfly16,
    orthogonality_error,
)
from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .output_aware import route_weighted_hessian
from .quantize_hessian_trellis_layer import _quantize
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, pack_ue8m0


def _relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    numerator = (actual.double() - reference.double()).square().sum()
    denominator = reference.double().square().sum().clamp_min(1e-30)
    return float(torch.sqrt(numerator / denominator))


def _linear_float(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use the encoder's FP32 diagnostic path across mixed storage dtypes."""
    return F.linear(inputs.float(), weight.float())


def _prequant_middle(
    hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax")
    gate_output = F.linear(carrier.float(), gate.float()).clamp(max=10.0)
    up_output = F.linear(carrier.float(), up.float()).clamp(-10.0, 10.0)
    # The fused P8 implementations round the FC1 epilogue/SwiGLU result to
    # BF16 before reading it back for the intermediate quantizer.
    return (F.silu(gate_output) * up_output).to(torch.bfloat16)


def rotated_runtime_middle(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    angle_radians: float | torch.Tensor,
) -> torch.Tensor:
    """Reference the fused butterfly followed by native E4M3/UE8M0-K32."""
    middle_bf16 = _prequant_middle(hidden, gate, up)
    rotated_bf16 = apply_shared_butterfly16_staged(
        middle_bf16, angle_radians, output_dtype=torch.bfloat16
    )
    return _qdq_e4m3_k32(rotated_bf16.float(), 1.0, "amax")


def _load_rotation(path: Path, layer: int, expected_sha256: str) -> tuple[float, torch.Tensor]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("selected rotation payload differs from integration design")
    with safe_open(path, framework="pt", device="cpu") as src:
        metadata = src.metadata() or {}
        key = f"layer_{layer:03d}_mid"
        if (
            metadata.get("schema") != "glm53-rotation-v8.shared-butterfly16.v1"
            or metadata.get("layer") != str(layer)
            or metadata.get("scope") != "mid-only"
            or set(src.keys()) != {key}
        ):
            raise RuntimeError("invalid selected rotation payload")
        angle_pi = float(metadata["angle_pi"])
        matrix = src.get_tensor(key).float()
    canonical = butterfly16(math.pi * angle_pi)
    if not torch.equal(matrix, canonical):
        raise RuntimeError("rotation matrix is not the canonical scalar butterfly")
    return angle_pi, matrix


def _validate_design(args: argparse.Namespace) -> tuple[dict, dict, float, torch.Tensor]:
    design = json.loads(args.design.read_text())
    if (
        design.get("schema") != "glm53-trellismx-p8.shared-mid-butterfly-design.v1"
        or design.get("decision_before_result") is not True
        or design.get("ldlq") is not False
        or design.get("protected_roles_opened") != []
        or args.layer not in design.get("layers", [])
    ):
        raise RuntimeError("invalid rotated P8 integration design")
    base_path = Path(design["base_p8_design"]["path"])
    if sha256_file(base_path) != design["base_p8_design"]["sha256"]:
        raise RuntimeError("base P8 design drift")
    base = json.loads(base_path.read_text())
    if (
        base.get("schema") != "glm53-p8.direct-kld-native-shapley-design.v2"
        or base.get("ldlq") is not False
        or base.get("encoder_contract", {}).get("ldlq") is not False
    ):
        raise RuntimeError("rotated encoder requires the no-LDLQ P8 v2 base")
    code_inputs = {
        "encoder": Path(__file__),
        "block_rotation": Path(__file__).with_name("block_rotation.py"),
        "trellis_codec": Path(__file__).with_name("trellis_mxf.py"),
    }
    for name, path in code_inputs.items():
        pinned = design.get("inputs", {}).get(name, {})
        if pinned.get("sha256") != sha256_file(path):
            raise RuntimeError(f"{name} code differs from integration design")
    actual_inputs = {
        "fit_roles": args.roles,
        "source_index": args.source_index,
        "capture_manifest": args.capture_root / "capture-manifest.json",
    }
    for name, path in actual_inputs.items():
        if sha256_file(path) != base["encoder_inputs"][name]["sha256"]:
            raise RuntimeError(f"{name} differs from base P8 design")
    selected = design["rotation"]
    if selected.get("id") != "p00625" or float(selected.get("angle_pi")) != 0.0625:
        raise RuntimeError("integration design did not freeze the selected +pi/16 arm")
    angle_pi, matrix = _load_rotation(
        Path(selected["path"]), args.layer, selected["sha256"]
    )
    if angle_pi != float(selected["angle_pi"]):
        raise RuntimeError("rotation angle differs from integration design")
    return design, base, angle_pi, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--dense-output", type=Path)
    parser.add_argument("--codec-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    outputs = [args.codec_output, args.receipt]
    if args.dense_output is not None:
        outputs.append(args.dense_output)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not (0 <= args.expert_start < args.expert_end <= 288):
        raise ValueError("invalid expert range")
    design, base_design, angle_pi, rotation_cpu = _validate_design(args)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("rotated P8 encoding requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    rotation = rotation_cpu.to(device)
    angle_radians = math.pi * angle_pi
    started = time.time()
    source = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root,
        args.layer,
        args.roles,
        max_samples=args.samples,
        sample_offset=0,
        data_role="fit",
        sampling_strategy="domain-balanced",
    )
    prefix = source.expert_prefix(args.layer, args.expert_start).split(
        f"layers.{args.layer}."
    )[0]
    dense_tensors: dict[str, torch.Tensor] = {}
    codec_tensors: dict[str, torch.Tensor] = {}
    metrics = []
    codebook = None
    logical_elements = 0
    for expert in range(args.expert_start, args.expert_end):
        hidden_cpu, route_cpu = capture.samples(expert)
        hidden = hidden_cpu.to(device)
        route = route_cpu.to(device)
        carrier = _qdq_e4m3_k32(hidden, 1.0, "amax")
        hidden_hessian = route_weighted_hessian(carrier, route)
        base = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        weights = {
            projection: source.get(f"{base}.{projection}.weight").to(device).float()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        payloads = {
            "gate_proj": _quantize(weights["gate_proj"], hidden_hessian),
            "up_proj": _quantize(weights["up_proj"], hidden_hessian),
        }
        middle = rotated_runtime_middle(
            hidden,
            payloads["gate_proj"].reconstruction,
            payloads["up_proj"].reconstruction,
            angle_radians,
        )
        down_basis = apply_weight_rotation(weights["down_proj"], rotation)
        payloads["down_proj"] = _quantize(
            down_basis, route_weighted_hessian(middle, route)
        )
        for projection, payload in payloads.items():
            target = down_basis if projection == "down_proj" else weights[projection]
            scale_codes = pack_ue8m0(payload.scales)
            decoded = decode_trellis_mxf(
                payload.trellis,
                payload.codebook_e4m3,
                scale_codes,
                bits=payload.bits,
                block_size=payload.block_size,
                rows=target.shape[0],
                width=target.shape[1],
                device=device,
            )
            if not torch.equal(decoded, payload.reconstruction):
                raise RuntimeError(f"codec closure failed for {base}.{projection}")
            if args.dense_output is not None:
                dense_tensors[f"{base}.{projection}.weight"] = (
                    decoded.to(torch.bfloat16).cpu().contiguous()
                )
            codec_tensors[f"{base}.{projection}.trellis"] = payload.trellis
            codec_tensors[f"{base}.{projection}.scale_ue8m0"] = scale_codes
            if codebook is None:
                codebook = payload.codebook_e4m3
            elif not torch.equal(codebook, payload.codebook_e4m3):
                raise RuntimeError("procedural MCG codebook changed within chunk")
            delta = decoded - target
            metrics.append(
                {
                    "expert": expert,
                    "projection": projection,
                    "weight_nmse": float(
                        (delta.double().square().sum() / target.double().square().sum()).item()
                    ),
                }
            )
            logical_elements += target.numel()
        unrotated_middle = _qdq_e4m3_k32(
            _prequant_middle(
                hidden,
                payloads["gate_proj"].reconstruction,
                payloads["up_proj"].reconstruction,
            ).float(),
            1.0,
            "amax",
        )
        reference_output = _linear_float(unrotated_middle, weights["down_proj"])
        candidate_output = _linear_float(
            middle, payloads["down_proj"].reconstruction
        )
        metrics.append(
            {
                "expert": expert,
                "projection": "full_expert",
                "routed_output_relative_l2": _relative_l2(reference_output, candidate_output),
            }
        )
        if expert % 4 == 3:
            print(json.dumps({"expert": expert, "completed": True}), flush=True)
        del hidden, route, carrier, hidden_hessian, weights, payloads
        del middle, down_basis, unrotated_middle, reference_output, candidate_output
        torch.cuda.empty_cache()

    metadata = {
        "schema": "glm53-rotated-hessian-trellis-p8-layer-chunk.v1",
        "role": "physical-codec",
        "layer": str(args.layer),
        "expert_range": f"{args.expert_start}:{args.expert_end}",
        "bits": "4",
        "alphabet": "e4m3",
        "block_size": "32",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": "shared-mid-butterfly-p00625",
        "angle_pi": format(angle_pi, ".17g"),
        "rotation_arithmetic": "bf16-input-fp32-four-stage-final-bf16",
        "ldlq": "false",
        "encoder": "gptq-feedback-static-in-group-act-order",
        "design_sha256": sha256_file(args.design),
    }
    assert codebook is not None
    payload_bytes = sum(t.numel() * t.element_size() for t in codec_tensors.values())
    payload_bpw = payload_bytes * 8.0 / logical_elements
    if not math.isclose(payload_bpw, 4.25, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"physical P8 payload is {payload_bpw} bpw, expected 4.25")
    args.codec_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(codec_tensors, str(args.codec_output), metadata=metadata)
    if args.dense_output is not None:
        args.dense_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            dense_tensors,
            str(args.dense_output),
            metadata={**metadata, "role": "decoded-rotated-basis-pseudoquant-carrier"},
        )
    output_receipts = {
        "codec": {
            "path": str(args.codec_output.resolve()),
            "bytes": args.codec_output.stat().st_size,
            "sha256": sha256_file(args.codec_output),
        }
    }
    if args.dense_output is not None:
        output_receipts["dense"] = {
            "path": str(args.dense_output.resolve()),
            "bytes": args.dense_output.stat().st_size,
            "sha256": sha256_file(args.dense_output),
        }
    receipt = {
        "schema": "glm53-rotated-hessian-trellis-p8-layer-chunk-receipt.v1",
        "command": sys.argv,
        "design_sha256": sha256_file(args.design),
        "base_p8_design_sha256": sha256_file(Path(design["base_p8_design"]["path"])),
        "layer": args.layer,
        "expert_range": [args.expert_start, args.expert_end],
        "calibration": {
            "role": "fit",
            "sampling": "domain-balanced routes per expert",
            "samples": args.samples,
            "activation_carrier": "E4M3 K32 UE8M0 amax",
            "causal_down_hessian": True,
        },
        "rotation": {
            "id": "p00625",
            "angle_pi": angle_pi,
            "path": str(Path(design["rotation"]["path"]).resolve()),
            "sha256": design["rotation"]["sha256"],
            "fp32_gram_max_abs": orthogonality_error(rotation_cpu),
            "runtime": "post-SwiGLU BF16 load, four FP32 Givens stages, final BF16 round",
            "stored_table_bytes": 0,
        },
        "algorithm": {
            "bits": 4,
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0",
            "law": "procedural MCG",
            "alpha": 2.0,
            "error_feedback": "GPTQ-style full-Hessian between native trellis groups",
            "activation_order": "static within native 16-column group",
            "scale_refit_iterations": 2,
            "physical_bpw": 4.25,
            "table_bytes": 0,
            "ldlq": False,
        },
        "codebook_sha256": hashlib.sha256(codebook.numpy().tobytes()).hexdigest(),
        "source_index_sha256": sha256_file(args.source_index),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "logical_elements": logical_elements,
        "physical_payload_bytes": payload_bytes,
        "physical_payload_bpw": payload_bpw,
        "projection_metrics": metrics,
        "outputs": output_receipts,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
        "runtime_status": "physical rotated P8 payload; pseudoquant and device closure remain separate gates",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
