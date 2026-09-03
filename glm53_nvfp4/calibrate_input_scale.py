"""Calibrate static ModelOpt NVFP4 routed-input scales after a block rotation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .block_rotation import apply_activation_rotation, hadamard16, load_layer_rotation
from .capture import LayerCapture
from .shard_index import IndexedCheckpoint, sha256_file


NVFP4_GLOBAL_DENOMINATOR = 6.0 * 448.0


def routed_amax(
    capture: LayerCapture,
    rotation: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one fit-only activation amax per routed expert.

    The fused MoE sees the same input row for every selected expert.  We apply
    the exact runtime basis change once per row, then scatter-reduce its scalar
    amax to each routed expert across every preregistered fit window.
    """
    maxima = torch.zeros(288, dtype=torch.float32, device=device)
    for window in capture.window_indices:
        start = window * 2048
        words = np.array(capture.hidden_words[start : start + 2048], copy=True)
        hidden = torch.from_numpy(words).view(torch.bfloat16).to(device=device).float()
        if rotation is not None:
            hidden = apply_activation_rotation(hidden, rotation)
        row_amax = hidden.abs().amax(dim=-1)
        ids = torch.from_numpy(
            np.array(capture.topk_ids[start : start + 2048], copy=True)
        ).to(device=device, dtype=torch.long)
        maxima.scatter_reduce_(
            0,
            ids.reshape(-1),
            row_amax[:, None].expand_as(ids).reshape(-1),
            reduce="amax",
            include_self=True,
        )
    if bool((maxima <= 0).any()):
        missing = torch.nonzero(maxima <= 0).flatten().tolist()
        raise RuntimeError(f"no routed fit activation observed for experts {missing}")
    return maxima.cpu()


@torch.no_grad()
def down_amax(
    capture: LayerCapture,
    checkpoint: IndexedCheckpoint,
    layer: int,
    rotation: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return per-expert post-SwiGLU amax in the down-projection basis."""
    maxima = torch.zeros(288, dtype=torch.float32)
    prefix = checkpoint.expert_prefix(layer, 0).split(f"layers.{layer}.")[0]
    for expert in range(288):
        base = f"{prefix}layers.{layer}.mlp.experts.{expert}"
        hidden, _ = capture.samples(expert)
        hidden = hidden.to(device=device)
        gate = checkpoint.get(f"{base}.gate_proj.weight").to(device=device)
        up = checkpoint.get(f"{base}.up_proj.weight").to(device=device)
        middle = F.silu(F.linear(hidden, gate).clamp(max=10.0)) * F.linear(
            hidden, up
        ).clamp(-10.0, 10.0)
        if rotation is not None:
            middle = apply_activation_rotation(middle, rotation)
        maxima[expert] = middle.abs().amax().cpu()
        del hidden, gate, up, middle
    if bool((maxima <= 0).any()):
        missing = torch.nonzero(maxima <= 0).flatten().tolist()
        raise RuntimeError(f"no down-input fit activation observed for experts {missing}")
    return maxima


def scale_tensors(
    layer: int,
    maxima: torch.Tensor,
    down_maxima: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if tuple(maxima.shape) != (288,):
        raise ValueError(f"expected 288 expert maxima, got {tuple(maxima.shape)}")
    result: dict[str, torch.Tensor] = {}
    # The carrier ABI stores identical static scales across experts for each
    # layer/family.  Humming and FlashInfer therefore see the same canonical
    # value rather than relying on a backend-specific max collapse.
    input_scale = (maxima.max().float() / NVFP4_GLOBAL_DENOMINATOR).reshape(())
    down_scale = (
        (down_maxima.max().float() / NVFP4_GLOBAL_DENOMINATOR).reshape(())
        if down_maxima is not None
        else None
    )
    for expert in range(288):
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
        result[f"{base}.gate_proj.input_scale"] = input_scale.clone()
        result[f"{base}.up_proj.input_scale"] = input_scale.clone()
        if down_maxima is not None:
            result[f"{base}.down_proj.input_scale"] = down_scale.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rotation", choices=("identity", "had16", "learned"), required=True)
    parser.add_argument("--rotation-file", type=Path)
    parser.add_argument(
        "--rotation-scope", choices=("gate-up", "mid-only", "all"), default="gate-up"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.layer <= 44:
        raise ValueError("layer must be 3..44")
    if args.rotation == "learned" and args.rotation_file is None:
        raise ValueError("learned rotation requires --rotation-file")
    if args.rotation_scope in {"mid-only", "all"} and args.source is None:
        raise ValueError("all-projection calibration requires --source")

    started = time.time()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("calibration requires a CUDA device")
    torch.cuda.set_device(device)
    capture = LayerCapture(
        args.capture_root, args.layer, args.roles, max_samples=args.max_samples
    )
    if args.rotation == "identity":
        rotation_in = rotation_mid = None
    elif args.rotation == "had16":
        fixed = hadamard16(device=device)
        rotation_in = fixed if args.rotation_scope in {"gate-up", "all"} else None
        rotation_mid = fixed if args.rotation_scope in {"mid-only", "all"} else None
    else:
        rotation_in = (
            load_layer_rotation(
                args.rotation_file, args.layer, kind="in", width=4096, device=device
            )
            if args.rotation_scope in {"gate-up", "all"}
            else None
        )
        rotation_mid = (
            load_layer_rotation(
                args.rotation_file,
                args.layer,
                kind="mid",
                width=2048,
                device=device,
            )
            if args.rotation_scope in {"mid-only", "all"}
            else None
        )
    maxima = routed_amax(capture, rotation_in, device=device)
    mid_maxima = None
    if args.rotation_scope in {"mid-only", "all"}:
        mid_maxima = down_amax(
            capture,
            IndexedCheckpoint(args.source, args.source_index),
            args.layer,
            rotation_mid,
            device=device,
        )
    tensors = scale_tensors(args.layer, maxima, mid_maxima)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v5.qwen-exact-rotated-input-scales.v1" if mid_maxima is not None else "glm53-nvfp4-v4.rotated-input-scales.v1",
            "layer": str(args.layer),
            "rotation": args.rotation,
            "rotation_scope": args.rotation_scope,
            "calibration": "fit-only routed amax / (6*448)",
        },
    )
    receipt = {
        "schema": "glm53-nvfp4-v5.qwen-exact-rotated-input-scale-receipt.v1" if mid_maxima is not None else "glm53-nvfp4-v4.rotated-input-scale-receipt.v1",
        "layer": args.layer,
        "rotation": args.rotation,
        "rotation_scope": args.rotation_scope,
        "rotation_file": str(args.rotation_file.resolve()) if args.rotation_file else None,
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "fit_windows": len(capture.window_indices),
        "formula": "amax / (6 * 448)",
        "amax": {
            "minimum": float(maxima.min()),
            "median": float(maxima.median()),
            "mean": float(maxima.mean()),
            "maximum": float(maxima.max()),
        },
        "down_amax": (
            {
                "minimum": float(mid_maxima.min()),
                "median": float(mid_maxima.median()),
                "mean": float(mid_maxima.mean()),
                "maximum": float(mid_maxima.max()),
            }
            if mid_maxima is not None
            else None
        ),
        "scale": {
            "minimum": float((maxima / NVFP4_GLOBAL_DENOMINATOR).min()),
            "median": float((maxima / NVFP4_GLOBAL_DENOMINATOR).median()),
            "mean": float((maxima / NVFP4_GLOBAL_DENOMINATOR).mean()),
            "maximum": float((maxima / NVFP4_GLOBAL_DENOMINATOR).max()),
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensors": len(tensors),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
