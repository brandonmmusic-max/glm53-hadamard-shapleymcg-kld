"""Fit one shared block-16 orthogonal routed-input rotation for a GLM MoE layer.

Only sealed fit-role BF16 captures and BF16 expert weights enter the optimizer.
The surrogate is Hessian-weighted gate/up reconstruction on the exact E2M1 and
E4M3 grid. The selected quantized point is treated as piecewise constant so
the rotation receives a useful gradient toward the current grid point.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .block_gptq import _best_scales, block_hessian
from .block_rotation import (
    apply_weight_rotation,
    cayley_rotation,
    hadamard16,
    orthogonality_error,
    rotate_block_hessian,
)
from .capture import LayerCapture
from .modelopt import E2M1_LEVELS, choose_global_scale
from .shard_index import IndexedCheckpoint, sha256_file


@torch.no_grad()
def _dequantize_on_device(weight: torch.Tensor, global_scale: torch.Tensor, search_grid: int) -> torch.Tensor:
    blocks = weight.reshape(weight.shape[0], weight.shape[1] // 16, 16)
    scale = _best_scales(blocks, global_scale, search_grid).float() * global_scale
    levels = E2M1_LEVELS.to(weight.device)
    mids = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize((blocks / scale[..., None]).abs(), mids)
    return (levels[codes] * torch.sign(blocks) * scale[..., None]).reshape_as(weight)


def _expert_loss(
    gate: torch.Tensor,
    up: torch.Tensor,
    hessian: torch.Tensor,
    rotation: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    gate_r = apply_weight_rotation(gate, rotation)
    up_r = apply_weight_rotation(up, rotation)
    h_r = rotate_block_hessian(hessian, rotation)
    global_scale = choose_global_scale(gate_r.detach(), up_r.detach()).to(gate_r.device)
    gate_q = _dequantize_on_device(gate_r.detach(), global_scale, search_grid)
    up_q = _dequantize_on_device(up_r.detach(), global_scale, search_grid)
    gate_delta = gate_r - gate_q
    up_delta = up_r - up_q
    gate_blocks = gate_delta.reshape(gate.shape[0], -1, 16)
    up_blocks = up_delta.reshape(up.shape[0], -1, 16)
    loss = torch.einsum("obg,bgh,obh->", gate_blocks, h_r, gate_blocks)
    loss = loss + torch.einsum("obg,bgh,obh->", up_blocks, h_r, up_blocks)
    return loss / (gate.numel() + up.numel())


def _selected_experts(count: int) -> list[int]:
    if not 1 <= count <= 288:
        raise ValueError("expert count must be 1..288")
    return torch.linspace(0, 287, count).round().to(torch.int64).unique().tolist()


@torch.no_grad()
def _polar(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix)
    return left @ right


@torch.no_grad()
def _procrustes_target(
    weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
    experts: list[int],
    rotation: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    if rotation.ndim != 3:
        raise ValueError("alternating Procrustes requires per-block rotations")
    cross = torch.zeros_like(rotation)
    for expert in experts:
        gate, up = weights[expert]
        gate_r = apply_weight_rotation(gate, rotation)
        up_r = apply_weight_rotation(up, rotation)
        global_scale = choose_global_scale(gate_r, up_r).to(gate_r.device)
        gate_q = _dequantize_on_device(gate_r, global_scale, search_grid)
        up_q = _dequantize_on_device(up_r, global_scale, search_grid)
        gate_b = gate.float().reshape(gate.shape[0], -1, 16)
        up_b = up.float().reshape(up.shape[0], -1, 16)
        cross += torch.einsum("obg,obh->bgh", gate_b, gate_q.reshape(gate.shape[0], -1, 16))
        cross += torch.einsum("obg,obh->bgh", up_b, up_q.reshape(up.shape[0], -1, 16))
    return _polar(cross)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--hessian-file", type=Path, action="append")
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--init", choices=("identity", "had16"), required=True)
    parser.add_argument("--sharing", choices=("shared", "per-block"), default="shared")
    parser.add_argument("--optimizer", choices=("cayley", "procrustes"), default="cayley")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if not 3 <= args.layer <= 44:
        raise ValueError("layer must be 3..44")
    if args.batch < 1 or args.batch > args.experts:
        raise ValueError("invalid batch")
    if args.optimizer == "procrustes" and args.sharing != "per-block":
        raise ValueError("Procrustes optimizer requires per-block sharing")
    if (args.capture_root is None) == (not args.hessian_file):
        raise ValueError("provide exactly one of --capture-root or --hessian-file")

    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(args.capture_root, args.layer, args.roles, args.max_samples) if args.capture_root else None
    saved_hessians = {}
    if args.hessian_file:
        for path in args.hessian_file:
            for key, tensor in load_file(str(path), device="cpu").items():
                if key.startswith("hessian_"):
                    if key in saved_hessians:
                        raise ValueError(f"duplicate {key}")
                    saved_hessians[key] = tensor
    experts = _selected_experts(args.experts)
    prefix = checkpoint.expert_prefix(args.layer, experts[0]).split(f"layers.{args.layer}.")[0]
    weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    hessians: dict[int, torch.Tensor] = {}
    source_files: set[str] = set()
    for expert in experts:
        stem = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        gate_name, up_name = f"{stem}.gate_proj.weight", f"{stem}.up_proj.weight"
        source_files.update((checkpoint.weight_map[gate_name], checkpoint.weight_map[up_name]))
        weights[expert] = (checkpoint.get(gate_name).to(device), checkpoint.get(up_name).to(device))
        if capture is not None:
            hidden, route = capture.samples(expert)
            hessians[expert] = block_hessian(hidden.to(device), route.to(device))
        else:
            hessians[expert] = saved_hessians[f"hessian_{expert:03d}"].to(device)

    base = torch.eye(16, device=device) if args.init == "identity" else hadamard16(device=device)
    @torch.no_grad()
    def score(rotation: torch.Tensor) -> float:
        return float(sum(_expert_loss(*weights[e], hessians[e], rotation, args.search_grid) for e in experts))

    identity_score = score(torch.eye(16, device=device))
    had16_score = score(hadamard16(device=device))
    history = []
    if args.optimizer == "cayley":
        parameter_shape = (16, 16) if args.sharing == "shared" else (4096 // 16, 16, 16)
        parameter = torch.zeros(parameter_shape, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([parameter], lr=args.lr)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + args.layer + (1000 if args.init == "had16" else 0))
        for _ in range(args.steps):
            picked = torch.randperm(len(experts), generator=generator)[: args.batch].tolist()
            rotation = cayley_rotation(parameter, base)
            loss = sum(_expert_loss(*weights[experts[i]], hessians[experts[i]], rotation, args.search_grid) for i in picked) / len(picked)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter], 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        learned = cayley_rotation(parameter, base).detach()
        learned_score = score(learned)
    else:
        learned = base.expand(4096 // 16, -1, -1).clone()
        learned_score = score(learned)
        history.append(learned_score)
        for _ in range(args.steps):
            target = _procrustes_target(weights, experts, learned, args.search_grid)
            candidates = [learned]
            for alpha in (1.0, 0.5, 0.25, 0.1):
                candidates.append(_polar((1.0 - alpha) * learned + alpha * target))
            scores = [score(candidate) for candidate in candidates]
            best = min(range(len(scores)), key=scores.__getitem__)
            learned, new_score = candidates[best], scores[best]
            history.append(new_score)
            if new_score >= learned_score * (1.0 - 1e-7):
                learned_score = new_score
                break
            learned_score = new_score

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"layer_{args.layer:03d}": learned.cpu().contiguous()}, str(args.output), metadata={
        "schema": "glm53-nvfp4-v3.learned-block16-layer.v1",
        "layer": str(args.layer),
        "init": args.init,
        "sharing": args.sharing,
    })
    receipt = {
        "schema": "glm53-nvfp4-v3.learned-block16-layer-receipt.v1",
        "layer": args.layer,
        "init": args.init,
        "experts": experts,
        "algorithm": {"parameterization": "Cayley SO16" if args.optimizer == "cayley" else "alternating orthogonal Procrustes O16", "optimizer": args.optimizer, "sharing": args.sharing, "steps": args.steps, "batch": args.batch, "lr": args.lr, "search_grid": args.search_grid, "seed": args.seed, "surrogate": "fit Hessian weighted exact-grid RTN reconstruction"},
        "scores": {"identity": identity_score, "had16": had16_score, "learned": learned_score, "gain_vs_identity": 1.0 - learned_score / identity_score, "gain_vs_init": 1.0 - learned_score / (identity_score if args.init == "identity" else had16_score)},
        "history": history,
        "orthogonality_max_abs": orthogonality_error(learned),
        "source_files": [{"path": name, "sha256": sha256_file(args.source / name)} for name in sorted(source_files)],
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json") if args.capture_root else None,
        "hessian_inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in (args.hessian_file or [])],
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)},
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["scores"], sort_keys=True))


if __name__ == "__main__":
    main()
