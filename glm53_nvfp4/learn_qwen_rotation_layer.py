"""Fit independent Qwen-style R_in and R_mid rotations for one GLM MoE layer.

Gate/up share R_in; down uses R_mid.  ``per-projection`` reproduces the
Qwen pilot's file-of-record Hessian-weighted projection objective (evaluated
directly on the routed samples); ``full-expert`` preserves its later nonlinear
diagnostic.  All fake quantization points obey the packed ModelOpt NVFP4 scale
ABI.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .block_rotation import (
    apply_activation_rotation,
    cayley_rotation,
    hadamard16,
    orthogonality_error,
)
from .capture import LayerCapture
from .modelopt import E2M1_LEVELS, E4M3_MAX
from .shard_index import IndexedCheckpoint, sha256_file


def _selected_experts(count: int) -> list[int]:
    if not 1 <= count <= 288:
        raise ValueError("expert count must be 1..288")
    return torch.linspace(0, 287, count).round().to(torch.int64).unique().tolist()


def _fake_nvfp4(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    search_grid: int,
    *,
    refit_iterations: int = 2,
) -> torch.Tensor:
    blocks = weight.reshape(weight.shape[0], weight.shape[1] // 16, 16)
    levels = E2M1_LEVELS.to(weight.device)
    mids = (levels[1:] + levels[:-1]) / 2

    def round_levels(value: torch.Tensor) -> torch.Tensor:
        codes = torch.bucketize(value.detach().abs(), mids)
        rounded = levels[codes] * torch.sign(value.detach())
        return value + (rounded - value).detach()

    def round_e4m3(value: torch.Tensor) -> torch.Tensor:
        rounded = value.clamp(0, E4M3_MAX).to(torch.float8_e4m3fn).float()
        return value + (rounded - value).detach()

    maximum = blocks.abs().amax(-1).clamp_min(1e-12)
    best_error = None
    best_real_scale = None
    for range_max in (6.0, 4.0):
        for shrink in torch.linspace(0.65, 1.0, search_grid, device=weight.device):
            raw = maximum / range_max * shrink / global_scale
            real_scale = round_e4m3(raw) * global_scale
            quantized = round_levels(blocks / real_scale[..., None]) * real_scale[..., None]
            error = (quantized.detach() - blocks.detach()).square().sum(-1)
            if best_error is None:
                best_error, best_real_scale = error, real_scale
            else:
                better = error < best_error
                best_error = torch.where(better, error, best_error)
                best_real_scale = torch.where(better, real_scale, best_real_scale)
    # Qwen's two whole-tensor least-squares refinements.  A common gamma is
    # representable by ModelOpt's FP32 weight_scale_2 while the E4M3 codes stay
    # fixed; the packed quantizer performs its corresponding frozen-scale pass.
    for _ in range(refit_iterations):
        codes = round_levels(blocks / best_real_scale[..., None])
        basis = codes * best_real_scale[..., None]
        gamma = (
            (blocks * basis).sum() / basis.square().sum().clamp_min(1e-20)
        ).clamp(0.8, 1.25)
        best_real_scale = best_real_scale * gamma
    return (
        round_levels(blocks / best_real_scale[..., None])
        * best_real_scale[..., None]
    ).reshape_as(weight)


def _fake_gate_up(
    gate: torch.Tensor, up: torch.Tensor, search_grid: int
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized_gate = []
    quantized_up = []
    for expert in range(gate.shape[0]):
        maximum = torch.maximum(
            gate[expert].abs().amax(), up[expert].abs().amax()
        )
        global_scale = (maximum / (6.0 * E4M3_MAX)).clamp_min(1e-12)
        # ModelOpt stores one weight_scale_2 for the fused gate/up family.
        # Refit that scalar jointly across both row sets, as Qwen does after
        # concatenating gate/up, rather than silently fitting one per tensor.
        for _ in range(2):
            q_gate = _fake_nvfp4(
                gate[expert], global_scale, search_grid, refit_iterations=0
            )
            q_up = _fake_nvfp4(
                up[expert], global_scale, search_grid, refit_iterations=0
            )
            numerator = (gate[expert] * q_gate).sum() + (up[expert] * q_up).sum()
            denominator = q_gate.square().sum() + q_up.square().sum()
            gamma = (numerator / denominator.clamp_min(1e-20)).clamp(0.8, 1.25)
            global_scale = global_scale * gamma
        quantized_gate.append(
            _fake_nvfp4(
                gate[expert], global_scale, search_grid, refit_iterations=0
            )
        )
        quantized_up.append(
            _fake_nvfp4(
                up[expert], global_scale, search_grid, refit_iterations=0
            )
        )
    return torch.stack(quantized_gate), torch.stack(quantized_up)


def _fake_down(down: torch.Tensor, search_grid: int) -> torch.Tensor:
    result = []
    for expert in range(down.shape[0]):
        maximum = down[expert].abs().amax()
        global_scale = (maximum / (6.0 * E4M3_MAX)).clamp_min(1e-12)
        result.append(_fake_nvfp4(down[expert], global_scale, search_grid))
    return torch.stack(result)


def _expert_output(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    gate_value = torch.einsum("esk,eik->esi", hidden, gate).clamp(max=10.0)
    up_value = torch.einsum("esk,eik->esi", hidden, up).clamp(-10.0, 10.0)
    middle = F.silu(gate_value) * up_value
    return torch.einsum("esi,eki->esk", middle, down)


def objective(
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    reference: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    rotation_in: torch.Tensor,
    rotation_mid: torch.Tensor,
    search_grid: int,
) -> torch.Tensor:
    gate_rotated = apply_activation_rotation(gate, rotation_in)
    up_rotated = apply_activation_rotation(up, rotation_in)
    down_rotated = apply_activation_rotation(down, rotation_mid)
    gate_quantized, up_quantized = _fake_gate_up(
        gate_rotated, up_rotated, search_grid
    )
    down_quantized = _fake_down(down_rotated, search_grid)
    hidden_rotated = apply_activation_rotation(hidden, rotation_in)
    gate_value = torch.einsum(
        "esk,eik->esi", hidden_rotated, gate_quantized
    ).clamp(max=10.0)
    up_value = torch.einsum(
        "esk,eik->esi", hidden_rotated, up_quantized
    ).clamp(-10.0, 10.0)
    middle = F.silu(gate_value) * up_value
    middle_rotated = apply_activation_rotation(middle, rotation_mid)
    output = torch.einsum("esi,eki->esk", middle_rotated, down_quantized)
    per_sample = (output - reference).square().mean(dim=-1)
    normalized = route_weights.square()
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (per_sample * normalized).sum(dim=-1).mean()


def projection_objective(
    hidden: torch.Tensor,
    middle: torch.Tensor,
    route_weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    rotation_in: torch.Tensor,
    rotation_mid: torch.Tensor,
    search_grid: int,
    family: str = "both",
) -> torch.Tensor:
    """Qwen file-of-record per-projection reconstruction objective.

    Computing the residual linear outputs on routed samples is algebraically
    equal to the Hessian trace objective while avoiding materialized rotated
    4096x4096 Hessians during every optimizer step.
    """
    if family not in {"both", "in", "mid"}:
        raise ValueError(f"invalid projection family {family!r}")
    per_sample = torch.zeros_like(route_weights)
    if family in {"both", "in"}:
        gate_rotated = apply_activation_rotation(gate, rotation_in)
        up_rotated = apply_activation_rotation(up, rotation_in)
        gate_quantized, up_quantized = _fake_gate_up(
            gate_rotated, up_rotated, search_grid
        )
        hidden_rotated = apply_activation_rotation(hidden, rotation_in)
        gate_error = torch.einsum(
            "esk,eik->esi", hidden_rotated, gate_quantized - gate_rotated
        ).square().mean(dim=-1)
        up_error = torch.einsum(
            "esk,eik->esi", hidden_rotated, up_quantized - up_rotated
        ).square().mean(dim=-1)
        per_sample = per_sample + gate_error + up_error
    if family in {"both", "mid"}:
        down_rotated = apply_activation_rotation(down, rotation_mid)
        down_quantized = _fake_down(down_rotated, search_grid)
        middle_rotated = apply_activation_rotation(middle, rotation_mid)
        down_error = torch.einsum(
            "esi,eki->esk", middle_rotated, down_quantized - down_rotated
        ).square().mean(dim=-1)
        per_sample = per_sample + down_error
    normalized = route_weights.square()
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (per_sample * normalized).sum(dim=-1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--init", choices=("identity", "had16"), required=True)
    parser.add_argument(
        "--objective",
        choices=("per-projection", "full-expert"),
        default="full-expert",
    )
    parser.add_argument(
        "--family", choices=("both", "in", "mid"), default="both"
    )
    parser.add_argument(
        "--sharing",
        choices=("shared", "per-block"),
        default="shared",
        help="share one SO16 transform across the family or learn one per 16-wide block",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="expert microbatches per optimizer step; preserves a larger effective batch",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.objective == "full-expert" and args.family != "both":
        raise ValueError("full-expert objective requires --family both")
    if args.sharing == "per-block" and args.family == "both":
        raise ValueError(
            "per-block fitting requires one family at a time because input and middle widths differ"
        )
    if not 3 <= args.layer <= 44:
        raise ValueError("layer must be 3..44")
    if not 1 <= args.batch <= args.experts or args.accumulation < 1:
        raise ValueError("invalid expert batch")
    if args.batch * args.accumulation > args.experts:
        raise ValueError("effective expert batch exceeds selected experts")

    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root, args.layer, args.roles, args.max_samples
    )
    experts = _selected_experts(args.experts)
    prefix = checkpoint.expert_prefix(args.layer, experts[0]).split(
        f"layers.{args.layer}."
    )[0]
    hidden_items = []
    route_items = []
    gate_items = []
    up_items = []
    down_items = []
    source_files: set[str] = set()
    for expert in experts:
        hidden, route = capture.samples(expert)
        stem = f"{prefix}layers.{args.layer}.mlp.experts.{expert}"
        names = [
            f"{stem}.gate_proj.weight",
            f"{stem}.up_proj.weight",
            f"{stem}.down_proj.weight",
        ]
        source_files.update(checkpoint.weight_map[name] for name in names)
        # Keep the sealed BF16 capture compact.  Selected expert batches are
        # promoted to FP32 inside the objective, matching X^T X accumulation
        # without materializing the entire padded calibration panel in FP32.
        hidden_items.append(hidden.to(device=device, dtype=torch.bfloat16))
        route_items.append(route.to(device=device, dtype=torch.float32))
        gate_items.append(checkpoint.get(names[0]).to(device=device).float())
        up_items.append(checkpoint.get(names[1]).to(device=device).float())
        down_items.append(checkpoint.get(names[2]).to(device=device).float())
    hidden = torch.nn.utils.rnn.pad_sequence(hidden_items, batch_first=True)
    route_weights = torch.nn.utils.rnn.pad_sequence(route_items, batch_first=True)
    gate = torch.stack(gate_items)
    up = torch.stack(up_items)
    down = torch.stack(down_items)
    del hidden_items, route_items, gate_items, up_items, down_items
    reference = None
    middle = None
    with torch.no_grad():
        if args.objective == "full-expert":
            reference = _expert_output(hidden.float(), gate, up, down)
        elif args.family in {"both", "mid"}:
            # Calculate the post-SwiGLU calibration panel in small batches and
            # retain it as BF16, just as the captured routed input is retained.
            # This makes all-route fitting possible on GLM's larger experts.
            middle_items = []
            for start in range(0, len(experts), args.batch):
                stop = min(start + args.batch, len(experts))
                hidden_batch = hidden[start:stop].float()
                gate_value = torch.einsum(
                    "esk,eik->esi", hidden_batch, gate[start:stop]
                ).clamp(max=10.0)
                up_value = torch.einsum(
                    "esk,eik->esi", hidden_batch, up[start:stop]
                ).clamp(-10.0, 10.0)
                middle_items.append((F.silu(gate_value) * up_value).to(torch.bfloat16))
            middle = torch.cat(middle_items, dim=0)
            del middle_items
        if args.objective == "per-projection" and args.family == "in":
            down = down[:1].clone()
        elif args.objective == "per-projection" and args.family == "mid":
            hidden = hidden[:1].clone()
            gate = gate[:1].clone()
            up = up[:1].clone()

    def fit_objective(
        selected: torch.Tensor,
        rotation_in: torch.Tensor,
        rotation_mid: torch.Tensor,
    ) -> torch.Tensor:
        if args.objective == "per-projection":
            hidden_batch = (
                hidden[selected].float()
                if args.family in {"both", "in"}
                else middle.new_empty(0)
            )
            middle_batch = (
                middle[selected].float()
                if middle is not None
                else hidden_batch.new_empty(0)
            )
            return projection_objective(
                hidden_batch,
                middle_batch,
                route_weights[selected],
                gate[selected] if args.family in {"both", "in"} else gate,
                up[selected] if args.family in {"both", "in"} else up,
                down[selected] if args.family in {"both", "mid"} else down,
                rotation_in,
                rotation_mid,
                args.search_grid,
                args.family,
            )
        assert reference is not None
        return objective(
            hidden[selected].float(),
            route_weights[selected],
            reference[selected],
            gate[selected],
            up[selected],
            down[selected],
            rotation_in,
            rotation_mid,
            args.search_grid,
        )

    def score_all(
        rotation_in: torch.Tensor, rotation_mid: torch.Tensor
    ) -> float:
        """Evaluate every expert without materializing all rotated weights twice."""
        total = torch.zeros((), device=device)
        count = 0
        for start in range(0, len(experts), args.batch):
            selected = torch.arange(
                start, min(start + args.batch, len(experts)), device=device
            )
            value = fit_objective(selected, rotation_in, rotation_mid)
            total += value * selected.numel()
            count += selected.numel()
        return float(total / count)

    eye = torch.eye(16, device=device)
    hadamard = hadamard16(device=device)
    base = eye if args.init == "identity" else hadamard
    in_shape = (4096 // 16, 16, 16) if args.sharing == "per-block" and args.family == "in" else (16, 16)
    mid_shape = (2048 // 16, 16, 16) if args.sharing == "per-block" and args.family == "mid" else (16, 16)
    parameter_in = torch.zeros(in_shape, device=device, requires_grad=True)
    parameter_mid = torch.zeros(mid_shape, device=device, requires_grad=True)
    optimized_parameters = (
        [parameter_in, parameter_mid]
        if args.family == "both"
        else [parameter_in if args.family == "in" else parameter_mid]
    )
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(
        args.seed + args.layer + (1000 if args.init == "had16" else 0)
    )
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        selected_all = torch.randperm(len(experts), generator=generator)[
            : args.batch * args.accumulation
        ].to(device)
        loss_value = 0.0
        for microbatch in selected_all.split(args.batch):
            rotation_in = cayley_rotation(parameter_in, base)
            rotation_mid = cayley_rotation(parameter_mid, base)
            micro_loss = fit_objective(microbatch, rotation_in, rotation_mid)
            (micro_loss / args.accumulation).backward()
            loss_value += float(micro_loss.detach()) / args.accumulation
        torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
        optimizer.step()
        history.append(loss_value)
        print(json.dumps({"step": step + 1, "loss": history[-1]}), flush=True)

    with torch.no_grad():
        learned_in = cayley_rotation(parameter_in, base).detach()
        learned_mid = cayley_rotation(parameter_mid, base).detach()
        scores = {}
        for name, rotation_in, rotation_mid in (
            ("identity", eye, eye),
            ("had16", hadamard, hadamard),
            (f"learned-{args.init}", learned_in, learned_mid),
        ):
            scores[name] = score_all(rotation_in, rotation_mid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"layer_{args.layer:03d}_in": learned_in.cpu().contiguous(),
            f"layer_{args.layer:03d}_mid": learned_mid.cpu().contiguous(),
        },
        str(args.output),
        metadata={
            "schema": "glm53-nvfp4-v5.qwen-full-expert-learned-rotation.v1",
            "layer": str(args.layer),
            "init": args.init,
        },
    )
    learned_key = f"learned-{args.init}"
    receipt = {
        "schema": "glm53-nvfp4-v5.qwen-full-expert-learned-rotation-receipt.v1",
        "layer": args.layer,
        "init": args.init,
        "experts": experts,
        "algorithm": {
            "parameterization": "independent Cayley SO16 R_in and R_mid",
            "sharing": args.sharing,
            "rotation_count": {
                "in": int(learned_in.shape[0]) if learned_in.ndim == 3 else 1,
                "mid": int(learned_mid.shape[0]) if learned_mid.ndim == 3 else 1,
            },
            "objective": args.objective,
            "family": args.family,
            "format": "packed-representable ModelOpt NVFP4",
            "steps": args.steps,
            "batch": args.batch,
            "accumulation": args.accumulation,
            "effective_batch": args.batch * args.accumulation,
            "lr": args.lr,
            "search_grid": args.search_grid,
            "seed": args.seed,
        },
        "scores": {
            **scores,
            "gain_vs_identity": 1.0 - scores[learned_key] / scores["identity"],
            "gain_vs_init": 1.0
            - scores[learned_key] / scores[args.init],
        },
        "history": history,
        "orthogonality_max_abs": {
            "in": orthogonality_error(learned_in),
            "mid": orthogonality_error(learned_mid),
        },
        "source_files": [
            {"path": name, "sha256": sha256_file(args.source / name)}
            for name in sorted(source_files)
        ],
        "roles_sha256": sha256_file(args.roles),
        "capture_manifest_sha256": sha256_file(
            args.capture_root / "capture-manifest.json"
        ),
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["scores"], sort_keys=True))


if __name__ == "__main__":
    main()
