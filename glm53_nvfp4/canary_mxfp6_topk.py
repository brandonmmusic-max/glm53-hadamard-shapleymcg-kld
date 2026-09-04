"""Compare packed W6A8 and exact decoded BF16 over multi-expert top-k routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().flatten()
    reference_f = reference.float().flatten()
    error = actual_f - reference_f
    return {
        "cosine": float(F.cosine_similarity(actual_f, reference_f, dim=0).item()),
        "nmse": float(error.double().square().sum().item() / reference_f.double().square().sum().item()),
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--a1-gscale", type=float, default=1.0)
    parser.add_argument("--a2-gscale", type=float, default=1.0)
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.topk <= args.experts:
        raise ValueError("topk must be between one and the number of experts")

    from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod, get_fp6_moe_weight_plan
    from b12x.moe import fused_moe

    device = torch.device("cuda")
    projections = ("gate_proj", "up_proj", "down_proj")
    expert_ids = list(range(args.expert_start, args.expert_start + args.experts))
    native: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    dense: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(str(args.native), framework="pt", device="cpu") as handle:
        for expert in expert_ids:
            native[expert] = {}
            for projection in projections:
                stem = f"model.language_model.layers.{args.layer}.mlp.experts.{expert}.{projection}"
                native[expert][projection] = {
                    suffix: handle.get_tensor(f"{stem}.{suffix}").to(device)
                    for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
                }
    with safe_open(str(args.dense), framework="pt", device="cpu") as handle:
        for expert in expert_ids:
            dense[expert] = {
                projection: handle.get_tensor(
                    f"model.language_model.layers.{args.layer}.mlp.experts.{expert}.{projection}.weight"
                ).to(device)
                for projection in projections
            }

    w1 = torch.stack(
        [
            torch.cat(
                (native[expert]["up_proj"]["weight"], native[expert]["gate_proj"]["weight"]),
                dim=0,
            )
            for expert in expert_ids
        ]
    ).contiguous()
    w1_scale = torch.stack(
        [
            torch.cat(
                (
                    native[expert]["up_proj"]["weight_scale"],
                    native[expert]["gate_proj"]["weight_scale"],
                ),
                dim=0,
            )
            for expert in expert_ids
        ]
    ).contiguous()
    w2 = torch.stack([native[expert]["down_proj"]["weight"] for expert in expert_ids]).contiguous()
    w2_scale = torch.stack(
        [native[expert]["down_proj"]["weight_scale"] for expert in expert_ids]
    ).contiguous()
    plan = get_fp6_moe_weight_plan(
        source_format="mxfp6_e2m3",
        activation="silu",
        num_experts=args.experts,
        hidden_size=4096,
        intermediate_size=2048,
    )
    ones = torch.ones(args.experts, dtype=torch.float32, device=device)
    a1_gscale = torch.full((args.experts,), args.a1_gscale, dtype=torch.float32, device=device)
    a2_gscale = torch.full((args.experts,), args.a2_gscale, dtype=torch.float32, device=device)
    prepared = fused_moe.prepare_weights(
        plan=plan,
        w1_fp4=w1,
        w1_blockscale=w1_scale,
        w1_global_scale=ones,
        a1_gscale=a1_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=ones.clone(),
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )
    method = B12XFP6MoEMethod(prepared, plan)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(args.tokens, 4096, generator=generator, device=device, dtype=torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(args.experts, generator=generator, device=device)[: args.topk] for _ in range(args.tokens)]
    ).to(torch.int32)
    raw_weights = torch.rand(args.tokens, args.topk, generator=generator, device=device)
    topk_weights = 2.5 * raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    fused = method.apply(x, topk_weights, topk_ids)

    reference = torch.zeros_like(fused)
    per_expert_outputs: dict[int, torch.Tensor] = {}
    for local_expert, expert in enumerate(expert_ids):
        gate = F.linear(x, dense[expert]["gate_proj"])
        up = F.linear(x, dense[expert]["up_proj"])
        if args.swiglu_limit is not None:
            gate = gate.clamp(max=args.swiglu_limit)
            up = up.clamp(min=-args.swiglu_limit, max=args.swiglu_limit)
        middle = F.silu(gate.float()).to(torch.bfloat16) * up
        per_expert_outputs[local_expert] = F.linear(middle, dense[expert]["down_proj"])
    for token in range(args.tokens):
        for slot in range(args.topk):
            local_expert = int(topk_ids[token, slot].item())
            reference[token] += (
                topk_weights[token, slot] * per_expert_outputs[local_expert][token].float()
            ).to(torch.bfloat16)
    torch.cuda.synchronize()

    payload = {
        "schema": "glm53-trellis-mxf.mxfp6-topk-canary.v1",
        "native": str(args.native),
        "dense": str(args.dense),
        "layer": args.layer,
        "expert_ids": expert_ids,
        "tokens": args.tokens,
        "topk": args.topk,
        "seed": args.seed,
        "routing_scale": 2.5,
        "a1_gscale": args.a1_gscale,
        "a2_gscale": args.a2_gscale,
        "swiglu_limit": args.swiglu_limit,
        "fused_w6a8_vs_exact_decoded_bf16": _metrics(fused, reference),
        "w13_runtime_order": "up-then-gate (W31), matching the pinned v79 plugin half-swap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
