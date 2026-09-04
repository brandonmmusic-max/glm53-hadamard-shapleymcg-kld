"""Compare one bridged MXFP6 expert through dense and fused native kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    a = actual.float().flatten()
    r = reference.float().flatten()
    delta = a - r
    return {
        "cosine": float(F.cosine_similarity(a, r, dim=0).item()),
        "nmse": float(delta.double().square().sum().item() / r.double().square().sum().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from b12x.integration.vllm.fp6_serving import (
        B12XFP6MoEMethod,
        get_fp6_moe_weight_plan,
    )
    from b12x.moe import fused_moe
    from b12x.quantization.mxfp6 import (
        dense_fp6_linear,
        load_fp6_dense_weight_from_safetensors,
    )
    from b12x.quantization.mxfp6.fp6_checkpoint import (
        dequantize_linear_from_fp6,
    )

    device = torch.device("cuda")
    base = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}"
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(args.native), framework="pt", device="cpu") as handle:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stem = f"{base}.{projection}"
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                state[f"{stem}.{suffix}"] = handle.get_tensor(f"{stem}.{suffix}").to(device)
    dense: dict[str, torch.Tensor] = {}
    with safe_open(str(args.dense), framework="pt", device="cpu") as handle:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{projection}.weight"
            dense[projection] = handle.get_tensor(name).to(device)

    loaded = {
        projection: load_fp6_dense_weight_from_safetensors(
            state.__getitem__,
            f"{base}.{projection}",
            source_format="mxfp6_w6a8",
            device=device,
        )
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    decode = {}
    for projection in loaded:
        stem = f"{base}.{projection}"
        decoded = dequantize_linear_from_fp6(
            state[f"{stem}.weight"],
            state[f"{stem}.weight_scale"],
            fmt="e2m3",
            weight_scale_2=state[f"{stem}.weight_scale_2"],
        )
        decode[projection] = _metrics(decoded, dense[projection])

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(args.tokens, 4096, generator=generator, device=device, dtype=torch.bfloat16)
    gate = dense_fp6_linear(x, loaded["gate_proj"])
    up = dense_fp6_linear(x, loaded["up_proj"])
    middle = F.silu(gate.float()).to(torch.bfloat16) * up
    dense_native = dense_fp6_linear(middle, loaded["down_proj"])
    bf16_reference = F.linear(
        (F.silu(F.linear(x, dense["gate_proj"]).float()).to(torch.bfloat16)
         * F.linear(x, dense["up_proj"])),
        dense["down_proj"],
    )

    # vLLM loads W13 as [gate, up].  The pinned B12X plugin swaps halves
    # before prepare_weights because the native fused kernel consumes W31.
    w1 = torch.cat(
        (state[f"{base}.up_proj.weight"], state[f"{base}.gate_proj.weight"]),
        dim=0,
    ).unsqueeze(0).contiguous()
    w1_scale = torch.cat(
        (state[f"{base}.up_proj.weight_scale"], state[f"{base}.gate_proj.weight_scale"]),
        dim=0,
    ).unsqueeze(0).contiguous()
    w2 = state[f"{base}.down_proj.weight"].unsqueeze(0).contiguous()
    w2_scale = state[f"{base}.down_proj.weight_scale"].unsqueeze(0).contiguous()
    weight_plan = get_fp6_moe_weight_plan(
        source_format="mxfp6_e2m3",
        activation="silu",
        num_experts=1,
        hidden_size=4096,
        intermediate_size=2048,
    )
    ones = torch.ones(1, dtype=torch.float32, device=device)
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        w1_fp4=w1,
        w1_blockscale=w1_scale,
        w1_global_scale=ones,
        a1_gscale=ones,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=ones,
        a2_gscale=ones,
        params_dtype=torch.bfloat16,
    )
    method = B12XFP6MoEMethod(prepared, weight_plan)
    fused = method.apply(
        x,
        torch.ones(args.tokens, 1, dtype=torch.float32, device=device),
        torch.zeros(args.tokens, 1, dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()

    payload = {
        "schema": "glm53-trellis-mxf.mxfp6-bridge-canary.v1",
        "native": str(args.native),
        "dense": str(args.dense),
        "layer": args.layer,
        "expert": args.expert,
        "tokens": args.tokens,
        "seed": args.seed,
        "decoded_vs_dense": decode,
        "dense_native_vs_bf16_reference": _metrics(dense_native, bf16_reference),
        "fused_native_vs_dense_native": _metrics(fused, dense_native),
        "fused_native_vs_bf16_reference": _metrics(fused, bf16_reference),
        "w13_runtime_order": "up-then-gate (W31), matching the pinned v79 plugin half-swap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
