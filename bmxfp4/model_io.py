"""Load Qwen3-30B-A3B (HF transformers, bf16, sdpa) and access/patch routed-expert weights.

Unit = (layer, expert, projection) with projection in {gate_proj, up_proj, down_proj}.
gate/up: [768, 2048] (input dim 2048 = hidden);  down: [2048, 768] (input dim 768 = expert intermediate).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJS = ("gate_proj", "up_proj", "down_proj")


def load_model(path: str | Path, device: str = "cuda:0", attn: str = "sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, attn_implementation=attn, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    return model


def moe_block(model, layer: int):
    return model.model.layers[layer].mlp


def num_layers(model) -> int:
    return len(model.model.layers)


def num_experts(model) -> int:
    return len(moe_block(model, 0).experts)


def expert_linear(model, layer: int, expert: int, proj: str):
    return getattr(moe_block(model, layer).experts[expert], proj)


@torch.no_grad()
def get_expert_weights(model, layer: int, proj: str) -> torch.Tensor:
    """Stack all experts' weights of one projection: [E, N, K] (bf16, on model device)."""
    blk = moe_block(model, layer)
    return torch.stack([getattr(ex, proj).weight.data for ex in blk.experts], dim=0)


@torch.no_grad()
def set_expert_weights(model, layer: int, proj: str, w: torch.Tensor):
    blk = moe_block(model, layer)
    for i, ex in enumerate(blk.experts):
        getattr(ex, proj).weight.data.copy_(w[i].to(getattr(ex, proj).weight.dtype))


@torch.no_grad()
def snapshot_layer(model, layer: int) -> dict[str, torch.Tensor]:
    return {p: get_expert_weights(model, layer, p).clone() for p in PROJS}


@torch.no_grad()
def restore_layer(model, layer: int, snap: dict[str, torch.Tensor]):
    for p, w in snap.items():
        set_expert_weights(model, layer, p, w)


def expert_shapes(model) -> dict[str, tuple[int, int]]:
    blk = moe_block(model, 0)
    return {p: tuple(getattr(blk.experts[0], p).weight.shape) for p in PROJS}


def routed_expert_param_count(model) -> int:
    shp = expert_shapes(model)
    per_expert = sum(a * b for a, b in shp.values())
    return per_expert * num_experts(model) * num_layers(model)


def describe(model) -> dict:
    return {
        "layers": num_layers(model),
        "experts_per_layer": num_experts(model),
        "top_k": getattr(model.config, "num_experts_per_tok", None),
        "expert_shapes": {p: list(s) for p, s in expert_shapes(model).items()},
        "routed_expert_params": routed_expert_param_count(model),
        "hidden": model.config.hidden_size,
        "moe_intermediate": getattr(model.config, "moe_intermediate_size", None),
    }
