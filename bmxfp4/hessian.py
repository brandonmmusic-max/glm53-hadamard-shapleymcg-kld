"""Per-expert input Hessians (X^T X) for routed experts of Qwen3-MoE via forward pre-hooks.

For each (layer, expert): H_in [2048, 2048] from the tokens routed to the expert (input to gate/up),
and H_mid [768, 768] from the activated intermediate (input to down_proj).  Accumulated in fp32 on
the model's GPU for a subset of layers per pass (memory: ~2.45 GB per layer), then saved to disk.
Token counts per expert are recorded so under-sampled experts can be flagged.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from model_io import moe_block, num_experts


class HessianCollector:
    def __init__(self, model, layers: list[int], device: str):
        self.model = model
        self.layers = layers
        self.device = device
        e = num_experts(model)
        blk = moe_block(model, layers[0])
        k_in = blk.experts[0].gate_proj.weight.shape[1]
        k_mid = blk.experts[0].down_proj.weight.shape[1]
        self.h_in = {l: torch.zeros(e, k_in, k_in, dtype=torch.float32, device=device) for l in layers}
        self.h_mid = {l: torch.zeros(e, k_mid, k_mid, dtype=torch.float32, device=device) for l in layers}
        self.counts = {l: torch.zeros(e, dtype=torch.int64, device=device) for l in layers}
        self.handles = []
        for l in layers:
            b = moe_block(model, l)
            for i, ex in enumerate(b.experts):
                self.handles.append(ex.gate_proj.register_forward_pre_hook(self._mk(l, i, "in")))
                self.handles.append(ex.down_proj.register_forward_pre_hook(self._mk(l, i, "mid")))

    def _mk(self, l, i, kind):
        def hook(module, args):
            x = args[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.to(torch.float32)
            if kind == "in":
                self.h_in[l][i].addmm_(x.T, x)
                self.counts[l][i] += x.shape[0]
            else:
                self.h_mid[l][i].addmm_(x.T, x)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for l in self.layers:
            # normalize by token count (per expert) so magnitudes are comparable across experts
            c = self.counts[l].clamp(min=1).to(torch.float32)
            save_file(
                {
                    "h_in": (self.h_in[l] / c[:, None, None]).cpu(),
                    "h_mid": (self.h_mid[l] / c[:, None, None]).cpu(),
                    "counts": self.counts[l].cpu(),
                },
                str(out_dir / f"layer-{l:03d}.safetensors"),
            )


@torch.no_grad()
def collect_hessians(model, ids: np.ndarray, layers: list[int], out_dir: Path, device: str, log=print):
    col = HessianCollector(model, layers, device)
    t0 = time.time()
    for i in range(len(ids)):
        model(input_ids=torch.from_numpy(ids[i:i + 1]).to(device), use_cache=False)
        if (i + 1) % 8 == 0:
            log(f"  hessian pass window {i+1}/{len(ids)} ({time.time()-t0:.0f}s)")
    col.remove()
    col.save(out_dir)
    stats = {l: {"min_tokens": int(col.counts[l].min()), "median_tokens": float(col.counts[l].float().median()), "max_tokens": int(col.counts[l].max())} for l in layers}
    del col
    torch.cuda.empty_cache()
    return stats
