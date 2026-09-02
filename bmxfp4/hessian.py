"""Per-expert input Hessians (route-weighted X^T X) for routed experts of Qwen3-MoE via hooks.

For each (layer, expert): H_in [2048, 2048] from the tokens routed to the expert (input to gate/up),
and H_mid [768, 768] from the activated intermediate (input to down_proj), each row weighted by the
routing weight^p of that token for that expert (p = 2 matches the prior ShapleyMCG production setting;
p = 0 is the plain GPTQ Hessian).  Accumulated in fp32 on the model's GPU for a subset of layers per
pass (~2.45 GB per layer), normalized by the summed weights, and saved to disk.

Routing weights are recomputed from the router logits in the same way as the transformers
Qwen3MoeSparseMoeBlock (softmax over experts in fp32, top-k, optional renormalization), and matched to
the expert calls by the ascending token order transformers uses when it gathers a given expert's tokens.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from model_io import moe_block, num_experts


class HessianCollector:
    def __init__(self, model, layers: list[int], device: str, route_power: float = 2.0):
        self.model = model
        self.layers = layers
        self.device = device
        self.p = route_power
        e = num_experts(model)
        blk = moe_block(model, layers[0])
        k_in = blk.experts[0].gate_proj.weight.shape[1]
        k_mid = blk.experts[0].down_proj.weight.shape[1]
        self.top_k = model.config.num_experts_per_tok
        self.norm_topk = getattr(model.config, "norm_topk_prob", False)
        self.h_in = {l: torch.zeros(e, k_in, k_in, dtype=torch.float32, device=device) for l in layers}
        self.h_mid = {l: torch.zeros(e, k_mid, k_mid, dtype=torch.float32, device=device) for l in layers}
        self.wsum = {l: torch.zeros(e, dtype=torch.float64, device=device) for l in layers}
        self.counts = {l: torch.zeros(e, dtype=torch.int64, device=device) for l in layers}
        # routed-input samples (first n_samples rows per expert) for full-expert residual scoring
        self.n_samples = 256
        self.x_samples = {l: torch.zeros(e, self.n_samples, k_in, dtype=torch.bfloat16, device=device) for l in layers}
        self.w_samples = {l: torch.zeros(e, self.n_samples, dtype=torch.float32, device=device) for l in layers}
        self.s_count = {l: torch.zeros(e, dtype=torch.int64, device=device) for l in layers}
        self.pending = {}  # (layer, expert) -> weights vector in expert-call order
        self.mismatch = 0
        self.handles = []
        for l in layers:
            b = moe_block(model, l)
            self.handles.append(b.gate.register_forward_hook(self._router_hook(l)))
            for i, ex in enumerate(b.experts):
                self.handles.append(ex.gate_proj.register_forward_pre_hook(self._mk(l, i, "in")))
                self.handles.append(ex.down_proj.register_forward_pre_hook(self._mk(l, i, "mid")))

    def _router_hook(self, l):
        def hook(module, args, output):
            logits = output.reshape(-1, output.shape[-1]).float()
            probs = F.softmax(logits, dim=-1)
            w, idx = torch.topk(probs, self.top_k, dim=-1)
            if self.norm_topk:
                w = w / w.sum(dim=-1, keepdim=True)
            # transformers 4.52 dispatch order: torch.where(one_hot(selected).permute(2,1,0)[e]) over a
            # [top_k, tokens] mask -> k-major, then ascending token.  Replicate exactly.
            for e in range(logits.shape[-1]):
                parts = []
                for kk in range(self.top_k):
                    tok = (idx[:, kk] == e).nonzero(as_tuple=True)[0]
                    if tok.numel():
                        parts.append(w[tok, kk])
                self.pending[(l, e)] = torch.cat(parts) if parts else torch.zeros(0, device=w.device)
        return hook

    def _mk(self, l, i, kind):
        def hook(module, args):
            x = args[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.to(torch.float32)
            if x.shape[0] == 0:
                return
            we = self.pending.get((l, i))
            if we is None or we.numel() != x.shape[0]:
                self.mismatch += 1
                wt = torch.ones(x.shape[0], device=x.device)
            else:
                wt = we.pow(self.p) if self.p != 0 else torch.ones_like(we)
            xw = x * wt.sqrt()[:, None]
            if kind == "in":
                self.h_in[l][i].addmm_(xw.T, xw)
                self.wsum[l][i] += wt.double().sum()
                self.counts[l][i] += x.shape[0]
                c = int(self.s_count[l][i])
                if c < self.n_samples:
                    take = min(self.n_samples - c, x.shape[0])
                    self.x_samples[l][i, c:c + take] = x[:take].to(torch.bfloat16)
                    self.w_samples[l][i, c:c + take] = (we[:take] if (we is not None and we.numel() == x.shape[0]) else torch.ones(take, device=x.device))
                    self.s_count[l][i] += take
            else:
                self.h_mid[l][i].addmm_(xw.T, xw)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for l in self.layers:
            c = self.wsum[l].clamp(min=1e-12).to(torch.float32)
            save_file(
                {"h_in": (self.h_in[l] / c[:, None, None]).cpu(), "h_mid": (self.h_mid[l] / c[:, None, None]).cpu(),
                 "counts": self.counts[l].cpu(), "wsum": self.wsum[l].cpu()},
                str(out_dir / f"layer-{l:03d}.safetensors"),
            )
            save_file(
                {"x": self.x_samples[l].cpu(), "w": self.w_samples[l].cpu(), "n": self.s_count[l].cpu()},
                str(out_dir / f"samples-{l:03d}.safetensors"),
            )


@torch.no_grad()
def collect_hessians(model, ids: np.ndarray, layers: list[int], out_dir: Path, device: str, route_power: float = 2.0, log=print):
    col = HessianCollector(model, layers, device, route_power)
    t0 = time.time()
    for i in range(len(ids)):
        model(input_ids=torch.from_numpy(ids[i:i + 1]).to(device), use_cache=False)
        if (i + 1) % 8 == 0:
            log(f"  hessian pass window {i+1}/{len(ids)} ({time.time()-t0:.0f}s) mismatches={col.mismatch}")
    col.remove()
    col.save(out_dir)
    stats = {l: {"min_tokens": int(col.counts[l].min()), "median_tokens": float(col.counts[l].float().median()), "max_tokens": int(col.counts[l].max())} for l in layers}
    stats["route_weight_mismatches"] = col.mismatch
    del col
    torch.cuda.empty_cache()
    return stats
