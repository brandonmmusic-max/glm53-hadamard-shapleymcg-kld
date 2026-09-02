"""Calibrated rotation learning: block-16 Cayley rotations per layer (R_in shared by gate/up, R_mid for down)
optimized against the FULL-EXPERT output error on the stored calibration samples (fit role, 256 routed tokens
per expert, route-weight^2 weighted) with the NVFP4 quantizer (MSE scales, E4M3 rounding, STE) in the loop:

  L(R_in, R_mid) = sum_e sum_t w_et || f_e(x_t; Q(W_g R_in)R_in^T, Q(W_u R_in)R_in^T, Q(W_d R_mid)R_mid^T) - f_e(x_t; W) ||^2

Both inits (I, Had16) are trained; per layer the best of {I, Had16, learned-from-I, learned-from-Had16} by the
same calibrated objective on ALL experts is selected (fit-role decision) and saved as R_in / R_mid.

usage: learn_rotation_v2.py --device cuda:0 --layers 0-23 --out ARMS/learnedR-cal --steps 80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, HESS, LOGS, receipt, log_line
from model_io import load_model, get_expert_weights
from nvfp4 import nvfp4_quantize, NVFP4Config
from rotation import hadamard16, cayley, apply_block_rotation


def expert_out(x, wg, wu, wd):
    g = torch.einsum("esk,eik->esi", x, wg)
    u = torch.einsum("esk,eik->esi", x, wu)
    return torch.einsum("esi,eki->esk", torch.nn.functional.silu(g) * u, wd)


def objective(x, wgt, base, wg, wu, wd, R_in, R_mid, cfg):
    qg = apply_block_rotation(nvfp4_quantize(apply_block_rotation(wg, R_in), cfg), R_in.T)
    qu = apply_block_rotation(nvfp4_quantize(apply_block_rotation(wu, R_in), cfg), R_in.T)
    qd = apply_block_rotation(nvfp4_quantize(apply_block_rotation(wd, R_mid), cfg), R_mid.T)
    y = expert_out(x, qg, qu, qd)
    return (((y - base) ** 2).sum(-1) * wgt).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--search-grid", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"learnR2-{a.layers}.log"
    cfg = NVFP4Config(search_grid=a.search_grid)
    device = a.device
    t0 = time.time()
    model = load_model(BF16, device)
    lo, hi = [int(x) for x in a.layers.split("-")]
    had = hadamard16().to(device); eye = torch.eye(16, device=device)
    summary = {}
    for layer in range(lo, hi + 1):
        smp = load_file(str(HESS / f"samples-{layer:03d}.safetensors"))
        x = smp["x"].to(device).float(); sw = smp["w"].to(device).float() ** 2; n = smp["n"].to(device)
        valid = (torch.arange(x.shape[1], device=device)[None, :] < n[:, None]).float() * sw
        wgt = valid / valid.sum(dim=1, keepdim=True).clamp(min=1e-12)
        wg = get_expert_weights(model, layer, "gate_proj").float(); wu = get_expert_weights(model, layer, "up_proj").float(); wd = get_expert_weights(model, layer, "down_proj").float()
        with torch.no_grad():
            base = expert_out(x, wg, wu, wd)
        E = x.shape[0]
        cands = {"identity": (eye, eye), "had16": (had, had)}
        t1 = time.time()
        for init_name, r0 in (("identity", eye), ("had16", had)):
            A_in = torch.zeros(16, 16, device=device, requires_grad=True); A_mid = torch.zeros(16, 16, device=device, requires_grad=True)
            opt = torch.optim.Adam([A_in, A_mid], lr=a.lr)
            g = torch.Generator(device="cpu").manual_seed(20260902 + layer)
            for step in range(a.steps):
                idx = torch.randperm(E, generator=g)[:a.batch].to(device)
                R_in = cayley(A_in, r0); R_mid = cayley(A_mid, r0)
                loss = objective(x[idx], wgt[idx], base[idx], wg[idx], wu[idx], wd[idx], R_in, R_mid, cfg)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                cands[f"learned-{init_name}"] = (cayley(A_in, r0).detach(), cayley(A_mid, r0).detach())
        with torch.no_grad():
            scores = {k: float(objective(x, wgt, base, wg, wu, wd, v[0], v[1], cfg)) for k, v in cands.items()}
        best = min(scores, key=scores.get)
        R_in, R_mid = cands[best]
        save_file({"R_in": R_in.cpu().contiguous(), "R_mid": R_mid.cpu().contiguous()}, str(out / f"layer-{layer:03d}.safetensors"))
        summary[layer] = {"scores": scores, "selected": best, "gain_vs_identity": 1 - scores[best] / scores["identity"], "seconds": time.time() - t1}
        log_line(log, f"layer {layer}: selected {best}; gain vs identity {summary[layer]['gain_vs_identity']:+.2%}; scores " + ", ".join(f"{k}={v:.4g}" for k, v in scores.items()))
        del x, sw, wgt, base, wg, wu, wd
        torch.cuda.empty_cache()
    receipt(out / f"receipt-{a.layers}.json", "learned-rotation-calibrated", steps=a.steps, lr=a.lr, batch=a.batch, layers=a.layers, summary=summary, seconds=time.time() - t0)


if __name__ == "__main__":
    main()
