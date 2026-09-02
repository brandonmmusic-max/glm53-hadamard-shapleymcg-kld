"""Learn a block-16 orthogonal transform per (layer, input kind) by Cayley parameterization.

R = R0 * (I - A)(I + A)^-1, A skew-symmetric 16x16 (120 free params), A = 0 at init, R0 in {I, Had16}.
Objective: Hessian-weighted reconstruction error of the projection stack under the NVFP4 quantizer with
MSE-optimal scales in the loop (straight-through estimators), i.e.  sum_e tr(D H' D^T) with
D = Q(W R) - W R and H' = R^T H R, on a random subset of experts per step (fit-role Hessians).
Selection between inits is by the same objective on ALL experts after training (fit role only; the
selection role decides between the finished arms via KLD, see run_arm.py).

usage: learn_rotation.py --device cuda:0 --init had16 --out ARMS/learnedR-had16 --steps 150 --layers 0-47
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
from model_io import load_model, get_expert_weights, num_layers
from nvfp4 import nvfp4_quantize, NVFP4Config
from rotation import hadamard16, cayley, apply_block_rotation, rotate_hessian


def objective(w, h, R, cfg):
    wr = apply_block_rotation(w, R)
    hr = rotate_hessian(h, R)
    d = nvfp4_quantize(wr, cfg) - wr
    return torch.einsum("enk,ekl,enl->e", d, hr, d).sum()


def learn(w, h, r0, steps, lr, cfg, batch, device, seed):
    e = w.shape[0]
    A = torch.zeros(16, 16, device=device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    r0d = r0.to(device)
    for step in range(steps):
        idx = torch.randperm(e, generator=g)[:batch].to(device)
        R = cayley(A, r0d)
        loss = objective(w[idx], h[idx], R, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return cayley(A, r0d).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default="identity", choices=["identity", "had16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--search-grid", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"learnR-{a.init}-{a.layers}.log"
    cfg = NVFP4Config(search_grid=a.search_grid)
    device = a.device
    t0 = time.time()
    model = load_model(BF16, device)
    lo, hi = [int(x) for x in a.layers.split("-")]
    r0 = torch.eye(16) if a.init == "identity" else hadamard16()
    summary = {}
    for layer in range(lo, hi + 1):
        hes = load_file(str(HESS / f"layer-{layer:03d}.safetensors"))
        res = {}
        tensors = {}
        for kind, projs, H in (("in", ("gate_proj", "up_proj"), hes["h_in"]), ("mid", ("down_proj",), hes["h_mid"])):
            w = torch.cat([get_expert_weights(model, layer, p) for p in projs], dim=1).float()
            h = H.to(device).float()
            t1 = time.time()
            with torch.no_grad():
                base_id = float(objective(w, h, torch.eye(16, device=device), cfg))
                base_r0 = float(objective(w, h, r0.to(device), cfg))
            R = learn(w, h, r0, a.steps, a.lr, cfg, a.batch, device, seed=20260902 + layer)
            with torch.no_grad():
                learned = float(objective(w, h, R, cfg))
            res[kind] = {"identity": base_id, "init": base_r0, "learned": learned, "gain_vs_identity": 1 - learned / base_id, "seconds": time.time() - t1}
            tensors[f"R_{kind}"] = R.cpu().contiguous()
            del w, h
        save_file(tensors, str(out / f"layer-{layer:03d}.safetensors"))
        summary[layer] = res
        log_line(log, f"layer {layer}: in gain {res['in']['gain_vs_identity']:+.3%} (init {1-res['in']['init']/res['in']['identity']:+.3%}), mid gain {res['mid']['gain_vs_identity']:+.3%}")
    receipt(out / f"receipt-{a.layers}.json", "learned-rotation", init=a.init, steps=a.steps, lr=a.lr, batch=a.batch, layers=a.layers, summary=summary, seconds=time.time() - t0)


if __name__ == "__main__":
    main()
