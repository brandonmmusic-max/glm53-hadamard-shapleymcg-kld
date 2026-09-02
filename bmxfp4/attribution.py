"""Path-integrated (Aumann-Shapley) attribution of measured end-to-end KLD to every routed-expert unit.

Path: W(alpha) = W_src + alpha * (W_end - W_src) for ALL routed-expert weights simultaneously (straight
line from the BF16 teacher to the provisional quantized endpoint, e.g. uniform NVFP4+Had16).  For each
Gauss-Legendre node alpha_k the model is set to W(alpha_k), the KLD(teacher || student) over the fit windows
is computed, and d KLD / d W_u is obtained by autograd for every expert matrix u.  The Aumann-Shapley
share of unit u is  a_u = sum_k w_k <grad_u(alpha_k), delta_u>,  and sum_u a_u equals KLD(end) - KLD(src)
up to quadrature error (the remainder is reported explicitly, as the ShapleyMCG repository does).

Memory: W_src and W_end are kept on the host (bf16); per node the model is set to W(alpha); the expert
weights of `chunk` layers at a time require grad, their deltas live on the GPU, and each gradient is
consumed (dot with its delta) by a post-accumulate hook and freed immediately, so peak GPU memory is
model + activations + one chunk of deltas.

usage: attribution.py --endpoint ARMS/B5-prov --device cuda:0 --windows 0-15 --out ARMS/attrib-B5prov/part-0.json
then:  attribution.py --merge ARMS/attrib-B5prov --out ARMS/attrib-B5prov/attribution.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, TEACHER, LOGS, load_seal, role_ids, write_json, log_line
from model_io import load_model, num_layers, num_experts, moe_block, get_expert_weights, PROJS


def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2   # nodes and weights on [0, 1]


def load_endpoint_layer(endpoint: Path, layer: int):
    t = load_file(str(endpoint / f"weights-layer-{layer:03d}.safetensors"))
    w_in = t["in"]        # [E, 2*I, K]  (gate rows then up rows), bf16
    w_mid = t["mid"]      # [E, K, I]
    I = w_in.shape[1] // 2
    return {"gate_proj": w_in[:, :I].contiguous(), "up_proj": w_in[:, I:].contiguous(), "down_proj": w_mid.contiguous()}


def kld_loss(student_logits, teacher_logits):
    tl = F.log_softmax(teacher_logits[:-1].float(), dim=-1)
    sl = F.log_softmax(student_logits[:-1].float(), dim=-1)
    return (tl.exp() * (tl - sl)).sum(-1).mean()


def merge(merge_dir: Path, out: Path):
    parts = sorted(merge_dir.glob("part-*.json"))
    acc = None; nwin = 0; src_sum = 0.0; end_sum = 0.0
    for p in parts:
        d = json.load(open(p)); n = len(d["windows"]); nwin += n
        arr = np.array(d["attribution"]) * n
        acc = arr if acc is None else acc + arr
        src_sum += d["kld_src"] * n; end_sum += d["kld_end"] * n
    attribution = acc / nwin   # window-weighted mean (parts may have different window counts)
    kld_src, kld_end = src_sum / nwin, end_sum / nwin
    d0 = json.load(open(parts[0]))
    res = {"units": d0["units"], "attribution": attribution.tolist(), "kld_src": kld_src, "kld_end": kld_end, "windows": nwin,
           "sum_attribution": float(attribution.sum()), "remainder": float((kld_end - kld_src) - attribution.sum()),
           "parts": [str(p) for p in parts], "nodes": d0["nodes"], "role": d0["role"], "endpoint": d0["endpoint"]}
    write_json(out, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("units", "attribution", "parts")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--windows", default="0-15")
    ap.add_argument("--role", default="fit")
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", type=Path, default=None, help="merge part-*.json in this dir into --out")
    ap.add_argument("--probe", action="store_true", help="one window, one node, report peak memory and exit")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, a.out); return

    seal = load_seal()
    ids_all = role_ids(seal, a.role)
    lo, hi = [int(x) for x in a.windows.split("-")]
    ids = ids_all[lo:hi + 1]
    log = LOGS / f"attrib-{a.endpoint.name}-{a.windows}.log"
    device = a.device
    t0 = time.time()
    model = load_model(BF16, device)
    L, E = num_layers(model), num_experts(model)
    src = {l: {p: get_expert_weights(model, l, p).cpu() for p in PROJS} for l in range(L)}      # bf16, host
    end = {l: load_endpoint_layer(a.endpoint, l) for l in range(L)}                           # bf16, host
    log_line(log, f"loaded model + endpoint ({time.time()-t0:.0f}s); windows {lo}-{hi} of {a.role}; nodes {a.nodes}; chunk {a.chunk}")
    nodes, weights = gauss_legendre(a.nodes)
    if a.probe:
        nodes, weights, ids = nodes[:1], weights[:1], ids[:1]
    units = [(l, e, p) for l in range(L) for e in range(E) for p in PROJS]
    attrib = torch.zeros(L, E, len(PROJS), dtype=torch.float64)
    teacher = [load_file(str(TEACHER / a.role / f"row-{lo + i:03d}.safetensors"))["logits"] for i in range(len(ids))]

    def set_alpha(alpha):
        with torch.no_grad():
            for l in range(L):
                blk = moe_block(model, l)
                for p in PROJS:
                    s = src[l][p].to(device, non_blocking=True).float(); e_ = end[l][p].to(device, non_blocking=True).float()
                    w = (s + alpha * (e_ - s)).to(torch.bfloat16)
                    for ei, ex in enumerate(blk.experts):
                        getattr(ex, p).weight.data.copy_(w[ei])
        torch.cuda.synchronize(device)

    def measure(alpha):
        set_alpha(alpha)
        with torch.no_grad():
            vals = []
            for i in range(len(ids)):
                lg = model(input_ids=torch.from_numpy(ids[i:i + 1]).to(device), use_cache=False).logits[0]
                vals.append(float(kld_loss(lg, teacher[i].to(device))))
        return float(np.mean(vals))

    kld_src = measure(0.0)
    kld_end = measure(1.0)
    log_line(log, f"KLD src={kld_src:.6f} end={kld_end:.6f} (fit windows {lo}-{hi}, n={len(ids)})")

    # per-layer activation checkpointing for the gradient passes (Qwen3-MoE has no dropout, so train() only
    # switches the checkpointing branch on; attention_dropout is 0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for k, (alpha, wk) in enumerate(zip(nodes, weights)):
        set_alpha(float(alpha))
        t1 = time.time()
        for c0 in range(0, L, a.chunk):
            layers = list(range(c0, min(c0 + a.chunk, L)))
            params, handles = [], []
            for l in layers:
                blk = moe_block(model, l)
                for pi, p in enumerate(PROJS):
                    # [E, n, k] delta kept in bf16 on the GPU (12 layers of fp32 deltas = 29 GB, too much next to the model);
                    # bf16 rounding of the difference is ~2^-8 relative, far below the quadrature error
                    d_all = (end[l][p].to(device).float() - src[l][p].to(device).float()).to(torch.bfloat16)
                    for ei, ex in enumerate(blk.experts):
                        prm = getattr(ex, p).weight
                        prm.requires_grad_(True); params.append(prm)

                        def hook(param, l=l, ei=ei, pi=pi, d=d_all[ei], wk=wk):
                            attrib[l, ei, pi] += wk * float((param.grad.float() * d).sum())
                            param.grad = None
                        handles.append(prm.register_post_accumulate_grad_hook(hook))
            for i in range(len(ids)):
                with torch.enable_grad():
                    lg = model(input_ids=torch.from_numpy(ids[i:i + 1]).to(device), use_cache=False).logits[0]
                    loss = kld_loss(lg, teacher[i].to(device)) / len(ids)
                    loss.backward()
                del lg, loss
            for h in handles:
                h.remove()
            for prm in params:
                prm.requires_grad_(False); prm.grad = None
            del d_all
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        log_line(log, f"node {k+1}/{len(nodes)} alpha={alpha:.4f} done ({time.time()-t1:.0f}s); running sum={float(attrib.sum()):.6f}; peak GPU {peak:.1f} GiB")

    total = float(attrib.sum())
    out = {"units": [f"{l}.{e}.{p}" for (l, e, p) in units], "attribution": attrib.reshape(-1).tolist(),
           "kld_src": kld_src, "kld_end": kld_end, "sum_attribution": total, "remainder": (kld_end - kld_src) - total,
           "windows": list(range(lo, lo + len(ids))), "role": a.role, "nodes": int(len(nodes)), "endpoint": str(a.endpoint), "seconds": time.time() - t0,
           "probe": a.probe}
    write_json(a.out, out)
    log_line(log, f"done: sum attribution {total:.6f} vs KLD(end)-KLD(src) {kld_end - kld_src:.6f}; remainder {(kld_end - kld_src) - total:+.6f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
