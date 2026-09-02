"""Causal layer-by-layer re-encode with confirmation-role KLD re-anchors (the ShapleyMCG install path).

For layer l = 0..L-1: capture layer l's expert Hessians (and routed samples) on the CURRENT model, in which
layers < l are already quantized, over the fit windows (early-exit forward after layer l); quantize layer l
under the tier map (GPTQ, rotation) with those Hessians; install; every `reanchor` layers score the
confirmation role and log it.  Then score the requested panels once.

usage: run_causal.py --arm B7 --device cuda:0 --rotation had16 --tiermap ARMS/alloc-.../tiermap.json --panels selection,final,wikitext --a4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, HESS, LOGS, TEACHER, load_seal, panel_ids, role_ids, receipt, write_json, log_line, sha256_json
from hessian import HessianCollector
from kld import score_student, save_report
from model_io import load_model, num_layers, num_experts, moe_block
from nvfp4 import NVFP4Config, MXFP6Config
import run_arm


class StopForward(Exception):
    pass


@torch.no_grad()
def capture_layer_hessian(model, layer, ids, device, out_dir: Path):
    col = HessianCollector(model, [layer], device)
    # early exit right after this layer's MoE block
    h = moe_block(model, layer).register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopForward()))
    for i in range(len(ids)):
        try:
            model(input_ids=torch.from_numpy(ids[i:i + 1]).to(device), use_cache=False)
        except StopForward:
            pass
    h.remove()
    col.remove()
    col.save(out_dir)
    stats = {"min_tokens": int(col.counts[layer].min()), "median_tokens": float(col.counts[layer].float().median()), "mismatches": col.mismatch}
    del col
    torch.cuda.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rotation", default="had16")
    ap.add_argument("--permutation", default="none")
    ap.add_argument("--tiermap", default=None)
    ap.add_argument("--tier", default="T1_nvfp4")
    ap.add_argument("--method", default="gptq")
    ap.add_argument("--panels", default="selection,final,wikitext")
    ap.add_argument("--a4", action="store_true")
    ap.add_argument("--reanchor", type=int, default=4)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--search-grid", type=int, default=8)
    a = ap.parse_args()
    out_dir = ARMS / a.arm; out_dir.mkdir(parents=True, exist_ok=True)
    hess_dir = out_dir / "causal-hessians"; hess_dir.mkdir(exist_ok=True)
    log = LOGS / f"arm-{a.arm}.log"
    seal = load_seal()
    fit_ids = role_ids(seal, "fit")
    conf_ids = role_ids(seal, "confirmation")
    device = a.device
    nv = NVFP4Config(search_grid=a.search_grid); mx = MXFP6Config()
    t0 = time.time()
    model = load_model(BF16, device)
    L, E = num_layers(model), num_experts(model)
    tiermap = json.load(open(a.tiermap)) if a.tiermap else None
    log_line(log, f"[{a.arm}] causal re-encode: rotation={a.rotation} tiermap={a.tiermap} reanchor every {a.reanchor} layers")
    # run_arm.process_layer reads Hessians from HESS/layer-XXX; point it at our causal dir per layer
    run_arm.HESS = hess_dir
    transforms = {}
    reanchors = []
    record = {} if a.save_weights else None
    for l in range(L):
        t1 = time.time()
        stats = capture_layer_hessian(model, l, fit_ids, device, hess_dir)
        tiers_for_unit = (lambda e, p: a.tier) if tiermap is None else (lambda e, p, lm=tiermap[str(l)]: lm[str(e)][p])
        losses, tr = run_arm.process_layer(model, l, a, nv, mx, tiers_for_unit, device, None, record)
        transforms.update(tr)
        if record is not None:
            save_file(record[l], str(out_dir / f"weights-layer-{l:03d}.safetensors")); record.pop(l)
        log_line(log, f"[{a.arm}] layer {l}: hessian {stats} ; quantized ({time.time()-t1:.0f}s)")
        if (l + 1) % a.reanchor == 0 or l == L - 1:
            rep = score_student(model, conf_ids, TEACHER / "confirmation", device=device)
            reanchors.append({"after_layer": l, "confirmation_mean_kld": rep["mean_kld"], "top1": rep["top1_agreement"]})
            log_line(log, f"[{a.arm}] re-anchor after layer {l}: confirmation KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f}")
    write_json(out_dir / "reanchors.json", reanchors)
    results = {}
    for panel in a.panels.split(","):
        ids = panel_ids(panel, seal)
        rep = score_student(model, ids, TEACHER / panel, device=device)
        rep = save_report(rep, out_dir / f"kld-{panel}-a16.json", out_dir / f"token-kld-{panel}-a16.npy")
        results[f"{panel}/a16"] = rep["mean_kld"]
        log_line(log, f"[{a.arm}] {panel} W4A16: mean KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f} p99 {rep['p99_token_kld']:.3f}")
        if a.a4:
            from actquant import ActQuantHooks
            hooks = ActQuantHooks(model, transforms)
            rep4 = score_student(model, ids, TEACHER / panel, device=device)
            hooks.remove()
            rep4 = save_report(rep4, out_dir / f"kld-{panel}-a4.json", out_dir / f"token-kld-{panel}-a4.npy")
            results[f"{panel}/a4"] = rep4["mean_kld"]
            log_line(log, f"[{a.arm}] {panel} W4A4 : mean KLD {rep4['mean_kld']:.5f} top1 {rep4['top1_agreement']:.4f}")
    receipt(out_dir / "receipt.json", "causal-arm", arm=a.arm, rotation=a.rotation, tiermap=a.tiermap, method=a.method, panels=a.panels.split(","),
            a4=a.a4, reanchor_every=a.reanchor, reanchors=reanchors, results=results, seal_sha256=seal["_sha256"], seconds=time.time() - t0,
            config_sha256=sha256_json(vars(a)))
    log_line(log, f"[{a.arm}] done in {time.time()-t0:.0f}s: {results}")


if __name__ == "__main__":
    main()
