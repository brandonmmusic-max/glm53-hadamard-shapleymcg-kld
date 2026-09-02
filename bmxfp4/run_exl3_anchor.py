"""Score a turboderp EXL3 checkpoint reconstructed to BF16 (E1a/E1b anchors).

Two variants: --scope full (experts + attention q/k/v/o + lm_head from the EXL3 reconstruction, router/embeddings/
norms BF16 = the deployable EXL3 model) and --scope experts (only routed experts replaced; attention BF16 = the
routed-expert-bytes-parity row).

usage: run_exl3_anchor.py --arm E1b --recon /media/.../exl3-recon/5.0bpw --scope full --device cuda:0 --panels selection,final
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, LOGS, TEACHER, load_seal, panel_ids, receipt, log_line
from kld import score_student, save_report
from model_io import load_model, num_layers, num_experts, moe_block


@torch.no_grad()
def install(model, recon: Path, scope: str, device: str):
    n_attn = n_exp = 0
    for layer in range(num_layers(model)):
        t = load_file(str(recon / f"layer-{layer:03d}.safetensors"))
        blk = moe_block(model, layer)
        for e in range(num_experts(model)):
            for p in ("gate_proj", "up_proj", "down_proj"):
                w = t[f"experts.{e}.{p}"].to(device)
                dst = getattr(blk.experts[e], p).weight.data
                assert dst.shape == w.shape, (layer, e, p, dst.shape, w.shape)
                dst.copy_(w.to(dst.dtype)); n_exp += 1
        if scope == "full":
            attn = model.model.layers[layer].self_attn
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"self_attn.{p}"
                if key in t:
                    dst = getattr(attn, p).weight.data
                    w = t[key].to(device)
                    assert dst.shape == w.shape, (layer, p, dst.shape, w.shape)
                    dst.copy_(w.to(dst.dtype)); n_attn += 1
    n_head = 0
    if scope == "full" and (recon / "lm_head.safetensors").exists():
        w = load_file(str(recon / "lm_head.safetensors"))["lm_head"].to(device)
        dst = model.lm_head.weight.data
        assert dst.shape == w.shape
        dst.copy_(w.to(dst.dtype)); n_head = 1
    return {"experts": n_exp, "attention": n_attn, "lm_head": n_head}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--scope", default="full", choices=["full", "experts"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--panels", default="selection,final")
    a = ap.parse_args()
    out = ARMS / a.arm; out.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"arm-{a.arm}.log"
    seal = load_seal()
    t0 = time.time()
    model = load_model(BF16, a.device)
    counts = install(model, a.recon, a.scope, a.device)
    manifest = json.load(open(a.recon / "manifest.json"))
    log_line(log, f"[{a.arm}] installed {counts} from {a.recon} scope={a.scope} quant={manifest.get('quantization_config')}")
    results = {}
    for panel in a.panels.split(","):
        ids = panel_ids(panel, seal)
        rep = score_student(model, ids, TEACHER / panel, device=a.device)
        rep = save_report(rep, out / f"kld-{panel}-a16.json", out / f"token-kld-{panel}-a16.npy")
        results[f"{panel}/a16"] = rep["mean_kld"]
        log_line(log, f"[{a.arm}] {panel}: mean KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f} p99 {rep['p99_token_kld']:.3f} ({rep['seconds']:.0f}s)")
    receipt(out / "receipt.json", "exl3-anchor", arm=a.arm, recon=str(a.recon), scope=a.scope, installed=counts, quantization_config=manifest.get("quantization_config"),
            results=results, seal_sha256=seal["_sha256"], seconds=time.time() - t0)


if __name__ == "__main__":
    main()
