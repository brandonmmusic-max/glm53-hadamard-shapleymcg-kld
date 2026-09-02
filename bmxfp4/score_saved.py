"""Score a saved expert weight set (weights-layer-NNN.safetensors, keys "in" = gate rows then up rows, "mid" = down)
with optional FP8 (E4M3, per-row scale) embeddings and/or lm_head, through the standard KLD harness.

usage: score_saved.py --arm B7n-fp8h --weights ARMS/B7n --heads emb,lm_head --rotation had16 --panels selection,final,wikitext --a4
       score_saved.py --arm E0-fp8h --heads emb,lm_head            (BF16 experts: head cost alone)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, LOGS, TEACHER, load_seal, panel_ids, receipt, log_line, sha256_json
from kld import score_student, save_report
from model_io import load_model, num_layers, moe_block, set_expert_weights
from nvfp4 import fp8_quantize, fp8_payload_bytes, bf16_payload_bytes
from rotation import hadamard16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--heads", default="", help="comma list of emb,lm_head to quantize to FP8 E4M3 per-row")
    ap.add_argument("--rotation", default="had16", help="rotation used by the saved weights (for the W4A4 activation path)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--panels", default="selection,final,wikitext")
    ap.add_argument("--a4", action="store_true")
    a = ap.parse_args()
    seal = load_seal()
    device = a.device
    out_dir = ARMS / a.arm; out_dir.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"arm-{a.arm}.log"
    t0 = time.time()
    model = load_model(BF16, device)
    L = num_layers(model)
    if a.weights:
        for l in range(L):
            t = load_file(str(a.weights / f"weights-layer-{l:03d}.safetensors"))
            w_in = t["in"].to(device); w_mid = t["mid"].to(device)
            I = w_in.shape[1] // 2
            set_expert_weights(model, l, "gate_proj", w_in[:, :I]); set_expert_weights(model, l, "up_proj", w_in[:, I:]); set_expert_weights(model, l, "down_proj", w_mid)
        log_line(log, f"[{a.arm}] installed saved expert weights from {a.weights} ({time.time()-t0:.0f}s)")
    heads = [h for h in a.heads.split(",") if h]
    bytes_info = {}
    with torch.no_grad():
        for h in heads:
            mod = model.model.embed_tokens if h == "emb" else model.lm_head
            w = mod.weight.data
            wq = fp8_quantize(w.float()).to(w.dtype)
            err = float(((wq.float() - w.float()) ** 2).mean().sqrt() / w.float().pow(2).mean().sqrt())
            mod.weight.data.copy_(wq)
            bytes_info[h] = {"bf16_bytes": bf16_payload_bytes(w.shape), "fp8_bytes": fp8_payload_bytes(w.shape), "rel_rms_err": err, "shape": list(w.shape)}
            log_line(log, f"[{a.arm}] {h} -> FP8 E4M3 per-row: rel RMS err {err:.4f}; bytes {bf16_payload_bytes(w.shape)/1e9:.3f} GB -> {fp8_payload_bytes(w.shape)/1e9:.3f} GB")
    transforms = {}
    if a.rotation == "had16":
        R = hadamard16().to(device)
        for l in range(L):
            transforms[(l, "in")] = {"R": R, "perm": None}; transforms[(l, "mid")] = {"R": R, "perm": None}
    results = {}
    for panel in a.panels.split(","):
        ids = panel_ids(panel, seal)
        rep = score_student(model, ids, TEACHER / panel, device=device)
        rep = save_report(rep, out_dir / f"kld-{panel}-a16.json", out_dir / f"token-kld-{panel}-a16.npy")
        results[f"{panel}/a16"] = rep["mean_kld"]
        log_line(log, f"[{a.arm}] {panel} W4A16: mean KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f} p99 {rep['p99_token_kld']:.3f}")
        if a.a4 and a.weights:
            from actquant import ActQuantHooks
            hooks = ActQuantHooks(model, transforms)
            rep4 = score_student(model, ids, TEACHER / panel, device=device)
            hooks.remove()
            rep4 = save_report(rep4, out_dir / f"kld-{panel}-a4.json", out_dir / f"token-kld-{panel}-a4.npy")
            results[f"{panel}/a4"] = rep4["mean_kld"]
            log_line(log, f"[{a.arm}] {panel} W4A4 : mean KLD {rep4['mean_kld']:.5f} top1 {rep4['top1_agreement']:.4f}")
    receipt(out_dir / "receipt.json", "saved-weights-arm", arm=a.arm, weights=str(a.weights), heads=heads, heads_bytes=bytes_info, rotation=a.rotation,
            panels=a.panels.split(","), a4=a.a4, results=results, seal_sha256=seal["_sha256"], seconds=time.time() - t0, config_sha256=sha256_json(vars(a) | {"weights": str(a.weights)}))
    log_line(log, f"[{a.arm}] done in {time.time()-t0:.0f}s: {results}")


if __name__ == "__main__":
    main()
