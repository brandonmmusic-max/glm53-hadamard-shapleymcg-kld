"""Score stock NVFP4 checkpoints (ModelOpt: nvidia/Qwen3-30B-A3B-NVFP4; LLM-Compressor: RedHatAI/Qwen3-30B-A3B-NVFP4)
through the same KLD harness as every other arm: dequantize the packed E2M1 weights with their E4M3 block scales and
FP32 global scale to BF16, install them into the BF16 model, score the sealed panels.

Two scopes per checkpoint:
  full          every quantized Linear (attention q/k/v/o + expert gate/up/down) installed        -> arm <name>
  experts       only the routed experts installed (attention stays BF16, like B5/B7n)             -> arm <name>-experts
W4A4 emulation: NVFP4 dynamic per-16-block activation quantization on every installed Linear (attention too for
`full`), through the same fake-quant used for the pilot's A4 rows.

Formats (verified per checkpoint by reconstruction error against the BF16 weights and reported in the receipt):
  ModelOpt:            weight (u8 packed), weight_scale (e4m3 [N, K/16]), weight_scale_2 (f32)  -> w = q * sf * ws2
  compressed-tensors:  weight_packed (u8), weight_scale (e4m3 [N, K/16]), weight_global_scale (f32) -> w = q * sf / gs
Nibble order is resolved empirically (low-nibble-first vs high-nibble-first, whichever reconstructs the BF16 weight).

usage: stock_nvfp4.py --ckpt models/nvidia_Qwen3-30B-A3B-NVFP4 --name S1-nvidia --device cuda:0 --panels selection,final,wikitext --a4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, LOGS, TEACHER, load_seal, panel_ids, receipt, write_json, log_line, sha256_file
from kld import score_student, save_report
from model_io import load_model, num_layers, num_experts, moe_block, PROJS
from nvfp4 import nvfp4_quantize, NVFP4Config

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def unpack_e2m1(packed: torch.Tensor, low_first: bool) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    a, b = (lo, hi) if low_first else (hi, lo)
    def dec(n):
        mag = E2M1.to(n.device)[n & 0x7]
        return torch.where((n & 0x8) != 0, -mag, mag)
    out = torch.stack([dec(a), dec(b)], dim=-1)          # [..., K/2, 2]
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


class Ckpt:
    def __init__(self, d: Path):
        self.d = d
        idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.map = idx
        self.handles = {}
        keys = set(idx)
        self.fmt = "modelopt" if any(k.endswith("weight_scale_2") for k in keys) else "ct"
        self.keys = keys

    def get(self, key):
        f = self.map[key]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.d / f), framework="pt", device="cpu")
        return self.handles[f].get_tensor(key)

    def has(self, prefix):
        return (prefix + (".weight" if self.fmt == "modelopt" else ".weight_packed")) in self.keys and (prefix + ".weight_scale") in self.keys

    def dequant(self, prefix, low_first, device):
        if self.fmt == "modelopt":
            q = self.get(prefix + ".weight").to(device)
            sf = self.get(prefix + ".weight_scale").to(device).view(torch.float8_e4m3fn).float()
            gs = self.get(prefix + ".weight_scale_2").to(device).float()
            w = unpack_e2m1(q, low_first)
            return (w.reshape(w.shape[0], -1, 16) * sf[..., None] * gs).reshape(w.shape)
        q = self.get(prefix + ".weight_packed").to(device)
        sf = self.get(prefix + ".weight_scale").to(device).view(torch.float8_e4m3fn).float()
        gs = self.get(prefix + ".weight_global_scale").to(device).float()
        w = unpack_e2m1(q, low_first)
        return (w.reshape(w.shape[0], -1, 16) * sf[..., None] / gs).reshape(w.shape)

    def input_scale(self, prefix):
        k = prefix + (".input_scale" if self.fmt == "modelopt" else ".input_global_scale")
        return float(self.get(k)) if k in self.keys else None


class A4Hooks:
    """NVFP4 dynamic per-16 activation fake-quant on the given Linear modules (same fake-quant as actquant.ACT_CFG)."""
    def __init__(self, modules, cfg):
        self.h = []
        for m in modules:
            self.h.append(m.register_forward_pre_hook(self._mk(cfg)))

    @staticmethod
    def _mk(cfg):
        def hook(module, args):
            x = args[0]
            if x.numel() == 0:
                return None
            xq = nvfp4_quantize(x.reshape(-1, x.shape[-1]).float(), cfg).reshape(x.shape).to(x.dtype)
            return (xq,) + tuple(args[1:])
        return hook

    def remove(self):
        for h in self.h:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--panels", default="selection,final,wikitext")
    ap.add_argument("--a4", action="store_true")
    ap.add_argument("--scopes", default="full,experts")
    a = ap.parse_args()
    seal = load_seal()
    device = a.device
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    model = load_model(BF16, device)
    L, E = num_layers(model), num_experts(model)
    log = LOGS / f"arm-{a.name}.log"
    log_line(log, f"[{a.name}] ckpt {a.ckpt.name} format={ck.fmt}; model loaded {time.time()-t0:.0f}s")
    # resolve nibble order on one expert matrix against the BF16 original
    ref = moe_block(model, 0).experts[0].gate_proj.weight.float()
    errs = {}
    for lf in (True, False):
        w = ck.dequant("model.layers.0.mlp.experts.0.gate_proj", lf, device)
        errs[lf] = float(((w - ref) ** 2).mean().sqrt() / ref.pow(2).mean().sqrt())
    low_first = min(errs, key=errs.get)
    log_line(log, f"[{a.name}] nibble order low_first={low_first}; relative RMS reconstruction error {errs[low_first]:.4f} (other order {errs[not low_first]:.4f})")
    assert errs[low_first] < 0.3, "dequantization convention wrong"
    # snapshot BF16 originals we may overwrite (attention) for the experts-only scope
    bf16_attn = {}
    quantized = {"attn": [], "experts": []}
    rel = {"attn": [], "experts": []}
    for l in range(L):
        layer = model.model.layers[l]
        for p in ATTN:
            lin = getattr(layer.self_attn, p)
            prefix = f"model.layers.{l}.self_attn.{p}"
            if ck.has(prefix):
                bf16_attn[(l, p)] = lin.weight.data.clone().cpu()
                w = ck.dequant(prefix, low_first, device)
                rel["attn"].append(float(((w - lin.weight.float()) ** 2).mean().sqrt() / lin.weight.float().pow(2).mean().sqrt()))
                lin.weight.data.copy_(w.to(lin.weight.dtype)); quantized["attn"].append(lin)
        blk = moe_block(model, l)
        for e, ex in enumerate(blk.experts):
            for p in PROJS:
                lin = getattr(ex, p)
                prefix = f"model.layers.{l}.mlp.experts.{e}.{p}"
                if ck.has(prefix):
                    w = ck.dequant(prefix, low_first, device)
                    if e == 0 and p == "gate_proj":
                        rel["experts"].append(float(((w - lin.weight.float()) ** 2).mean().sqrt() / lin.weight.float().pow(2).mean().sqrt()))
                    lin.weight.data.copy_(w.to(lin.weight.dtype)); quantized["experts"].append(lin)
        other = [k for k in ck.keys if k.startswith(f"model.layers.{l}.") and ("mlp.gate." in k or ".shared_expert" in k) and ("scale" in k)]
        if l == 0 and other:
            log_line(log, f"[{a.name}] note: layer 0 has other quantized keys: {other[:4]}")
    log_line(log, f"[{a.name}] installed attn {len(quantized['attn'])} matrices (mean rel err {sum(rel['attn'])/max(len(rel['attn']),1):.4f}), experts {len(quantized['experts'])} matrices (mean rel err layer-e0-gate {sum(rel['experts'])/max(len(rel['experts']),1):.4f})")
    cfg = NVFP4Config(scale_search=False, range_max_search=False, tensor_scale_iters=0)
    results = {}
    for scope in a.scopes.split(","):
        arm = a.name if scope == "full" else f"{a.name}-experts"
        if scope == "experts":
            for (l, p), w in bf16_attn.items():
                getattr(model.model.layers[l].self_attn, p).weight.data.copy_(w.to(device))
        out_dir = ARMS / arm; out_dir.mkdir(parents=True, exist_ok=True)
        for panel in a.panels.split(","):
            ids = panel_ids(panel, seal)
            rep = score_student(model, ids, TEACHER / panel, device=device)
            rep = save_report(rep, out_dir / f"kld-{panel}-a16.json", out_dir / f"token-kld-{panel}-a16.npy")
            results[f"{arm}/{panel}/a16"] = rep["mean_kld"]
            log_line(log, f"[{arm}] {panel} W4A16: mean KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f} p99 {rep['p99_token_kld']:.3f}")
            if a.a4:
                mods = quantized["experts"] + (quantized["attn"] if scope == "full" else [])
                hooks = A4Hooks(mods, cfg)
                rep4 = score_student(model, ids, TEACHER / panel, device=device)
                hooks.remove()
                rep4 = save_report(rep4, out_dir / f"kld-{panel}-a4.json", out_dir / f"token-kld-{panel}-a4.npy")
                results[f"{arm}/{panel}/a4"] = rep4["mean_kld"]
                log_line(log, f"[{arm}] {panel} W4A4 : mean KLD {rep4['mean_kld']:.5f} top1 {rep4['top1_agreement']:.4f}")
        receipt(out_dir / "receipt.json", "stock-nvfp4", checkpoint=str(a.ckpt), format=ck.fmt, scope=scope, low_first=low_first, recon_rel_err=errs,
                installed={k: len(v) for k, v in quantized.items()}, config_sha256=sha256_file(a.ckpt / "config.json"), seal_sha256=seal["_sha256"],
                results={k: v for k, v in results.items() if k.startswith(arm + "/")}, seconds=time.time() - t0)
    log_line(log, f"[{a.name}] done in {time.time()-t0:.0f}s: {results}")


if __name__ == "__main__":
    main()
