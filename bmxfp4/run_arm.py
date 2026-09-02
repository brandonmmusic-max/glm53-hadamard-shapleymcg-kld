"""Quantize the routed experts under an arm configuration, install the pseudo-quant weights, score KLD.

Modes
  arm         : quantize every layer (default tier T1 or a tier map), score on --panels (W4A16, optionally W4A4)
  candidates  : for every layer and tier in --ladder, run GPTQ and record per-unit Hessian-weighted output
                error + exact payload bytes (no model patching); output ARMS/<arm>/candidates/layer-*.json
  calib       : quantize ONLY --calib-layer to T1 (identity), score on the confirmation role -> per-layer KLD sensitivity

usage examples
  run_arm.py --arm B3p --device cuda:1 --method gptq --panels selection
  run_arm.py --arm B5  --device cuda:2 --method gptq --rotation had16 --panels selection --a4
  run_arm.py --arm B7  --device cuda:1 --method gptq --tiermap ARMS/alloc-E1b/tiermap.json --panels selection,final --a4 --save-weights
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, BF16, HESS, LOGS, TEACHER, load_seal, panel_ids, receipt, write_json, log_line, sha256_json
from gptq import gptq_quantize, hessian_weighted_error
from kld import score_student, save_report
from model_io import load_model, num_layers, num_experts, get_expert_weights, set_expert_weights
from nvfp4 import GroupQuantizer, NVFP4Config, MXFP6Config, fp8_quantize, nvfp4_quantize, mxfp6_quantize, tier_payload_bytes, sparse24_mask
from rotation import hadamard16, random_so16, apply_block_rotation, rotate_hessian, hessian_permutation, inverse_permutation

PROJ_KIND = {"gate_proj": "in", "up_proj": "in", "down_proj": "mid"}


def make_rotation(spec: str, layer: int, kind: str, device, learned_dir: Path | None):
    """spec: identity | had16 | random:SEED | learned:DIR"""
    if spec == "identity":
        return None
    if spec == "had16":
        return hadamard16().to(device)
    if spec == "hadfull":
        # QuaRot-style full-width Hadamard on the expert input (2048 = 2^11) and on the down_proj input (768 = 12 x 64);
        # folded into the weights offline, applied to activations online (fast Hadamard transform) -> W4A4 outlier spreading
        from rotation import hadamard_n
        return hadamard_n(2048 if kind == "in" else 768).to(device)
    if spec.startswith("random:"):
        seed = int(spec.split(":")[1])
        return random_so16(seed * 1000 + layer * 2 + (0 if kind == "in" else 1)).to(device)
    if spec.startswith("learned:"):
        d = Path(spec.split(":", 1)[1])
        t = load_file(str(d / f"layer-{layer:03d}.safetensors"))
        return t[f"R_{kind}"].to(device)
    raise ValueError(spec)


def quantize_stack(w: torch.Tensor, h: torch.Tensor, tier: str, method: str, nv: NVFP4Config, mx: MXFP6Config):
    """w [E,N,K], h [E,K,K] (already in the transform basis). Returns dequantized w and per-expert loss."""
    if tier == "T4_bf16":
        return w.clone(), torch.zeros(w.shape[0], device=w.device)
    if tier == "T3_fp8":
        wq = fp8_quantize(w)
        return wq, hessian_weighted_error(w, wq, h)
    if method == "rtn":
        if tier == "T1_nvfp4":
            wq = nvfp4_quantize(w, nv)
        elif tier == "T2_mxfp6":
            wq = mxfp6_quantize(w, mx)
        elif tier == "T0_sparse_nvfp4":
            from nvfp4 import sparse48_mask, SPARSE_PATTERN
            m = sparse48_mask(w) if SPARSE_PATTERN == "4:8" else sparse24_mask(w)
            wq = nvfp4_quantize(w * m, nv) * m
        else:
            raise KeyError(tier)
        return wq, hessian_weighted_error(w, wq, h)
    from nvfp4 import SPARSE_PATTERN
    q = GroupQuantizer(tier, nv, mx)
    wq, _ = gptq_quantize(w, h, q, sparse24=(tier == "T0_sparse_nvfp4"), sparse_pattern=SPARSE_PATTERN)
    return wq, hessian_weighted_error(w, wq, h)


def process_layer(model, layer, args, nv, mx, tiers_for_unit, device, learned_dir, record):
    """Quantize one layer's experts in place. tiers_for_unit(expert, proj) -> tier. Returns per-unit losses dict."""
    hes = load_file(str(HESS / f"layer-{layer:03d}.safetensors"))
    h_in = hes["h_in"].to(device)
    h_mid = hes["h_mid"].to(device)
    E = h_in.shape[0]
    losses = {}
    transforms = {}
    for kind, projs, H in (("in", ("gate_proj", "up_proj"), h_in), ("mid", ("down_proj",), h_mid)):
        # stack the projections that share this input: rows concatenated
        ws = [get_expert_weights(model, layer, p) for p in projs]
        w = torch.cat(ws, dim=1).float()                      # [E, N_total, K]
        K = w.shape[-1]
        Ht = H
        perm = None
        if args.permutation != "none":
            perm = hessian_permutation(torch.diagonal(H, dim1=-2, dim2=-1).mean(0), H.mean(0) if args.permutation == "corr_cluster" else None, mode=args.permutation)
            w = w[:, :, perm]
            Ht = H[:, perm][:, :, perm]
        R = make_rotation(args.rotation, layer, kind, device, learned_dir)
        if R is not None:
            w = apply_block_rotation(w, R, group=R.shape[0])
            Ht = rotate_hessian(Ht, R, group=R.shape[0])
        transforms[(layer, kind)] = {"R": R, "perm": perm}
        # group experts by tier
        tier_of = [tiers_for_unit(e, projs[0]) for e in range(E)]
        wq = torch.empty_like(w)
        for tier in sorted(set(tier_of)):
            idx = torch.tensor([e for e in range(E) if tier_of[e] == tier], device=device)
            wq_t, loss_t = quantize_stack(w[idx], Ht[idx], tier, args.method, nv, mx)
            wq[idx] = wq_t
            for j, e in enumerate(idx.tolist()):
                losses[(e, kind)] = (tier, float(loss_t[j]))
        # back to the original basis
        if R is not None:
            wq = apply_block_rotation(wq, R.T, group=R.shape[0])
        if perm is not None:
            wq = wq[:, :, inverse_permutation(perm)]
        # split rows back into projections and install
        off = 0
        for p, wp in zip(projs, ws):
            n = wp.shape[1]
            set_expert_weights(model, layer, p, wq[:, off:off + n].to(wp.dtype))
            off += n
        if record is not None:
            record[layer] = record.get(layer, {})
            record[layer][kind] = wq.to(torch.bfloat16).cpu()
    del h_in, h_mid
    return losses, transforms


@torch.no_grad()
def expert_forward(x, wg, wu, wd):
    """Batched Qwen3 expert: x [E,S,K]; wg,wu [E,I,K]; wd [E,K,I] -> [E,S,K] (fp32)."""
    g = torch.einsum("esk,eik->esi", x, wg)
    u = torch.einsum("esk,eik->esi", x, wu)
    return torch.einsum("esi,eki->esk", torch.nn.functional.silu(g) * u, wd)


def candidates_layer(layer, model, args, nv, mx, device, ladder):
    """Per-unit per-tier value and bytes for allocation (no patching).

    value = route-weight^2-weighted mean squared FULL-EXPERT output error on stored routed samples when ONLY
    that unit (one projection of one expert) is quantized to the tier (others BF16) -> comparable across
    projections and tiers, all measured in the residual-stream space.  The per-projection GPTQ Hessian loss
    is also recorded for reference.
    """
    hes = load_file(str(HESS / f"layer-{layer:03d}.safetensors"))
    smp = load_file(str(HESS / f"samples-{layer:03d}.safetensors"))
    x = smp["x"].to(device).float()                      # [E, S, K]
    sw = smp["w"].to(device).float() ** 2                # route weight^2
    n = smp["n"].to(device)
    valid = (torch.arange(x.shape[1], device=device)[None, :] < n[:, None]).float() * sw
    valid = valid / valid.sum(dim=1, keepdim=True).clamp(min=1e-12)   # [E, S] normalized weights
    W = {p: get_expert_weights(model, layer, p).float() for p in ("gate_proj", "up_proj", "down_proj")}
    base = expert_forward(x, W["gate_proj"], W["up_proj"], W["down_proj"])
    out = {}
    for kind, projs, H in (("in", ("gate_proj", "up_proj"), hes["h_in"].to(device)), ("mid", ("down_proj",), hes["h_mid"].to(device))):
        R = make_rotation(args.rotation, layer, kind, device, None)
        for p in projs:
            w = W[p]
            Ht = H
            if R is not None:
                w = apply_block_rotation(w, R); Ht = rotate_hessian(H, R)
            shape = tuple(w.shape[1:])
            for tier in ladder:
                wq, loss = quantize_stack(w, Ht, tier, args.method, nv, mx)
                if R is not None:
                    wq = apply_block_rotation(wq, R.T)
                Wq = dict(W); Wq[p] = wq
                y = expert_forward(x, Wq["gate_proj"], Wq["up_proj"], Wq["down_proj"])
                resid = ((y - base) ** 2).sum(-1)                  # [E, S]
                value = (resid * valid).sum(1)                       # weighted mean per expert
                out.setdefault(p, {})[tier] = {"loss": [float(v) for v in value], "hessian_loss": [float(v) for v in loss], "bytes": tier_payload_bytes(tier, shape)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="arm", choices=["arm", "candidates", "calib"])
    ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"])
    ap.add_argument("--tier", default="T1_nvfp4")
    ap.add_argument("--tiermap", default=None, help="json {layer: {expert: {proj: tier}}} from allocate.py")
    ap.add_argument("--rotation", default="identity")
    ap.add_argument("--permutation", default="none", choices=["none", "diag_band", "corr_cluster"])
    ap.add_argument("--panels", default="selection")
    ap.add_argument("--a4", action="store_true", help="also score with NVFP4 activations at expert inputs")
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--layers", default=None, help="a-b subset (debug)")
    ap.add_argument("--calib-layer", type=int, default=None)
    ap.add_argument("--ladder", default="T0_sparse_nvfp4,T1_nvfp4,T2_mxfp6,T3_fp8,T4_bf16")
    ap.add_argument("--search-grid", type=int, default=8)
    a = ap.parse_args()

    out_dir = ARMS / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"arm-{a.arm}.log"
    seal = load_seal()
    nv = NVFP4Config(search_grid=a.search_grid)
    mx = MXFP6Config()
    device = a.device
    t0 = time.time()
    model = load_model(BF16, device)
    L = num_layers(model); E = num_experts(model)
    layers = list(range(L)) if a.layers is None else list(range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1))
    log_line(log, f"[{a.arm}] model loaded {time.time()-t0:.0f}s; mode={a.mode} method={a.method} tier={a.tier} rot={a.rotation} perm={a.permutation}")

    tiermap = json.load(open(a.tiermap)) if a.tiermap else None

    def tiers_for_unit_factory(layer):
        if a.mode == "calib":
            return lambda e, p: ("T1_nvfp4" if layer == a.calib_layer else "T4_bf16")
        if tiermap is None:
            return lambda e, p: a.tier
        lm = tiermap[str(layer)]
        return lambda e, p: lm[str(e)][p]

    if a.mode == "candidates":
        ladder = a.ladder.split(",")
        cdir = out_dir / "candidates"; cdir.mkdir(exist_ok=True)
        for layer in layers:
            t1 = time.time()
            res = candidates_layer(layer, model, a, nv, mx, device, ladder)
            write_json(cdir / f"layer-{layer:03d}.json", res)
            log_line(log, f"[{a.arm}] candidates layer {layer} ({time.time()-t1:.1f}s)")
        receipt(out_dir / "candidates-receipt.json", "candidates", arm=a.arm, ladder=ladder, method=a.method, rotation=a.rotation, layers=layers, seconds=time.time() - t0)
        return

    all_losses = {}
    transforms = {}
    record = {} if a.save_weights else None
    q_layers = [a.calib_layer] if a.mode == "calib" else layers
    for layer in q_layers:
        t1 = time.time()
        losses, tr = process_layer(model, layer, a, nv, mx, tiers_for_unit_factory(layer), device, None, record)
        transforms.update(tr)
        all_losses[layer] = {f"{e}.{k}": v for (e, k), v in losses.items()}
        if record is not None:
            save_file(record[layer], str(out_dir / f"weights-layer-{layer:03d}.safetensors")); record.pop(layer)
        log_line(log, f"[{a.arm}] layer {layer} quantized ({time.time()-t1:.1f}s) mean loss in={np.mean([v[1] for (e,k),v in losses.items() if k=='in']):.4g} mid={np.mean([v[1] for (e,k),v in losses.items() if k=='mid']):.4g}")
    write_json(out_dir / "unit-losses.json", {str(k): v for k, v in all_losses.items()})
    if a.rotation != "identity" or a.permutation != "none":
        save_file({f"{l}_{k}_R": (v["R"].cpu().contiguous() if v["R"] is not None else torch.eye(16)) for (l, k), v in transforms.items()} |
                  {f"{l}_{k}_perm": (v["perm"].cpu().contiguous() if v["perm"] is not None else torch.arange(1)) for (l, k), v in transforms.items()},
                  str(out_dir / "transforms.safetensors"))

    panels = a.panels.split(",") if a.mode != "calib" else ["confirmation"]
    results = {}
    for panel in panels:
        ids = panel_ids(panel, seal)
        rep = score_student(model, ids, TEACHER / panel, device=device)
        rep = save_report(rep, out_dir / f"kld-{panel}-a16.json", out_dir / f"token-kld-{panel}-a16.npy")
        results[f"{panel}/a16"] = rep["mean_kld"]
        log_line(log, f"[{a.arm}] {panel} W4A16: mean KLD {rep['mean_kld']:.5f} top1 {rep['top1_agreement']:.4f} p99 {rep['p99_token_kld']:.3f} ({rep['seconds']:.0f}s)")
        if a.a4:
            from actquant import ActQuantHooks
            hooks = ActQuantHooks(model, transforms)
            rep4 = score_student(model, ids, TEACHER / panel, device=device)
            hooks.remove()
            rep4 = save_report(rep4, out_dir / f"kld-{panel}-a4.json", out_dir / f"token-kld-{panel}-a4.npy")
            results[f"{panel}/a4"] = rep4["mean_kld"]
            log_line(log, f"[{a.arm}] {panel} W4A4 : mean KLD {rep4['mean_kld']:.5f} top1 {rep4['top1_agreement']:.4f} ({rep4['seconds']:.0f}s)")
    receipt(out_dir / "receipt.json", "arm", arm=a.arm, mode=a.mode, method=a.method, tier=a.tier, tiermap=a.tiermap, rotation=a.rotation,
            permutation=a.permutation, panels=panels, a4=a.a4, calib_layer=a.calib_layer, results=results, seal_sha256=seal["_sha256"],
            nvfp4_config=vars(nv), seconds=time.time() - t0, config_sha256=sha256_json(vars(a)))
    log_line(log, f"[{a.arm}] done in {time.time()-t0:.0f}s: {results}")


if __name__ == "__main__":
    main()
