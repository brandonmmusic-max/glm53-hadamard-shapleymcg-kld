"""Exact-byte multiple-choice knapsack over per-unit tier candidates.

Inputs: candidates/layer-*.json from run_arm.py --mode candidates:
   {proj: {tier: {"loss": [E floats], "bytes": int}}}
Value per (unit, tier) = loss (Hessian-weighted output error of the unit under that tier), optionally
multiplied by a per-layer calibration factor (KLD increase per unit of loss, measured on the confirmation
role by quantizing one layer at a time).  Minimize total value subject to sum(bytes) <= budget.

Solver: Lagrangian relaxation with bisection on lambda (the LP-relaxation / convex-hull greedy for MCKP),
then a final exact fill by marginal value per byte.  With ~18k units and 5 tiers this is exact up to one
unit's worth of bytes, which is far below the resolution of the comparison.

Budget: --budget-bytes (exact routed-expert payload bytes of the reference checkpoint) or --budget-bpw
(bits per routed-expert weight; converted using the unit element counts).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TIERS = ["T0_sparse_nvfp4", "T1_nvfp4", "T2_mxfp6", "T3_fp8", "T4_bf16"]


def expert_frequency(layer: int, hess_dir: Path | None):
    """Route-weight^2 mass share of each expert in the fit role (from the Hessian receipts), normalized to mean 1."""
    if hess_dir is None:
        return None
    from safetensors.torch import load_file
    w = load_file(str(hess_dir / f"layer-{layer:03d}.safetensors"))["wsum"].double().numpy()
    return w / max(w.mean(), 1e-12)


def load_candidates(cdir: Path, calib: dict | None, ladder: list[str], hess_dir: Path | None = None):
    units = []  # (layer, expert, proj)
    losses = []  # [U, T]
    bytes_ = []  # [U, T]
    elems = []
    files = sorted(cdir.glob("layer-*.json"))
    for f in files:
        layer = int(f.stem.split("-")[1])
        d = json.load(open(f))
        factor = 1.0 if calib is None else calib.get(str(layer), 1.0)
        freq = expert_frequency(layer, hess_dir)
        for proj, tiers in d.items():
            E = len(tiers[ladder[0]]["loss"])
            for e in range(E):
                units.append((layer, e, proj))
                fe = 1.0 if freq is None else float(freq[e])
                losses.append([tiers[t]["loss"][e] * factor * fe for t in ladder])
                bytes_.append([tiers[t]["bytes"] for t in ladder])
                elems.append(tiers["T4_bf16"]["bytes"] // 2 if "T4_bf16" in tiers else tiers[ladder[-1]]["bytes"] // 2)
    return units, np.array(losses, dtype=np.float64), np.array(bytes_, dtype=np.int64), np.array(elems, dtype=np.int64)


def solve(losses: np.ndarray, bytes_: np.ndarray, budget: int):
    U, T = losses.shape
    # Lagrangian: for lambda, choose per unit argmin(loss + lambda*bytes)
    def choose(lam):
        return np.argmin(losses + lam * bytes_, axis=1)
    lo, hi = 0.0, 1.0
    # find hi such that bytes(hi) <= budget
    while bytes_[np.arange(U), choose(hi)].sum() > budget:
        hi *= 2
        if hi > 1e12:
            raise RuntimeError("budget infeasible even at the cheapest tiers")
    if bytes_[np.arange(U), choose(0.0)].sum() <= budget:
        return choose(0.0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if bytes_[np.arange(U), choose(mid)].sum() <= budget:
            hi = mid
        else:
            lo = mid
    sel = choose(hi)
    used = bytes_[np.arange(U), sel].sum()
    # greedy fill: upgrade units with best (loss reduction / extra bytes) while budget remains
    slack = budget - used
    improved = True
    while improved and slack > 0:
        improved = False
        best = None
        for t in range(T):
            extra = bytes_[:, t] - bytes_[np.arange(U), sel]
            gain = losses[np.arange(U), sel] - losses[:, t]
            ok = (extra > 0) & (extra <= slack) & (gain > 0)
            if ok.any():
                ratio = np.where(ok, gain / np.maximum(extra, 1), -1)
                u = int(np.argmax(ratio))
                if best is None or ratio[u] > best[0]:
                    best = (ratio[u], u, t)
        if best is not None:
            _, u, t = best
            slack -= int(bytes_[u, t] - bytes_[u, sel[u]])
            sel[u] = t
            improved = True
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-bytes", type=int, default=None)
    ap.add_argument("--budget-bpw", type=float, default=None)
    ap.add_argument("--calib", type=Path, default=None, help="json {layer: factor}")
    ap.add_argument("--ladder", default=",".join(TIERS))
    ap.add_argument("--name", default="alloc")
    ap.add_argument("--freq-hessians", type=Path, default=None, help="hessians dir; weight each unit's value by its expert's route-weight^2 mass share")
    ap.add_argument("--attribution", type=Path, default=None, help="attribution.json (path-integrated per-unit KLD shares of the provisional endpoint)")
    ap.add_argument("--provisional-tier", default="T1_nvfp4", help="tier of the provisional endpoint the attribution was measured on")
    a = ap.parse_args()
    ladder = a.ladder.split(",")
    calib = json.load(open(a.calib)) if a.calib else None
    units, losses, bytes_, elems = load_candidates(a.candidates, calib, ladder, a.freq_hessians)
    attribution_stats = None
    if a.attribution:
        # Repository rule (qwen_services.py:1146-1171): per unit, scale = causal share / anchor proxy, applied to every
        # tier's proxy, then a non-negative per-unit offset so the objective is finite and non-negative; within-unit
        # ordering is preserved.  Units whose anchor proxy is ~0 (never routed in fit) fall back to the layer-median scale.
        att = json.load(open(a.attribution))
        share = {u: v for u, v in zip(att["units"], att["attribution"])}
        pt = ladder.index(a.provisional_tier)
        scaled = np.zeros_like(losses)
        layer_scales = {}
        raw_scale = np.full(len(units), np.nan)
        for i, (layer, e, p) in enumerate(units):
            s_u = share.get(f"{layer}.{e}.{p}", 0.0)
            anchor = losses[i, pt]
            if anchor > 1e-12:
                raw_scale[i] = s_u / anchor
                layer_scales.setdefault(layer, []).append(raw_scale[i])
        med = {l: float(np.median(v)) for l, v in layer_scales.items()}
        n_fallback = 0
        for i, (layer, e, p) in enumerate(units):
            sc = raw_scale[i]
            if not np.isfinite(sc):
                sc = med.get(layer, 1.0); n_fallback += 1
            row = losses[i] * sc
            off = max(0.0, -row.min())
            scaled[i] = row + off
        attribution_stats = {"kld_end": att["kld_end"], "sum_attribution": att["sum_attribution"], "remainder": att["remainder"],
                             "negative_shares": int(sum(1 for v in share.values() if v < 0)), "fallback_units": n_fallback}
        losses = scaled
    total_elems = int(elems.sum())
    budget = a.budget_bytes if a.budget_bytes is not None else int(a.budget_bpw * total_elems / 8)
    sel = solve(losses, bytes_, budget)
    U = len(units)
    used = int(bytes_[np.arange(U), sel].sum())
    tiermap = {}
    hist = {t: {"units": 0, "bytes": 0} for t in ladder}
    per_layer = {}
    per_proj = {}
    for i, ((layer, e, p), s) in enumerate(zip(units, sel)):
        tiermap.setdefault(str(layer), {}).setdefault(str(e), {})[p] = ladder[s]
        hist[ladder[s]]["units"] += 1
        hist[ladder[s]]["bytes"] += int(bytes_[i, s])
        per_layer.setdefault(str(layer), {t: 0 for t in ladder})[ladder[s]] += 1
        per_proj.setdefault(p, {t: 0 for t in ladder})[ladder[s]] += 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tiermap.json").write_text(json.dumps(tiermap))
    summary = {"name": a.name, "budget_bytes": budget, "used_bytes": used, "budget_bpw": 8.0 * budget / total_elems, "used_bpw": 8.0 * used / total_elems,
               "units": U, "total_loss": float(losses[np.arange(U), sel].sum()), "uniform_T1_loss": float(losses[:, ladder.index("T1_nvfp4")].sum()) if "T1_nvfp4" in ladder else None,
               "uniform_T1_bytes": int(bytes_[:, ladder.index("T1_nvfp4")].sum()) if "T1_nvfp4" in ladder else None,
               "histogram": hist, "per_layer": per_layer, "per_projection": per_proj, "calibrated": calib is not None, "ladder": ladder,
               "frequency_weighted": a.freq_hessians is not None, "attribution": str(a.attribution) if a.attribution else None,
               "attribution_stats": attribution_stats}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_layer",)}, indent=1))


if __name__ == "__main__":
    main()
