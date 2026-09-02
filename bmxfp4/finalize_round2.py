"""Append the Round-2 (ShapleyMCG path) section to REPORT.md: rotation-calibration verdict, attribution
reconciliation, calibrated allocations, causal re-anchor trajectories, and the headline comparisons
(B7 vs E1b primary, B7a vs E1a, B7u control, calibration effect B7 vs B7u), W4A16 and W4A4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, ROOT
from finalize_report import rep, cmp, fmt, alloc_summary


def hist_line(name):
    s = alloc_summary(name)
    if s is None:
        return f"- {name}: n/a"
    h = s["histogram"]
    parts = ", ".join(f"{t.split('_')[0]} {v['units']}" for t, v in h.items() if v["units"])
    st = s.get("attribution_stats") or {}
    extra = f"; attribution: sum {st['sum_attribution']:.5f} of KLD_end {st['kld_end']:.5f}, remainder {st['remainder']:+.5f}, negative shares {st['negative_shares']}, fallback units {st['fallback_units']}" if st else "; uncalibrated proxy"
    return f"- {name}: {s['used_bpw']:.3f} bpw used of {s['budget_bpw']:.3f}; units {parts}{extra}"


def reanchors(name):
    p = ARMS / name / "reanchors.json"
    if not p.exists():
        return "n/a"
    r = json.load(open(p))
    return ", ".join(f"L{x['after_layer']}: {x['confirmation_mean_kld']:.5f}" for x in r)


def main():
    out = ["", "---", "", "# Round 2 — ShapleyMCG path (2026-09-02, after Brandon's 06:09 correction)", ""]
    out.append("Calibration data = Brandon's REAP calibration corpus (reap_recall_calib.jsonl, 12,228 samples, 4 balanced axes) packed into 64 x 2048-token windows (role calib; first 32 = calib-attrib) — NOT the sealed evaluation windows (DECISIONS #15). Path: Hessians + routed samples on calib (64 windows) -> calibrated block-16 rotation (calib samples) vs Had16 decided on the selection role -> provisional endpoint B5-prov2 (uniform NVFP4, Had16, GPTQ) -> path-integrated Aumann-Shapley attribution of measured end-to-end KLD (calib-attrib, 32 windows, 5 Gauss-Legendre nodes) -> calibrated exact-byte allocation with the 4:8 pair-structured sparse tier (PTX ISA 9.3 verified) -> causal layer-by-layer re-encode (Hessians recaptured on calib-attrib per layer) with confirmation-role re-anchors -> final/selection/WikiText KLD scored exactly as the EXL3 controls. Declared in DECISIONS.md items 7-16 before any allocation ran.")
    out.append("")
    out.append("## Rotation calibration verdict (selection role, W4A16, uniform NVFP4, GPTQ)")
    out.append(f"- B5L2 (calibrated-learned per layer, REAP calib) vs B5 (Had16): {fmt(cmp('B5L2', 'B5', 'selection'))}")
    lr = ARMS / "learnedR-cal2"
    recs = sorted(lr.glob("receipt-*.json"))
    if recs:
        sel = {}
        for r in recs:
            for l, v in json.load(open(r))["summary"].items():
                sel[v["selected"]] = sel.get(v["selected"], 0) + 1
        out.append(f"- per-layer basis chosen by the calibrated objective: {sel}")
    out.append("")
    att = ARMS / "attrib-B5prov2" / "attribution.json"
    if att.exists():
        a = json.load(open(att))
        out.append("## Attribution reconciliation (calib-attrib windows, REAP corpus)")
        out.append(f"- KLD(src)={a['kld_src']:.6f}, KLD(end)={a['kld_end']:.6f}, sum of unit shares={a['sum_attribution']:.6f}, remainder={a['remainder']:+.6f} ({100*a['remainder']/max(a['kld_end']-a['kld_src'],1e-12):+.1f}% of the endpoint KLD), windows={a['windows']}, nodes={a['nodes']}")
        import numpy as np
        arr = np.array(a["attribution"])
        out.append(f"- shares: negative {int((arr < 0).sum())} of {arr.size}; top-1% of units carry {100*np.sort(arr)[::-1][:arr.size//100].sum()/max(arr.sum(),1e-12):.1f}% of the total")
        out.append("")
    out.append("## Allocations (18,432 units; T0 = 4:8 pair-structured sparse NVFP4)")
    for n in ("alloc-B7", "alloc-B7a", "alloc-B7u", "alloc-B7ua", "alloc-B7n", "alloc-B7un"):
        out.append(hist_line(n))
    out.append("")
    out.append("## Causal re-encode re-anchors (confirmation role, W4A16, cumulative)")
    for n in ("B7", "B7a", "B7u", "B7ua", "B7n", "B7un"):
        out.append(f"- {n}: {reanchors(n)}")
    out.append("")
    out.append("## Headline comparisons (final role, 25 x 2048; paired bootstrap over windows)")
    out.append(f"- PRIMARY B7 (Shapley-calibrated, E1b bytes) vs E1b (EXL3 5.0 bpw, full): {fmt(cmp('B7', 'E1b', 'final'))}")
    out.append(f"- B7 vs E1b-experts (EXL3 experts only, attention BF16): {fmt(cmp('B7', 'E1b-experts', 'final'))}")
    out.append(f"- B7a (Shapley-calibrated, E1a bytes) vs E1a (EXL3 4.0 bpw, full): {fmt(cmp('B7a', 'E1a', 'final'))}")
    out.append(f"- B7a vs E1a-experts: {fmt(cmp('B7a', 'E1a-experts', 'final'))}")
    out.append(f"- B7u (uncalibrated proxy control, E1b bytes) vs E1b: {fmt(cmp('B7u', 'E1b', 'final'))}")
    out.append(f"- B7ua (uncalibrated proxy control, E1a bytes) vs E1a: {fmt(cmp('B7ua', 'E1a', 'final'))}")
    out.append(f"- Calibration effect B7 vs B7u (same candidates, same bytes): {fmt(cmp('B7', 'B7u', 'final'))}")
    out.append(f"- Calibration effect B7a vs B7ua: {fmt(cmp('B7a', 'B7ua', 'final'))}")
    out.append(f"- B7 vs B5-final (uniform NVFP4 Had16, 4.5 bpw): {fmt(cmp('B7', 'B5-final', 'final'))}")
    out.append(f"- B7 W4A4 vs E1b: {fmt(cmp('B7', 'E1b', 'final', mode_a='a4'))}")
    out.append("")
    out.append("## POST-HOC ablation (DECISIONS #18, decided after the results above): no 4:8 tier, ladder {NVFP4, MXFP6, FP8, BF16}, E1b bytes, causal")
    out.append(f"- B7n (Shapley-calibrated, no T0) vs E1b: {fmt(cmp('B7n', 'E1b', 'final'))}")
    out.append(f"- B7n vs E1b-experts: {fmt(cmp('B7n', 'E1b-experts', 'final'))}")
    out.append(f"- B7un (proxy, no T0) vs E1b: {fmt(cmp('B7un', 'E1b', 'final'))}")
    out.append(f"- B7n vs B7un (calibration effect without the sparse tier): {fmt(cmp('B7n', 'B7un', 'final'))}")
    out.append(f"- B7n vs B7 (cost of the 4:8 tier under Shapley allocation): {fmt(cmp('B7n', 'B7', 'final'))}")
    out.append(f"- B7n vs B8c (pilot post-hoc: proxy, no T0, one-shot, fit-window Hessians): {fmt(cmp('B7n', 'B8c', 'final'))}")
    out.append(f"- B7n vs B5-final (uniform NVFP4 Had16 4.5 bpw): {fmt(cmp('B7n', 'B5-final', 'final'))}")
    out.append(f"- B7n W4A4 vs E1b: {fmt(cmp('B7n', 'E1b', 'final', mode_a='a4'))}")
    out.append(f"- [selection] B7n vs E1b: {fmt(cmp('B7n', 'E1b', 'selection'))}; B7un vs E1b: {fmt(cmp('B7un', 'E1b', 'selection'))}")
    out.append(f"- B7a W4A4 vs E1a: {fmt(cmp('B7a', 'E1a', 'final', mode_a='a4'))}")
    out.append("")
    out.append("## Same comparisons on the selection role (16 x 2048) and WikiText (10 x 2048)")
    for panel in ("selection", "wikitext"):
        out.append(f"- [{panel}] B7 vs E1b: {fmt(cmp('B7', 'E1b', panel))}; B7a vs E1a: {fmt(cmp('B7a', 'E1a', panel))}; B7u vs E1b: {fmt(cmp('B7u', 'E1b', panel))}; B7ua vs E1a: {fmt(cmp('B7ua', 'E1a', panel))}")
    out.append("")
    out.append("## Stock NVFP4 anchors (B1/B2 of the plan): dequantized through the same harness; final role W4A16 unless stated")
    out.append("Scopes: `full` = every quantized Linear installed (attention + experts, as shipped); `-experts` = routed experts only, attention BF16 (the scope of every BMXFP4 arm). Routed-expert bytes of stock NVFP4 = 4.500 bpw = B5r; B7n carries 5.029 bpw.")
    for st in ("S1-nvidia", "S2-redhat"):
        for sc in ("", "-experts"):
            arm = st + sc
            out.append(f"- {arm} vs E1b: {fmt(cmp(arm, 'E1b', 'final'))}")
        out.append(f"- {st}-experts vs E1b-experts: {fmt(cmp(st + '-experts', 'E1b-experts', 'final'))}")
        out.append(f"- B5r (uniform NVFP4 Had16 GPTQ REAP, same bytes) vs {st}-experts: {fmt(cmp('B5r', st + '-experts', 'final'))}")
        out.append(f"- B7n (Shapley multi-tier, +0.53 bpw) vs {st}-experts: {fmt(cmp('B7n', st + '-experts', 'final'))}")
        out.append(f"- B7 (main run, 4:8 tier) vs {st}-experts: {fmt(cmp('B7', st + '-experts', 'final'))}")
        out.append(f"- W4A4: {st} full A4 vs {st} full A16: {fmt(cmp(st, st, 'final', mode_a='a4'))}; {st}-experts A4: {fmt(cmp(st + '-experts', st + '-experts', 'final', mode_a='a4'))}")
        out.append(f"- [selection] {st}-experts vs E1b: {fmt(cmp(st + '-experts', 'E1b', 'selection'))}; B5r vs {st}-experts: {fmt(cmp('B5r', st + '-experts', 'selection'))}; [wikitext] B5r vs {st}-experts: {fmt(cmp('B5r', st + '-experts', 'wikitext'))}")
    out.append("")
    out.append("## W4A4 lever: full-width Hadamard (QuaRot-style, 2048 on expert inputs, 12x64 Kronecker on down_proj inputs) vs block-16 Hadamard, uniform NVFP4, GPTQ, REAP Hessians")
    out.append(f"- B5h vs B5r W4A16: {fmt(cmp('B5h', 'B5r', 'final'))}")
    out.append(f"- B5h vs B5r W4A4: {fmt(cmp('B5h', 'B5r', 'final', mode_a='a4', mode_b='a4'))}")
    out.append(f"- B5h W4A4 vs B5h W4A16 (activation cost after full rotation): {fmt(cmp('B5h', 'B5h', 'final', mode_a='a4'))}; B5r: {fmt(cmp('B5r', 'B5r', 'final', mode_a='a4'))}")
    out.append(f"- B5h W4A4 vs E1b: {fmt(cmp('B5h', 'E1b', 'final', mode_a='a4'))}; [selection] B5h vs B5r A4: {fmt(cmp('B5h', 'B5r', 'selection', mode_a='a4', mode_b='a4'))}")
    out.append("")
    p = ROOT / "REPORT.md"
    txt = p.read_text()
    marker = "# Round 2 — ShapleyMCG path"
    if marker in txt:
        txt = txt[:txt.index("\n---\n\n" + marker)] if ("\n---\n\n" + marker) in txt else txt[:txt.index(marker)]
    p.write_text(txt.rstrip("\n") + "\n" + "\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
