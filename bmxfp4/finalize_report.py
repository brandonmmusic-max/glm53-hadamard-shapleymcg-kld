"""Compose the final REPORT.md: headline, primary/secondary comparisons with CIs, ladder tables, allocation
histograms, rotation verdict, decisions, deviations, artifact index."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, ROOT, SEALS, HESS
from kld import paired_bootstrap


def rep(arm, panel, mode="a16"):
    p = ARMS / arm / f"kld-{panel}-{mode}.json"
    return json.load(open(p)) if p.exists() else None


def cmp(a, b, panel, mode_a="a16", mode_b="a16"):
    ra, rb = rep(a, panel, mode_a), rep(b, panel, mode_b)
    if ra is None or rb is None:
        return None
    bs = paired_bootstrap(ra["per_window_mean_kld"], rb["per_window_mean_kld"])
    return {"a": ra["mean_kld"], "b": rb["mean_kld"], "ratio": ra["mean_kld"] / rb["mean_kld"], **bs}


def fmt(c):
    if c is None:
        return "n/a"
    return f"{c['a']:.5f} vs {c['b']:.5f} (ratio {c['ratio']:.2f}x; diff {c['diff_mean']:+.5f}, 95% CI [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]{', excludes 0' if c['excludes_zero'] else ', includes 0'})"


def alloc_summary(name):
    p = ARMS / name / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    out = []
    b5 = json.load(open(SEALS / "bytes-exl3-5.0bpw.json")); b4 = json.load(open(SEALS / "bytes-exl3-4.0bpw.json"))
    out.append("# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — results (2026-09-02)\n")
    out.append("Pre-registration: bmxfp4/PREREGISTRATION.md (committed before candidate generation). Decision log: DECISIONS.md. Method notes and declared deviations: REPORT_NOTES.md. Every arm directory under arms/ holds its receipt.json, per-panel kld-*.json, and per-token KLD arrays.\n")
    out.append(f"Budgets (exact routed-expert payload bytes from the EXL3 safetensors headers): E1b = {b5['routed_expert_payload_bytes']:,} B ({b5['routed_expert_bpw']:.3f} bpw), E1a = {b4['routed_expert_payload_bytes']:,} B ({b4['routed_expert_bpw']:.3f} bpw). Uniform NVFP4 = 4.500 bpw (n/2 + n/16 + 4 bytes per matrix).\n")

    out.append("## Headline comparisons (mean per-window KLD, W4A16 unless stated; paired bootstrap over windows)\n")
    rows = [
        ("PRIMARY (pre-registered): B8 five-tier proxy allocation at E1b bytes vs E1b (EXL3 5.0, full checkpoint), final role", cmp("B8", "E1b", "final")),
        ("PRIMARY, selection role", cmp("B8", "E1b", "selection")),
        ("Bytes-parity row: B8 vs E1b experts-only (attention BF16), final", cmp("B8", "E1b-experts", "final")),
        ("Secondary: B8a five-tier at E1a bytes vs E1a (EXL3 4.0 full), final", cmp("B8a", "E1a", "final")),
        ("Secondary: B8 W4A4 vs E1b, final", cmp("B8", "E1b", "final", "a4", "a16")),
        ("Rotation verdict (selection): B5 Had16 vs B3' control", cmp("B5", "B3p", "selection")),
        ("Rotation verdict (selection): B6h learned (Had init) vs B3'", cmp("B6h", "B3p", "selection")),
        ("Rotation verdict (selection): B6i learned (identity init) vs B3'", cmp("B6i", "B3p", "selection")),
        ("Permutation (selection): B5p Had16+diag_band vs B5", cmp("B5p", "B5", "selection")),
        ("POST-HOC (exploratory): B8c {NVFP4,MXFP6,BF16} at E1b bytes vs E1b full, final", cmp("B8c", "E1b", "final")),
        ("POST-HOC: B8c vs E1b full, selection", cmp("B8c", "E1b", "selection")),
        ("POST-HOC: B8c vs E1b experts-only (bytes parity), final", cmp("B8c", "E1b-experts", "final")),
        ("POST-HOC: B8c vs B5 uniform NVFP4 Had16 (4.5 bpw), final", cmp("B8c", "B5-final", "final")),
        ("POST-HOC: B8b (route-frequency-weighted) vs B8c, final", cmp("B8b", "B8c", "final")),
        ("POST-HOC: B8r3 (same tier map, random rotation seed 3) vs B8c, final", cmp("B8r3", "B8c", "final")),
        ("POST-HOC: B8c W4A4 vs E1b, final", cmp("B8c", "E1b", "final", "a4", "a16")),
        ("Uniform NVFP4 GPTQ control B3' (4.5 bpw) vs E1a (EXL3 4.0 full), final", cmp("B3p", "E1a", "final")),
        ("Uniform NVFP4 Had16 B5 (4.5 bpw) vs E1a full, final", cmp("B5-final", "E1a", "final")),
    ]
    for label, c in rows:
        out.append(f"- {label}: {fmt(c)}")
    out.append("")

    out.append("## Allocation histograms (units = expert x projection, 18,432 total)\n")
    for name, label in (("alloc-E1b-had16", "B8: five tiers, uncalibrated full-expert residual, E1b bytes (pre-registered)"),
                        ("alloc-E1a-had16", "B8a: five tiers, E1a bytes (pre-registered stress test)"),
                        ("alloc-E1b-had16-noT0-nofreq", "B8c: {NVFP4, MXFP6, BF16}, E1b bytes (post-hoc)"),
                        ("alloc-E1b-had16-noT0-freq", "B8b: same ladder, route-frequency-weighted values (post-hoc)")):
        s = alloc_summary(name)
        if s is None:
            continue
        h = {k: v["units"] for k, v in s["histogram"].items() if v["units"]}
        pp = {p: {t: n for t, n in v.items() if n} for p, v in s["per_projection"].items()}
        out.append(f"- {label}: used {s['used_bpw']:.3f} bpw of {s['budget_bpw']:.3f}; units per tier {h}; per projection {pp}")
    out.append("")

    out.append("## Rotation learning (Cayley, 80 steps, fit-role Hessians; objective = Hessian-weighted RTN error)\n")
    for init in ("identity", "had16"):
        p = ARMS / f"learnedR-{init}" / "receipt-0-47.json"
        if p.exists():
            s = json.load(open(p))["summary"]
            gi = np.mean([v["in"]["gain_vs_identity"] for v in s.values()]); gm = np.mean([v["mid"]["gain_vs_identity"] for v in s.values()])
            out.append(f"- init {init}: mean objective gain vs identity — gate/up {gi:+.2%}, down {gm:+.2%} (negative = worse than identity on the proxy objective)")
    out.append("")

    ladder = subprocess.run([sys.executable, str(Path(__file__).parent / "report.py"), "--out", str(ROOT / "REPORT-ladder.md"), "--panels", "selection,final,wikitext"], capture_output=True, text=True).stdout
    out.append(ladder)
    out.append("\n## Decisions and deviations\n")
    out.append((ROOT / "DECISIONS.md").read_text())
    out.append("\n" + (ROOT / "REPORT_NOTES.md").read_text())
    (ROOT / "REPORT.md").write_text("\n".join(out))
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
