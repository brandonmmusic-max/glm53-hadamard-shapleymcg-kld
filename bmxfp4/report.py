"""Assemble the baseline ladder: per-arm KLD on each panel, paired bootstrap CIs vs named comparators.

usage: report.py --out /media/.../bmxfp4/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from campaign import ARMS, ROOT, write_json
from kld import paired_bootstrap

COMPARATOR = {"B3": "B3p", "B4s1": "B3p", "B4s2": "B3p", "B4s3": "B3p", "B4s4": "B3p", "B4s5": "B3p", "B5": "B3p", "B6i": "B3p", "B6h": "B3p", "B5p": "B5",
              "B7": "E1b", "B7a": "E1a", "B7u": "E1b", "B7ua": "E1a", "B7n": "E1b", "B7un": "E1b", "B5L2": "B5", "B5-prov2": "B3p", "B7t": "B7", "B5L": "B5", "B5-prov": "B3p", "B8": "E1b", "B8a": "E1a", "B8b": "E1b", "B8c": "E1b", "B5-final": "B3p",
              "E1b-experts": "E1b", "E1a-experts": "E1a"}


def load_arm(name: str, panel: str, mode: str):
    p = ARMS / name / f"kld-{panel}-{mode}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "REPORT.md")
    ap.add_argument("--panels", default="selection,final,wikitext,confirmation")
    a = ap.parse_args()
    arms = sorted([d.name for d in ARMS.iterdir() if d.is_dir() and any(d.glob("kld-*.json"))])
    lines = ["# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — baseline ladder (mean per-window KLD, lower is better)", ""]
    table = {}
    for panel in a.panels.split(","):
        rows = []
        for arm in arms:
            for mode in ("a16", "a4"):
                r = load_arm(arm, panel, mode)
                if r is None:
                    continue
                comp = COMPARATOR.get(arm)
                ci = ""
                if comp:
                    rc = load_arm(comp, panel, "a16")
                    if rc is not None and len(rc["per_window_mean_kld"]) == len(r["per_window_mean_kld"]):
                        b = paired_bootstrap(r["per_window_mean_kld"], rc["per_window_mean_kld"])
                        ci = f"{b['diff_mean']:+.5f} [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]{' *' if b['excludes_zero'] else ''}"
                rows.append((arm, mode, r["mean_kld"], r["top1_agreement"], r["p99_token_kld"], r["windows"], comp or "", ci))
                table.setdefault(panel, {})[f"{arm}/{mode}"] = {"mean_kld": r["mean_kld"], "top1": r["top1_agreement"], "p99": r["p99_token_kld"], "windows": r["windows"], "vs": comp, "diff_ci": ci}
        if not rows:
            continue
        lines += [f"## Panel: {panel}", "", "| arm | act | mean KLD | top-1 | p99 token KLD | windows | vs | diff [95% CI] |", "|---|---|---:|---:|---:|---:|---|---|"]
        for arm, mode, k, t1, p99, nw, comp, ci in rows:
            lines.append(f"| {arm} | {'W4A4' if mode=='a4' else 'W4A16'} | {k:.5f} | {t1:.4f} | {p99:.3f} | {nw} | {comp} | {ci} |")
        lines.append("")
    a.out.write_text("\n".join(lines) + "\n")
    write_json(a.out.with_suffix(".json"), table)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
