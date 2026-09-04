#!/usr/bin/env python3
"""Build the public selective-H16 table and figure from opened evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/opened/exact-v10-selective-h16"
SUMMARY = ROOT / "results/selective-h16-summary.json"
FIGURE = ROOT / "figures/selective-h16-generalization.png"


def row(path: Path) -> dict:
    value = json.loads(path.read_text())
    return {
        "candidate": path.parent.name.removeprefix("layers-"),
        "role": value["role"],
        "decision": value["decision"],
        "stock_mean_kld": value["stock_mean_kld"],
        "candidate_mean_kld": value["candidate_mean_kld"],
        "mean_delta_kld": value["mean_delta_kld"],
        "relative_improvement": value["relative_improvement"],
        "delta_ci95_bca": value["delta_ci95_bca"],
        "windows": value["windows"],
        "evidence": str(path.relative_to(ROOT)),
    }


conditional = sorted(EVIDENCE.glob("layers-*/conditional-fit-vs-stock.json"))
selection = sorted(EVIDENCE.glob("layers-*/selection-wave*-vs-stock.json"))
payload = {
    "schema": "glm53-nvfp4.public-selective-h16-summary.v1",
    "metric": "equal-window FP64 KL(teacher || student); lower is better",
    "candidate_scope": "selective routed-expert layers, exact-Qwen H16 recipe, 4.5000076294 bpw",
    "conditional_fit": [row(path) for path in conditional],
    "protected_selection": [row(path) for path in selection],
    "claim_boundary": (
        "Conditional-fit results were used adaptively and are not claims. Both "
        "pre-frozen candidates failed their untouched selection waves."
    ),
}
SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

selected = {
    "003-019-020": next(x for x in payload["conditional_fit"] if x["candidate"] == "003-019-020"),
    "019-020": next(x for x in payload["conditional_fit"] if x["candidate"] == "019-020"),
}
protected = {x["candidate"]: x for x in payload["protected_selection"]}

labels = ["3,19,20\nconditional", "3,19,20\nselection", "19,20\nconditional", "19,20\nselection"]
rows = [selected["003-019-020"], protected["003-019-020"], selected["019-020"], protected["019-020"]]
values = [100 * x["relative_improvement"] for x in rows]
colors = ["#7aa6c2", "#d97757", "#7aa6c2", "#d97757"]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.suptitle("Selective H16 gains did not generalize to protected selection", y=0.98)
bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.7)
ax.axhline(0, color="#222222", linewidth=0.9)
ax.set_ylabel("KLD reduction vs stock (%)")
ax.set_ylim(min(values) - 0.35, max(values) + 0.75)
ax.grid(axis="y", alpha=0.2)
for bar, value in zip(bars, values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (0.14 if value >= 0 else -0.23),
        f"{value:+.2f}%",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=9,
    )
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIGURE, dpi=180)
print(f"wrote {SUMMARY.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
