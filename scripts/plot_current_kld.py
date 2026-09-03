#!/usr/bin/env python3
"""Render the current opened GLM layer-3 KLD comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/current-kld-summary.json"
OUTPUT = ROOT / "figures/current-layer3-kld.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    wanted = {
        "same-loader stock NVFP4": "Stock",
        "identity GPTQ, layer 3": "Identity GPTQ",
        "learned block rotation all projections, layer 3": "Learned",
        "fixed H16 all projections, layer 3, 10000 calibration cap": "H16 10k",
        "exact Qwen H16 all projections, layer 3, 256 samples per expert": "H16 Qwen-256",
    }
    rows = [row for row in data["candidates"] if row["name"] in wanted]
    labels = [wanted[row["name"]] for row in rows]
    values = [row["mean_kld"] for row in rows]
    colors = ["#6b7280", "#9ca3af", "#d97706", "#047857", "#10b981"]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean KLD (lower is better)")
    ax.set_title("GLM-5.3-Flash layer-3 rotation candidates\n16 opened conditional-fit windows")
    ax.set_ylim(0.0368, 0.03935)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.000035,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.018,
        "H16 10k vs stock: -2.625%; paired BCa 95% delta CI [-0.001717, -0.000177]",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
