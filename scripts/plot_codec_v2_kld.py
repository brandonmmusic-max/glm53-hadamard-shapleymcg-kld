#!/usr/bin/env python3
"""Render matched-path codec-v2 KLD effects from the machine-readable summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/codec-v2-redesign-summary.json"
OUTPUT = ROOT / "figures/codec-v2-matched-kld.png"


def _effect(delta: float, baseline: float) -> float:
    return -100.0 * delta / baseline


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    layer = data["matched_bf16_layer_attribution"]
    external = data["external_v5_subset_kld"]
    layer22 = data["layer22_matched_bf16_kld"]
    rows = [
        (
            "Layer 3\nconditional-fit",
            100.0 * layer["layer3_relative_improvement"],
            layer["layer3_delta_ci98_333_bca"],
            data["matched_bf16_composite_kld"]["gptq_nvfp4_bf16_mean"],
            "adaptive pass",
        ),
        (
            "Layer 20\nconditional-fit",
            100.0 * layer["layer20_relative_improvement"],
            layer["layer20_delta_ci98_333_bca"],
            data["matched_bf16_composite_kld"]["gptq_nvfp4_bf16_mean"],
            "adaptive pass",
        ),
        (
            "Layers 3+20\nexternal V5",
            100.0 * external["relative_improvement"],
            external["delta_ci95_bca"],
            external["gptq_nvfp4_bf16_mean"],
            "gate fail",
        ),
        (
            "Layer 22\nconditional-fit",
            100.0 * layer22["relative_improvement"],
            layer22["delta_ci95_bca"],
            layer22["gptq_nvfp4_bf16_mean"],
            "gate fail",
        ),
    ]
    labels = [row[0] for row in rows]
    effects = [row[1] for row in rows]
    bounds = [
        (_effect(row[2][1], row[3]), _effect(row[2][0], row[3]))
        for row in rows
    ]
    lower = [effect - interval[0] for effect, interval in zip(effects, bounds, strict=True)]
    upper = [interval[1] - effect for effect, interval in zip(effects, bounds, strict=True)]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(
        labels,
        effects,
        color=["#047857", "#10b981", "#d97706", "#dc2626"],
        width=0.62,
    )
    ax.errorbar(
        range(len(rows)),
        effects,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.5,
        capsize=5,
    )
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_ylabel("KLD reduction vs matched decoded-GPTQ control (%)")
    ax.set_title("GLM-5.3-Flash P8 codec: matched BF16 execution path")
    ax.grid(axis="y", alpha=0.22)
    for bar, row in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row[1] + 0.18,
            f"{row[1]:.2f}%\n{row[4]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.015,
        "Error bars: paired BCa intervals. Conditional-fit is adaptive; external V5 and layer 22 fail their gates.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
