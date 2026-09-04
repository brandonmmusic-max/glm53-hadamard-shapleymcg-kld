from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _percent_worse(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def build_summary(root: Path) -> dict[str, Any]:
    full = root / "layer3-full"
    sealed = root.parent / "sealed" / "k4-e2m3-b32-sqg-e0-selection-v1"
    kld_root = root.parents[1] / "kld-v3"

    paths = {
        "fit_screen": root / "k4-e2m3-b32-sqg-fit512-balanced-frozen-v2.json",
        "cross_fit_screen": root / "k4-e2m3-b32-sqg-fitcv512-balanced-v1.json",
        "expert0_selection": sealed / "selection-result.json",
        "expert0_record": sealed / "experiment-record.json",
        "trellis_vs_stock": full / "evidence" / "paired-kld-v3-wave1-vs-current-stock.json",
        "trellis_vs_scalar": full / "evidence" / "paired-kld-v3-wave1-vs-scalar-mxfp6.json",
        "bf16diag_vs_stock": full / "evidence" / "paired-kld-v3-wave1-bf16diag-vs-current-stock.json",
        "bridge_canary": full / "evidence" / "mxfp6-bridge-canary-e0.json",
        "mixed_receipt": full / "candidate-mxfp6-bridge" / "MIXED_RECEIPT.json",
        "bf16_receipt": full / "candidate-bf16-diagnostic" / "BF16_LAYER_RECEIPT.json",
        "scalar_receipt": root / "layer3-scalar-mxfp6" / "candidate" / "MIXED_RECEIPT.json",
        "stock_run": kld_root / "run-alphabet-v1-stock-current-v3-wave1.json",
        "trellis_run": kld_root / "run-alphabet-v1-k4-e2m3-sqg-l3-v3-wave1.json",
        "scalar_run": kld_root / "run-alphabet-v1-scalar-e2m3-l3-v3-wave1.json",
        "bf16diag_run": kld_root / "run-alphabet-v1-k4-e2m3-sqg-l3-bf16diag-r2-v3-wave1.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required evidence:\n" + "\n".join(missing))

    data = {name: _read(path) for name, path in paths.items()}
    stock = float(data["trellis_vs_stock"]["stock_mean_kld"])
    trellis = float(data["trellis_vs_stock"]["candidate_mean_kld"])
    scalar = float(data["trellis_vs_scalar"]["stock_mean_kld"])
    bf16diag = float(data["bf16diag_vs_stock"]["candidate_mean_kld"])

    return {
        "schema": "glm53-trellis-mxf.h-alphabet-results.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "A K3/K4 trellis decoded to an E2M3/E3M2/E4M3 alphabet can retain TCQ quality "
            "while feeding a single native mxf8f6f4-family matmul."
        ),
        "tested_scope": {
            "model": "GLM-5.3-Flash",
            "layer": 3,
            "experts": 288,
            "winner": "K4, E2M3, UE8M0/32, SQG XOR-Chebyshev t12, alpha=1.75",
            "conceptual_codec_bpw": 4.25,
            "actual_mxfp6_bridge_bpw_for_layer": 6.250007629394531,
            "evaluation_role": "reused V3 selection/development role; not a fresh protected holdout",
            "windows": int(data["trellis_vs_stock"]["windows"]),
        },
        "development_screen": {
            "fit_reduction_vs_matched_nvfp4_percent": data["fit_screen"]["aggregate"]["matched-nvfp4"]["trellis_reduction_percent"],
            "cross_fit_reduction_vs_matched_nvfp4_percent": data["cross_fit_screen"]["aggregate"]["matched-nvfp4"]["trellis_reduction_percent"],
            "expert0_reused_selection_reduction_percent": data["expert0_selection"]["primary_effect_percent"],
            "warning": "Expert 0 and the selection role were part of adaptive development; these are not confirmatory claims.",
        },
        "end_to_end_kld": {
            "stock_nvfp4_current": stock,
            "k4_e2m3_trellis_bf16_weight_diagnostic": {
                "mean": bf16diag,
                "percent_worse_than_stock": _percent_worse(bf16diag, stock),
                "delta_ci95_bca": data["bf16diag_vs_stock"]["delta_ci95_bca"],
            },
            "scalar_e2m3_native_mxfp6": {
                "mean": scalar,
                "percent_worse_than_stock": _percent_worse(scalar, stock),
            },
            "k4_e2m3_trellis_native_mxfp6_bridge": {
                "mean": trellis,
                "percent_worse_than_stock": _percent_worse(trellis, stock),
                "delta_ci95_bca_vs_stock": data["trellis_vs_stock"]["delta_ci95_bca"],
                "percent_better_than_scalar_mxfp6": 100.0 * data["trellis_vs_scalar"]["relative_improvement"],
                "delta_ci95_bca_vs_scalar": data["trellis_vs_scalar"]["delta_ci95_bca"],
            },
        },
        "bridge_correctness": data["bridge_canary"],
        "decision": {
            "quality_gate": "FAIL",
            "fused_decoder": "DEFER",
            "reason": (
                "The conceptual 4.25-bpw trellis candidate did not beat stock NVFP4 end to end. "
                "Its nominal 0.349 percent KLD advantage over the same-runtime scalar E2M3 control "
                "is not statistically resolved, and the BF16 weight-only diagnostic is also worse than stock."
            ),
            "bounded_positive": (
                "The trellis approximately retained scalar E2M3 end-to-end quality while conceptually storing "
                "two fewer weight bits, but no direct 4.25-bpw decoder or native-speed result exists."
            ),
        },
        "limitations": [
            "The V3 selection windows were previously opened and are development evidence, not protected confirmation.",
            "The full-layer test changes layer 3 only; it is not a fully quantized whole-model codec artifact.",
            "The executable bridge stores standard MXFP6 at 6.25 bpw for layer 3; 4.25 bpw is the unimplemented direct-codec target.",
            "No fused codec decode PTX/SASS or performance profile has been produced.",
            "The pinned V4 teacher-logit repository revision was unavailable, so fresh V4 scoring could not be completed.",
        ],
        "sources": {name: _source(path) for name, path in paths.items()},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    kld = summary["end_to_end_kld"]
    bf16 = kld["k4_e2m3_trellis_bf16_weight_diagnostic"]
    scalar = kld["scalar_e2m3_native_mxfp6"]
    trellis = kld["k4_e2m3_trellis_native_mxfp6_bridge"]
    screen = summary["development_screen"]
    lines = [
        "# H-ALPHABET result",
        "",
        "Decision: **FAIL the end-to-end quality gate; defer the fused decoder.**",
        "",
        "| Candidate | Mean KLD | Relative to current stock |",
        "| --- | ---: | ---: |",
        f"| Current stock NVFP4 | {kld['stock_nvfp4_current']:.10f} | baseline |",
        f"| K4 E2M3 trellis, BF16 weight diagnostic | {bf16['mean']:.10f} | {bf16['percent_worse_than_stock']:.3f}% worse |",
        f"| Scalar E2M3, native MXFP6 | {scalar['mean']:.10f} | {scalar['percent_worse_than_stock']:.3f}% worse |",
        f"| K4 E2M3 trellis, native MXFP6 bridge | {trellis['mean']:.10f} | {trellis['percent_worse_than_stock']:.3f}% worse |",
        "",
        f"The trellis bridge was nominally {trellis['percent_better_than_scalar_mxfp6']:.3f}% better than the same-runtime scalar E2M3 control, but its 95% BCa delta interval {trellis['delta_ci95_bca_vs_scalar']} crosses zero.",
        "",
        "## Why the early result did not hold",
        "",
        f"The development screen reported {screen['fit_reduction_vs_matched_nvfp4_percent']:.3f}% on fit, {screen['cross_fit_reduction_vs_matched_nvfp4_percent']:.3f}% on cross-fit, and {screen['expert0_reused_selection_reduction_percent']:.3f}% for expert 0 on a reused selection role. The 288-expert full-layer test did not reproduce that advantage.",
        "",
        "## What is established",
        "",
        summary["decision"]["bounded_positive"],
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(["", "The adjacent JSON record contains SHA-256 receipts for every source artifact used here.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.root.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "H-ALPHABET-RESULTS.json"
    md_path = args.out_dir / "H-ALPHABET-RESULTS.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
