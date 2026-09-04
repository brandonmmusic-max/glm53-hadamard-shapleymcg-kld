"""Freeze powered replication of the two surviving scaled-H128 family laws."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prior = json.loads(args.screen_result.read_text())
    if not (prior["scaled_h128_relative_improvement"] >= 0.10 and prior["decision"] == "fail-do-not-advance-scaled-h128"):
        raise ValueError("replication requires the promising but unqualified scaled-H128 screen")
    repo = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "glm53-p8-scaled-h128-replication-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-developmental-new-disjoint-slices",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "calibration": {"role": "fit", "strategy": "domain-balanced", "offset": 64, "count": 128},
            "evaluation": {"role": "fit", "strategy": "domain-balanced", "offset": 192, "count": 128},
        },
        "arms": {
            "balance-0.5": "d=exp(0.5*(log down-column RMS - log post-SwiGLU RMS)), centered per H128 block",
            "whiten-0.5": "d=exp(-0.5*log post-SwiGLU RMS), centered per H128 block",
        },
        "common": {"basis": "normalized Sylvester H128", "diagonal_clamp": [0.25, 4.0], "ldlq": False},
        "estimand": "paired per-expert log routed-output NMSE ratio for each fixed family law versus ordinary E4M3 on the new evaluation slice",
        "multiplicity": "two-family Bonferroni; each comparison uses a 97.5 percent BCa interval",
        "decision_rule": "a law advances to an MCG Viterbi quantized-weight screen only if geometric-mean routed-output NMSE improves at least 10 percent, at least 12 of 16 experts improve, its Bonferroni 97.5 percent BCa upper bound is below zero, and maximum unquantized closure NMSE is at most 1e-10; if both pass choose the larger improvement",
        "bootstrap": {"method": "BCa", "unit": "expert", "replicates": 50000, "seed": 2026090505, "alpha": 0.025},
        "interpretation_boundary": "this is a powered activation-carrier replication, not weight-codec evidence, end-to-end KLD, or device closure",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "prior_screen": {"path": str(args.screen_result), "sha256": sha256_file(args.screen_result)},
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_root / "capture-manifest.json"), "sha256": sha256_file(args.capture_root / "capture-manifest.json")},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
        "analysis_lock": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "screen_code_sha256": sha256_file(repo / "glm53_nvfp4/screen_p8_scaled_h128_replication.py"),
            "analysis_code_sha256": sha256_file(repo / "glm53_nvfp4/analyze_p8_scaled_h128_replication.py"),
            "transform_code_sha256": sha256_file(repo / "glm53_nvfp4/p8_h128.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
