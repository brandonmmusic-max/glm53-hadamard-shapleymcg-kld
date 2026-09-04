"""Freeze the fixed-H128 P8 activation-boundary developmental screen."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    payload = {
        "schema": "glm53-p8-h128-activation-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-developmental-disjoint-slice",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {"role": "fit", "strategy": "domain-balanced", "offset": 32, "count": 32},
        "arms": {
            "ordinary-e4m3": "original BF16 weights; E4M3 K32 quantization at FC1 input and post-SwiGLU",
            "fixed-h128-e4m3": "separately H128-transformed gate/up/down BF16 weights; exact uncoupled H128 kernel boundary; same two E4M3 K32 hops",
            "fixed-h128-unquantized": "same transformed weights and boundary with no activation quantization; closure oracle only",
        },
        "estimand": "paired per-expert log routed-output NMSE ratio, fixed-H128 E4M3 versus ordinary E4M3",
        "decision_rule": "advance fixed H128 to a quantized-weight codec screen only if geometric-mean routed-output NMSE improves by at least 10 percent, at least 12 of 16 experts improve, the paired 95 percent BCa upper bound on the mean log ratio is below zero, and maximum unquantized closure NMSE is at most 1e-10",
        "bootstrap": {"method": "BCa", "unit": "expert", "replicates": 50000, "seed": 2026090501},
        "algorithm": {
            "hadamard": "normalized Sylvester H128 independently on each intermediate 128-block",
            "projection_transforms": {
                "gate": "Wg_prime = H128 Wg",
                "up": "Wu_prime = H128 Wu",
                "down": "Wd_prime = Wd H128",
            },
            "kernel_boundary": "g=H128(Wg_prime x); u=H128(Wu_prime x); h_prime=H128(SiLU(g)*u)",
            "diagonal_scales": "identity in this first fixed-H128 screen",
            "error_feedback": "none in this activation-only diagnostic",
            "ldlq": False,
        },
        "interpretation_boundary": "a pass diagnoses the P8 activation carrier and permits a separate MCG Viterbi weight-codec screen; it is not end-to-end KLD or device closure",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_root / "capture-manifest.json"), "sha256": sha256_file(args.capture_root / "capture-manifest.json")},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
        "analysis_lock": {
            "git_head": git_head,
            "screen_code_sha256": sha256_file(repo / "glm53_nvfp4/screen_p8_h128_activation.py"),
            "analysis_code_sha256": sha256_file(repo / "glm53_nvfp4/analyze_p8_h128_activation.py"),
            "transform_code_sha256": sha256_file(repo / "glm53_nvfp4/p8_h128.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
