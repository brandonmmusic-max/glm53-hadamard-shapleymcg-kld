"""Freeze learned positive-diagonal H128 P8 selection and validation."""
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
    parser.add_argument("--signed-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    signed = json.loads(args.signed_result.read_text())
    if signed["decision"] != "fail-do-not-advance-signed-h128":
        raise ValueError("scaled-H128 follow-up requires the frozen signed-H128 failure")
    repo = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "glm53-p8-scaled-h128-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-developmental-with-disjoint-validation",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "selection": {"role": "fit", "strategy": "domain-balanced", "offset": 0, "count": 32},
            "validation": {"role": "fit", "strategy": "domain-balanced", "offset": 32, "count": 32},
        },
        "candidate_family": {
            "basis": "normalized Sylvester H128 independently per intermediate block",
            "statistics": "per-expert post-SwiGLU activation RMS and down-projection column RMS on selection routes",
            "laws": ["balance", "whiten", "sensitivity", "smoothquant"],
            "alpha_grid": [0.25, 0.5, 0.75, 1.0],
            "control": "unit diagonal fixed H128",
            "normalization": "zero mean log diagonal independently per 128-block",
            "clamp": [0.25, 4.0],
            "selection": "minimum routed expert-output SSE on selection routes; one chosen law and alpha per expert",
            "stored_runtime_metadata": "one FP16 positive diagonal per intermediate coordinate; no law LUT",
            "ldlq": False,
        },
        "estimand": "paired per-expert log routed-output NMSE ratio on disjoint validation, learned scaled-H128 E4M3 versus ordinary E4M3",
        "decision_rule": "advance scaled H128 to the MCG Viterbi weight-codec screen only if validation geometric-mean routed-output NMSE improves by at least 10 percent, at least 12 of 16 experts improve, the paired 95 percent BCa upper bound on mean log ratio is below zero, and maximum unquantized closure NMSE is at most 1e-10",
        "bootstrap": {"method": "BCa", "unit": "expert", "replicates": 50000, "seed": 2026090504},
        "interpretation_boundary": "a pass repairs only the P8 activation carrier and permits a separately frozen quantized-weight MCG Viterbi screen; it is not end-to-end KLD or device closure",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "signed_h128_result": {"path": str(args.signed_result), "sha256": sha256_file(args.signed_result)},
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_root / "capture-manifest.json"), "sha256": sha256_file(args.capture_root / "capture-manifest.json")},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
        "analysis_lock": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "screen_code_sha256": sha256_file(repo / "glm53_nvfp4/screen_p8_scaled_h128.py"),
            "analysis_code_sha256": sha256_file(repo / "glm53_nvfp4/analyze_p8_scaled_h128.py"),
            "transform_code_sha256": sha256_file(repo / "glm53_nvfp4/p8_h128.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
