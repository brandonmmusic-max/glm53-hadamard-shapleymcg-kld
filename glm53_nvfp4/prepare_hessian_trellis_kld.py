"""Freeze the n=32 conditional-fit KLD gate for the P8 pseudoquant layer."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full-screen-analysis", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    windows = roles["roles"]["conditional-fit"]
    if len(windows) != 32 or roles["domains"] != {
        "axis1_general": 8,
        "axis2_legal": 8,
        "axis3_code_agentic": 8,
        "axis4_reasoning_termination": 8,
    }:
        raise RuntimeError("KLD gate requires the fixed domain-balanced n=32 role")
    if any(roles["roles"][role] for role in ("selection", "confirmation", "final")):
        raise RuntimeError("KLD gate role file exposes protected roles")
    candidate_receipt = args.candidate / "BF16_LAYER_RECEIPT.json"
    full_analysis = json.loads(args.full_screen_analysis.read_text())
    if full_analysis["decision"] != "pass-build-layer3-kld-candidate":
        raise RuntimeError("full-expert pseudoquant gate did not pass")
    payload = {
        "schema": "glm53-hessian-trellis-p8-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit",
        "windows": 32,
        "domain_counts": roles["domains"],
        "run_order": ["uniform-gptq-baseline", "hessian-trellis-p8-bf16-pseudoquant"],
        "candidate": {
            "model": str(args.candidate),
            "receipt_sha256": sha256_file(candidate_receipt),
            "physical_codec_bpw": 4.25,
            "charged_bpw": 4.5,
            "execution": "dense BF16 pseudoquant overlay; numerical encoder gate only",
        },
        "baseline": {
            "model": str(args.baseline),
            "index_sha256": sha256_file(args.baseline / "model.safetensors.index.json"),
            "format": "full-Hessian calibrated ModelOpt NVFP4",
            "bpw": 4.5,
        },
        "primary_estimand": "equal-window mean KLD(candidate) minus KLD(uniform GPTQ NVFP4 baseline)",
        "decision_rule": "pass only if mean delta is at most -0.0014 nats and the paired BCa 95 percent upper bound is below zero",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 20000, "seed": 20260902},
        "stopping_rule": "run each arm once on the exact 32 windows; preserve launch failures; no exclusions, substitutions, or rerolls",
        "runtime_boundary": "a pass qualifies pseudoquant encoder quality only; P8 kernel closure and speed remain blocked on the prologue",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is the quality product; P4 is the NVFP4-speed product",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
        "inputs": {
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
            "full_screen_analysis": {"path": str(args.full_screen_analysis), "sha256": sha256_file(args.full_screen_analysis)},
            "runtime_manifest": {"path": str(args.runtime_manifest), "sha256": sha256_file(args.runtime_manifest)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
