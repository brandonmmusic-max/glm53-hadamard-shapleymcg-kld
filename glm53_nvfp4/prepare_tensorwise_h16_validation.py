"""Freeze disjoint validation for the tensorwise signed-H16 candidate family."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [5, 15, 25, 35, 45, 55, 65, 71, 80, 90, 100, 110, 120, 130, 140, 150]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    pilot = json.loads(args.pilot_analysis.read_text())
    if pilot["decision"] != "pass-scale-to-16":
        raise RuntimeError("pilot did not authorize disjoint validation")
    payload = {
        "schema": "glm53-tensorwise-signed-h16-validation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-only disjoint expert validation; no teacher logits",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "rotation_fit": {"offset": 0, "count": 128},
            "evaluation": {"offset": 128, "count": 128},
        },
        "candidate": {
            "name": "tensorwise-signed-h16",
            "transform": "independent D*H16 for every expert gate, up, and down tensor",
            "variants_per_tensor": 8,
            "variant_zero": "fixed H16",
            "randomization": "SHA-256-derived diagonal signs with first sign fixed positive",
            "selection": "lowest fit routed projection-output NMSE under block16 GPTQ",
            "materialization": "selected transforms re-encoded with full-Hessian GPTQ and exact ModelOpt NVFP4 E2M1/E4M3/16",
            "gate_up_scale": "one jointly refit FP32 global scale after independent gate/up transform selection",
            "stored_bpw": 4.5,
            "seed": 2026090401,
        },
        "controls": [
            "identity full-Hessian GPTQ NVFP4 at 4.5 bpw",
            "fixed shared H16 full-Hessian GPTQ NVFP4 at 4.5 bpw",
        ],
        "estimand": "paired per-expert log causal full-MoE output-NMSE ratio on disjoint routes",
        "bootstrap": {"unit": "expert", "replicates": 20000, "seed": 2026090402},
        "thresholds": {
            "minimum_improvement_vs_identity": 0.03,
            "minimum_wins_vs_identity": 12,
            "require_improvement_vs_fixed_h16": True,
            "require_identity_bca_upper_below_zero": True,
        },
        "decision_rule": (
            "build the full layer-3 pseudoquant candidate only if improvement versus "
            "identity is at least 3 percent, at least 12/16 experts win, the paired "
            "expert BCa upper mean-log-ratio bound is below zero, and the aggregate "
            "also beats ordinary fixed H16"
        ),
        "pass_decision": "pass-build-layer3-tensorwise-h16",
        "fail_decision": "fail-do-not-open-v4-selection",
        "protected_boundary": "V4 selection and 28 confirmation logits remain unopened",
        "prior": {"path": str(args.pilot_analysis), "sha256": sha256_file(args.pilot_analysis)},
        "inputs": {
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_manifest), "sha256": sha256_file(args.capture_manifest)},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
