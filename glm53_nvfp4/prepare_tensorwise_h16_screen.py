"""Freeze the first genuinely per-tensor in-block H16 calibration screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 10, 20, 30, 40, 50, 60, 70]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-tensorwise-signed-h16-pilot-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-only engineering screen; no teacher logits",
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
        "estimand": "per-expert route-weighted causal full-MoE output NMSE on disjoint fit-role routes",
        "decision_rule": (
            "advance to the frozen 16-expert screen only if tensorwise H16 improves "
            "geometric-mean evaluation NMSE by at least 5 percent versus identity, "
            "beats fixed H16, and wins at least 6 of 8 experts versus identity"
        ),
        "protected_boundary": (
            "V4 selection and the 28 confirmation logits remain unopened; this pilot "
            "may change the candidate but may not consume teacher KLD"
        ),
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
