"""Freeze the exact pseudoquant execution inputs before opening conditional-fit."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("execution role must contain exactly 32 conditional-fit windows")
    if any(roles["roles"][name] for name in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("execution role exposes a protected/non-conditional partition")

    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-layer3-execution.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "run_order": ["gptq-nvfp4-control", "scaled-h128-mcg-candidate"],
        "run_ids": {
            "control": "p8-scaled-mcg-l3-control-cf32-v1",
            "candidate": "p8-scaled-mcg-l3-candidate-cf32-v1",
        },
        "stopping_rule": "one complete run per arm; no exclusions, substitutions, or rerolls",
        "decision_rule": "mean paired delta KLD <= -0.0014 nats and paired BCa 95 percent upper bound < 0",
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "activation_carrier": "dynamic E4M3 with UE8M0-equivalent power-of-two scale per 32 values at routed input and post-SwiGLU",
        "inputs": {
            "plan": _file(args.plan),
            "roles": _file(args.roles),
            "candidate_overlay": _file(args.candidate / "BF16_LAYER_RECEIPT.json"),
            "control_overlay": _file(args.control / "BF16_LAYER_RECEIPT.json"),
            "runtime_manifest": _file(args.runtime_manifest),
        },
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "runtime_interpretation": "BF16 decoded-weight pseudoquant linkage test only; not native P8 kernel closure",
        "isa_cost": "P8 mxf8f6f4 is twice NVFP4 MMA issue count; P8 is quality, P4 is speed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
