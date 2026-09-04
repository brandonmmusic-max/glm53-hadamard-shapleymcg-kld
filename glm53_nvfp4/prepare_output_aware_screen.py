"""Freeze the first real-REAP output-aware trellis screen before execution."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


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
        "schema": "glm53-output-aware-trellis-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering screen; not an acceptance or confirmation role",
        "layer": 3,
        "experts": [0],
        "projections": ["gate_proj", "up_proj"],
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "fit": {"offset": 0, "count": 128},
            "validation": {"offset": 128, "count": 128},
        },
        "carrier": "P8 E4M3 K32 UE8M0 amax pseudoquant",
        "candidate": {
            "rate": "K4 plus one UE8M0 byte per 32 weights = 4.25 bpw",
            "runtime": "P8 mxf8f6f4; quality product; twice NVFP4 MMA issue rate",
            "law": "procedural MCG alpha=2.0",
            "encoder": "Viterbi per native 16x16 tile plus full-Hessian GPTQ-style inter-group error feedback and static within-group act order",
            "scale_refit_iterations": 2,
            "fit_grid": {
                "ridge_ratio": [0.0001, 0.001, 0.01],
                "correction_scale": [0.25, 0.5, 0.75, 1.0],
            },
        },
        "controls": [
            "BF16 weight on P8 activation carrier",
            "scalar E4M3 K32",
            "plain MCG K4 Viterbi",
            "Hessian-feedback MCG K4 Viterbi without output correction",
            "standard GPTQ NVFP4 at 4.5 bpw",
        ],
        "selection_rule": "choose one ridge/correction pair by lowest aggregate unquantized fit output NMSE, then freeze it before validation and quantization",
        "advance_rule": "advance to a preregistered 16-expert projection screen only if held-out aggregate output NMSE improves at least 10 percent versus both Hessian-only trellis and GPTQ NVFP4, with neither gate nor up worse than Hessian-only trellis",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
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
