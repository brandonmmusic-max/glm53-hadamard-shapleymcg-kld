"""Freeze the real-input W6A8 activation-stage decomposition."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-w6a8-stage-ablation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit",
        "layer": 3,
        "experts": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71],
        "samples": "offset 32, n32 domain-balanced REAP routes per expert",
        "arms": {
            "R0": "decoded BF16 weights and BF16 activations",
            "A1": "E4M3 K32 FC1 input only, decoded BF16 weights",
            "A2": "E4M3 K32 post-SwiGLU input only, decoded BF16 weights",
            "A12": "both E4M3 activation hops, decoded BF16 weights",
            "F": "fused W6A8 kernel with the same effective MXFP6 weights and both activation hops",
        },
        "metric": "geometric mean routed expert-output NMSE across fixed experts",
        "decision_rule": {
            "activation_dominant": "A12-R0 NMSE >= F-A12 NMSE",
            "fused_carrier_dominant": "F-A12 NMSE > A12-R0 NMSE",
            "closure": "F-A12 NMSE <= 1e-4",
        },
        "scale": "unit a1 and unit a2 retained after disjoint validation rejected sqrt(2)",
        "model_semantics": "swiglu_limit 10; precise sigmoid; deterministic route reduction; static FC2 scale for exact stage matching",
        "protected_boundary": "selection, confirmation, final, and 28 confirmation logits remain unopened",
        "ldlq": False,
        "prior_scale_result": {"path": str(args.scale_result), "sha256": sha256_file(args.scale_result)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
