"""Freeze the family-law fit before learning the compact T12 table."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


FIT_EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]
VALIDATION_EXPERTS = [72, 77, 82, 87, 92, 97, 102, 107, 112, 117, 122, 127, 132, 137, 142, 143]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-kld", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prior = json.loads(args.prior_kld.read_text())
    if prior["decision"] != "fail-pseudoquant-kld":
        raise RuntimeError("learned-law redesign requires the fixed MCG KLD failure")
    payload = {
        "schema": "glm53-learned-t12-family-law-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering redesign after MCG KLD null",
        "layer": 3,
        "fit_experts": FIT_EXPERTS,
        "validation_experts": VALIDATION_EXPERTS,
        "fit_tensors": "gate, up, and down BF16 weights for the 16 fit experts",
        "law": {
            "state_mixer": "SQG XOR rank",
            "runtime_table": "4096 monotone E4M3 bytes indexed by rank>>4",
            "initialization": "frozen SQG Gaussian Chebyshev T12 staircase",
            "update": "Viterbi assignment followed by prior-regularized weighted isotonic M-step and exact E4M3 projection",
            "iterations": 2,
            "prior_strength": 8.0,
        },
        "encoder_held_constant": "K4 P8 Viterbi plus GPTQ-style full-Hessian inter-group feedback, static within-group act order, joint code/UE8M0 scale refit",
        "validation_rule": "on the 16 disjoint experts, causal full-expert NMSE must improve at least 15 percent versus procedural MCG and 20 percent versus GPTQ NVFP4, BCa upper mean-log-ratio below zero for both, and at least 12 expert wins versus MCG before another full-layer KLD build",
        "runtime_boundary": "the only table is exactly 4 KiB; the 65536-state expansion is encoder/reference scratch and is not stored in the runtime codec",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
        "prior_kld": {"path": str(args.prior_kld), "sha256": sha256_file(args.prior_kld)},
        "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
