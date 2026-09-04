"""Freeze the disjoint expert validation of the learned 4 KiB law."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_learned_t12_fit import VALIDATION_EXPERTS
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-plan", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-learned-t12-16expert-validation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit engineering; expert-disjoint family-law validation",
        "layer": 3,
        "experts": VALIDATION_EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 256, "count": 128},
        },
        "candidate": {
            "variant": "learned-t12-k4",
            "product": "P8",
            "bits": 4,
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0",
            "runtime_table_bytes": 4096,
            "encoder": "Viterbi plus GPTQ-style full-Hessian inter-group feedback, static within-group act order, iterative scale refit",
            "physical_bpw": 4.25,
            "charged_bpw": 4.5,
        },
        "controls": ["procedural MCG K4 P8", "full-Hessian GPTQ NVFP4"],
        "decision_rule": "pass only if causal full-expert geometric-mean NMSE improves at least 15 percent versus MCG and 20 percent versus GPTQ NVFP4, paired expert BCa upper mean-log-ratio below zero for both, and at least 12 of 16 experts beat MCG",
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 20000, "seed": 20260904},
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ code path, objective, or result is used",
        "fit_plan": {"path": str(args.fit_plan), "sha256": sha256_file(args.fit_plan)},
        "fit_receipt": {"path": str(args.fit_receipt), "sha256": sha256_file(args.fit_receipt)},
        "codebook": {"path": str(args.codebook), "sha256": sha256_file(args.codebook)},
        "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
