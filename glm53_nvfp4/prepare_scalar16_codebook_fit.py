"""Freeze the SQG-initialized scalar16 checkpoint-family law fit."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_learned_t12_fit import FIT_EXPERTS, VALIDATION_EXPERTS
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-scalar16-codebook-fit-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "BF16-weight-only checkpoint-family law fit; no routed evaluation or teacher logits",
        "layer": 3,
        "fit_experts": FIT_EXPERTS,
        "reserved_validation_experts": VALIDATION_EXPERTS,
        "projections": ["gate_proj", "up_proj", "down_proj"],
        "initialization": "16 SQG-normal quantile centers at alpha 1.5, exactly projected to E4M3",
        "fit": "two Lloyd centroid updates after UE8M0/K32 scale fitting, independently per projection, followed by ordered distinct E4M3 projection",
        "runtime_budget": "three 16-byte tables = 48 bytes; no other learned table",
        "downstream_rule": "the learned tables are frozen before expert-disjoint routed validation and may advance only under that separately preregistered gate",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "source_index": {"path": str(args.source_index), "bytes": args.source_index.stat().st_size, "sha256": sha256_file(args.source_index)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
