"""Freeze a no-LDLQ equal-rate scalar16/MCG hybrid screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .prepare_learned_t12_fit import FIT_EXPERTS, VALIDATION_EXPERTS
from .shard_index import sha256_file


def _file(path: Path) -> dict[str, str | int]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-p8-scalar16-mcg-hybrid-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-only expert-disjoint P8 encoder validation before teacher KLD",
        "layer": 3,
        "law_fit_experts": FIT_EXPERTS,
        "experts": VALIDATION_EXPERTS,
        "sampling": {
            "strategy": "domain-balanced routed REAP samples per expert",
            "hessian_and_policy_fit": {"offset": 0, "count": 256},
            "evaluation": {"offset": 256, "count": 128},
        },
        "candidate": {
            "product": "P8 quality",
            "symbol_bits": 4,
            "block_size": 32,
            "scale": "UE8M0",
            "arms": {
                "mcg": "K4 Viterbi stream decoded procedurally to E4M3",
                "scalar16": "independent K4 indices decoded through three learned 16-byte E4M3 projection tables",
            },
            "hybrid_policy": "for each expert choose the best of four gate/up arm pairs on fit middle-output NMSE, then choose MCG or scalar16 down on fit full-output NMSE",
            "policy_storage": "three bits per expert (108 bytes for 288 experts) plus 48 bytes of projection tables",
            "physical_bpw": 4.25,
            "charged_bpw": 4.5,
            "rate_accounting": "reserve the spare 0.25 bpw so the comparison is at the same charged stored budget as GPTQ NVFP4",
            "encoder": "Viterbi or scalar nearest-code choice with GPTQ-style full-Hessian inter-group error feedback, static within-group activation order, and alternating UE8M0 scale refit",
            "activation_carrier": "E4M3 with UE8M0 per 32 elements",
            "runtime_table_bytes": 48,
        },
        "control": {"name": "full-Hessian GPTQ NVFP4", "physical_bpw": 4.5, "charged_bpw": 4.5},
        "primary_estimand": "paired per-expert log ratio of held-out causal full-expert routed-output NMSE, hybrid divided by GPTQ NVFP4",
        "secondary_estimands": ["hybrid versus procedural MCG", "scalar16 versus MCG", "per-projection weight NMSE"],
        "decision_rule": "advance to one unopened n=32 teacher-KLD role only if the frozen hybrid improves held-out geometric-mean full-expert NMSE at least 10 percent versus GPTQ NVFP4 and at least 5 percent versus MCG, wins at least 12 of 16 experts versus both, and both paired BCa 95 percent upper mean-log-ratio bounds are below zero",
        "bootstrap": {"unit": "expert", "method": "paired BCa", "replicates": 50000, "seed": 2026090531},
        "isa_cost": "P8 mxf8f6f4 issues twice the MMA instructions of NVFP4; this is a quality product, not an NVFP4-speed claim",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "inputs": {
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_manifest),
            "roles": _file(args.roles),
            "codebooks": _file(args.codebooks),
            "fit_receipt": _file(args.fit_receipt),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
