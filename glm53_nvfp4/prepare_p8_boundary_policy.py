"""Freeze selection and validation slices for a tiny identity/H128 P8 policy."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--identity-plan", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    attribution = json.loads(args.attribution.read_text())
    if attribution.get("decision") != "pass-build-identity-p8-fallback":
        raise RuntimeError("boundary policy was not authorized")
    payload = {
        "schema": "glm53-p8-identity-h128-boundary-policy-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": 3,
        "experts": list(range(288)),
        "h128_eligible_experts": attribution["h128_eligible_experts"],
        "sampling": {
            "selection": {"role": "fit", "strategy": "domain-balanced", "offset": 576, "count": 128},
            "validation": {"role": "fit", "strategy": "domain-balanced", "offset": 704, "count": 128},
        },
        "policy_rule": "choose H128 only for a preregistered eligible expert when its selection output NMSE is at least 2 percent below identity-MCG; otherwise choose identity-MCG",
        "validation_rule": "advance the hybrid to adaptive KLD only if validation geometric-mean output NMSE beats GPTQ NVFP4 by at least 10 percent, wins at least 173 of 288 experts, and the paired BCa 95 percent upper bound on mean log ratio is below zero",
        "storage": {
            "expert_mask_bytes": 36,
            "checkpoint_family_law_ids_bytes": 3,
            "total_policy_bytes": 39,
            "table_limit_bytes": 4096,
        },
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "kld_boundary": "the opened conditional-fit32 role is adaptive-only after the first codec result and cannot qualify this policy",
        "bootstrap": {"method": "paired BCa", "unit": "expert", "replicates": 50000, "seed": 2026090513},
        "inputs": {
            "attribution": _file(args.attribution),
            "identity_plan": _file(args.identity_plan),
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "roles": _file(args.roles),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
