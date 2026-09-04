"""Freeze the adaptive KLD linkage run for the validated expert policy."""
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
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    validation = json.loads(args.validation.read_text())
    if validation.get("decision") != "pass-adaptive-kld" or validation.get("ldlq_used"):
        raise RuntimeError("boundary policy did not pass its disjoint fit validation")
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("adaptive linkage role must contain the same 32 windows")
    payload = {
        "schema": "glm53-p8-identity-h128-policy-adaptive-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "p8-boundary-policy-l3-candidate-cf32-adaptive-v1",
        "status": "adaptive-reuse-not-qualifying",
        "estimand": "paired mean KLD(hybrid)-KLD(existing decoded GPTQ NVFP4 control)",
        "decision_rule": "developmental signal only: mean delta <= -0.0014 and paired BCa 95 percent upper bound below zero",
        "stopping_rule": "one hybrid run on the exact 32 already-opened windows; no exclusions or rerolls",
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "physical_codec_bpw": 4.250443113070947,
        "runtime": "decoded BF16 weights with E4M3 K32 activation carriers and a 39-byte identity/H128 expert policy",
        "claim_boundary": "because these conditional-fit windows informed the redesign, this run cannot qualify or validate the policy even if it passes the numerical target",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "validation": _file(args.validation),
            "policy": _file(args.policy),
            "overlay": _file(args.overlay / "P8_POLICY_OVERLAY_RECEIPT.json"),
            "roles": _file(args.roles),
            "control_run": _file(args.control_run),
            "runtime_manifest": _file(args.runtime_manifest),
        },
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090514},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
