"""Freeze the causal layer-22 route-divergence diagnostic."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--kld-analysis", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path)
    parser.add_argument("--prior-failure-log", type=Path)
    parser.add_argument("--amendment-reason")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("requires exact opened conditional-fit n=32")
    if any(roles["roles"][key] for key in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("diagnostic role file must expose only conditional-fit")
    kld = json.loads(args.kld_analysis.read_text())
    if kld.get("decision") != "fail-pseudoquant-kld" or kld.get("confirmation_logits_opened"):
        raise RuntimeError("expected the failed developmental layer-22 KLD gate")
    if (args.prior_plan is None) != (args.prior_failure_log is None):
        raise RuntimeError("prior plan and failure log must be supplied together")
    if (args.prior_plan is None) != (args.amendment_reason is None):
        raise RuntimeError("amendment reason is required exactly with a prior plan")
    payload = {
        "schema": "glm53-layer22-route-divergence-plan.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "opened conditional-fit diagnostic",
        "windows": 32,
        "candidate_layer": 22,
        "run_order": ["matched-gptq-control", "p8-layer22-candidate"],
        "execution": "TP4/EP4/DCP1 eager Humming with vLLM logical routed-expert capture; DCP1 is required because route return rejects context parallelism",
        "validity_rule": "logical top-8 sets must be bitwise equal at every routed layer 3 through 22; otherwise cross-launch nondeterminism invalidates causal attribution",
        "causal_rule": "any first divergence must occur at layer 23 or later because layer-22 routing precedes its changed expert output",
        "materiality_rule": "router-instability is a supported redesign target only if mean post-layer-22 top-8 set-change rate is at least 0.5 percent and Spearman correlation between per-window post-layer-22 route divergence and the already measured paired KLD delta is at least 0.30",
        "stopping_rule": "capture each arm once on all 32 windows; no exclusions or rerolls",
        "protected_boundary": "no teacher logits are read by this diagnostic; selection, confirmation, and final remain closed",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "inputs": {
            "roles": _file(args.roles),
            "candidate_receipt": _file(args.candidate / "BF16_LAYER_RECEIPT.json"),
            "baseline_receipt": _file(args.baseline / "BF16_LAYER_RECEIPT.json"),
            "kld_analysis": _file(args.kld_analysis),
            "runtime_manifest": _file(args.runtime_manifest),
            "runner": _file(args.runner),
            "route_evaluator": _file(Path(__file__).with_name("route_eval.py")),
            "route_analyzer": _file(
                Path(__file__).with_name("analyze_route_divergence.py")
            ),
            "plan_preparer": _file(Path(__file__)),
        },
    }
    if args.prior_plan is not None:
        payload["amendment"] = {
            "reason": args.amendment_reason,
            "prior_plan": _file(args.prior_plan),
            "prior_failure_log": _file(args.prior_failure_log),
            "prior_attempt_scored_windows": 0,
            "decision_rules_changed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
