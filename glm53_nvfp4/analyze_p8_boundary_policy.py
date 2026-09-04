"""Freeze an identity/H128 expert mask, then validate it on disjoint fit routes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def _gmean(values: np.ndarray) -> float:
    return float(math.exp(np.log(values).mean()))


def _load(plan_sha: str, phase: str, paths: list[Path]) -> tuple[dict[int, dict], list[dict]]:
    rows, inputs = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["plan_sha256"] != plan_sha or payload["phase"] != phase:
            raise RuntimeError(f"invalid policy input: {path}")
        if payload["protected_roles_opened"] or payload["ldlq_used"]:
            raise RuntimeError(f"role or algorithm boundary violated: {path}")
        rows.extend(payload["rows"])
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    by_expert = {int(row["expert"]): row for row in rows}
    if len(by_expert) != 288 or len(rows) != 288:
        raise RuntimeError("policy inputs must cover each expert exactly once")
    return by_expert, inputs


def _packed_mask(selected: set[int]) -> str:
    data = bytearray(36)
    for expert in selected:
        data[expert // 8] |= 1 << (expert % 8)
    return data.hex()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("selection", "validation"), required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    plan_sha = sha256_file(args.plan)
    rows, inputs = _load(plan_sha, args.phase, args.input)
    experts = plan["experts"]
    h128 = np.asarray([rows[e]["h128_output_nmse"] for e in experts])
    identity = np.asarray([rows[e]["identity_output_nmse"] for e in experts])
    gptq = np.asarray([rows[e]["gptq_output_nmse"] for e in experts])

    if args.phase == "selection":
        eligible = set(plan["h128_eligible_experts"])
        selected = {
            e for e, h, i in zip(experts, h128, identity, strict=True)
            if e in eligible and h <= 0.98 * i
        }
        chosen = np.asarray([h128[e] if e in selected else identity[e] for e in experts])
        payload = {
            "schema": "glm53-p8-identity-h128-boundary-policy.v1",
            "plan_sha256": plan_sha,
            "phase": "selection",
            "inputs": inputs,
            "policy_rule": plan["policy_rule"],
            "h128_experts": sorted(selected),
            "identity_experts": sorted(set(experts) - selected),
            "h128_count": len(selected),
            "identity_count": len(experts) - len(selected),
            "packed_mask_hex": _packed_mask(selected),
            "policy_bytes": 39,
            "selection_geometric_mean_nmse": {
                "hybrid": _gmean(chosen), "h128": _gmean(h128),
                "identity": _gmean(identity), "gptq": _gmean(gptq),
            },
            "protected_roles_opened": [],
            "confirmation_logits_opened": False,
            "ldlq_used": False,
        }
    else:
        if args.policy is None:
            raise RuntimeError("validation requires --policy")
        policy = json.loads(args.policy.read_text())
        if policy["plan_sha256"] != plan_sha or policy["phase"] != "selection":
            raise RuntimeError("policy does not match the frozen plan")
        selected = set(policy["h128_experts"])
        chosen = np.asarray([h128[e] if e in selected else identity[e] for e in experts])
        log_ratio = np.log(chosen / gptq)
        spec = plan["bootstrap"]
        rng = np.random.default_rng(spec["seed"])
        indexes = rng.integers(0, len(experts), size=(spec["replicates"], len(experts)))
        interval = bca_mean_interval(log_ratio, log_ratio[indexes].mean(1))
        improvement = 1.0 - _gmean(chosen) / _gmean(gptq)
        wins = int((chosen < gptq).sum())
        passed = improvement >= 0.10 and wins >= 173 and interval[1] < 0
        payload = {
            "schema": "glm53-p8-identity-h128-boundary-policy-validation.v1",
            "plan_sha256": plan_sha,
            "policy_sha256": sha256_file(args.policy),
            "phase": "validation",
            "inputs": inputs,
            "experts": 288,
            "h128_count": len(selected),
            "geometric_mean_nmse": {
                "hybrid": _gmean(chosen), "h128": _gmean(h128),
                "identity": _gmean(identity), "gptq": _gmean(gptq),
            },
            "hybrid_vs_gptq": {
                "relative_improvement": improvement,
                "wins": wins,
                "mean_log_ratio_ci95_bca": [float(interval[0]), float(interval[1])],
            },
            "validation_rule": plan["validation_rule"],
            "decision": "pass-adaptive-kld" if passed else "fail-boundary-policy",
            "protected_roles_opened": [],
            "confirmation_logits_opened": False,
            "ldlq_used": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
