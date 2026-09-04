"""Apply the frozen powered-validation rule to four routed-sum partials."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--partials", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    tokens = len(plan["windows"]) * plan["rows_per_window"]
    reference = torch.zeros((tokens, 4096), dtype=torch.float32)
    candidate = torch.zeros_like(reference)
    gptq = torch.zeros_like(reference)
    inputs, expected_start = [], 0
    for path in args.partials:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("schema") != "glm53-p8-frozen-policy-partial.v1"
                or metadata.get("ldlq") != "false"
                or metadata.get("policy_sha256") != plan["policy_sha256"]
                or int(metadata.get("expert_start", -1)) != expected_start
            ):
                raise RuntimeError(f"invalid frozen-policy partial: {path}")
            reference += handle.get_tensor("reference_sum").float()
            candidate += handle.get_tensor("candidate_error").float()
            gptq += handle.get_tensor("gptq_error").float()
            expected_start = int(metadata["expert_end"])
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if expected_start != 288:
        raise RuntimeError("partial expert ranges do not cover all 288 experts")

    rows = []
    for index, item in enumerate(plan["windows"]):
        start, stop = index * plan["rows_per_window"], (index + 1) * plan["rows_per_window"]
        denominator = reference[start:stop].double().square().sum().clamp_min(1e-30)
        candidate_nmse = float(candidate[start:stop].double().square().sum() / denominator)
        gptq_nmse = float(gptq[start:stop].double().square().sum() / denominator)
        rows.append({
            "window": item["id"], "domain": item["domain"],
            "candidate_nmse": candidate_nmse, "gptq_nmse": gptq_nmse,
            "log_ratio": math.log(candidate_nmse / gptq_nmse),
        })
    candidate_values = np.asarray([row["candidate_nmse"] for row in rows])
    gptq_values = np.asarray([row["gptq_nmse"] for row in rows])
    log_ratio = np.asarray([row["log_ratio"] for row in rows])
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(rows), size=(spec["replicates"], len(rows)))
    bootstrap = log_ratio[indices].mean(axis=1)
    interval = bca_mean_interval(log_ratio, bootstrap)
    improvement = 1.0 - float(candidate_values.sum() / gptq_values.sum())
    wins = int((candidate_values < gptq_values).sum())
    passed = improvement >= 0.10 and wins >= 24 and interval[1] < 0
    domains = defaultdict(list)
    for row in rows:
        domains[row["domain"]].append(row)
    payload = {
        "schema": "glm53-p8-fc1-three-state-powered-validation-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "policy_sha256": plan["policy_sha256"],
        "windows": len(rows),
        "candidate_aggregate_nmse": float(candidate_values.sum()),
        "gptq_aggregate_nmse": float(gptq_values.sum()),
        "relative_improvement": improvement,
        "window_wins": wins,
        "mean_log_ratio": float(log_ratio.mean()),
        "mean_log_ratio_ci95_bca": [float(interval[0]), float(interval[1])],
        "mean_log_ratio_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "by_domain": {
            domain: {
                "candidate_geometric_mean_nmse": float(np.exp(np.mean(np.log([row["candidate_nmse"] for row in values])))),
                "gptq_geometric_mean_nmse": float(np.exp(np.mean(np.log([row["gptq_nmse"] for row in values])))),
                "wins": sum(row["candidate_nmse"] < row["gptq_nmse"] for row in values),
            }
            for domain, values in sorted(domains.items())
        },
        "window_results": rows,
        "decision_rule": plan["decision_rule"],
        "decision": "pass-freeze-for-unopened-kld" if passed else "fail-powered-fit-validation",
        "inputs": inputs,
        "protected_roles_opened": [],
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "decision", "windows", "relative_improvement", "window_wins",
        "mean_log_ratio", "mean_log_ratio_ci95_bca", "by_domain",
    )}, indent=2, sort_keys=True))
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
