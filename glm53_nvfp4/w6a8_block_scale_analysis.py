"""Analyze native E4M3 K32 exponent-search pseudoquant evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def _values(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    return {
        int(cell["expert"]): float(cell["ablations"]["both_activation_quant_vs_exact"]["nmse"])
        for cell in payload["cells"]
    }


def _gmean(values: dict[int, float]) -> float:
    return math.exp(sum(math.log(value) for value in values.values()) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    baseline = _values(args.baseline)
    candidate = _values(args.candidate)
    if set(baseline) != set(plan["experts"]) or set(candidate) != set(baseline):
        raise RuntimeError("results do not match the frozen expert set")
    baseline_gmean = _gmean(baseline)
    candidate_gmean = _gmean(candidate)
    improvement = (baseline_gmean - candidate_gmean) / baseline_gmean
    wins = sum(candidate[expert] < baseline[expert] for expert in baseline)
    passed = improvement >= 0.15 and wins >= 12
    payload = {
        "schema": "glm53-w6a8-block-scale-result.v1",
        "baseline_gmean_nmse": baseline_gmean,
        "candidate_gmean_nmse": candidate_gmean,
        "relative_improvement": improvement,
        "expert_wins": wins,
        "experts": len(baseline),
        "decision": "pass-implement-kernel" if passed else "null-do-not-implement",
        "plan_sha256": sha256_file(args.plan),
        "baseline_sha256": sha256_file(args.baseline),
        "candidate_sha256": sha256_file(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
