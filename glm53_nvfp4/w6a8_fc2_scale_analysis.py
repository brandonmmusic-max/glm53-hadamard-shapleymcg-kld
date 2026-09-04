"""Analyze frozen per-expert FC2 phase validation against the unit control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def _gmean(values: dict[int, float]) -> float:
    return math.exp(sum(math.log(value) for value in values.values()) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scale-map", type=Path, required=True)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    unit_raw = json.loads(args.unit.read_text())
    candidate_raw = json.loads(args.candidate.read_text())
    unit = {int(cell["expert"]): float(cell["nmse"]) for cell in unit_raw["cells"]}
    candidate = {int(cell["expert"]): float(cell["nmse"]) for cell in candidate_raw["cells"]}
    if set(unit) != set(plan["experts"]) or set(candidate) != set(unit):
        raise RuntimeError("validation results do not match the frozen expert set")
    unit_gmean = _gmean(unit)
    candidate_gmean = _gmean(candidate)
    improvement = (unit_gmean - candidate_gmean) / unit_gmean
    wins = sum(candidate[expert] < unit[expert] for expert in unit)
    material = improvement >= 0.05 and wins >= 12
    payload = {
        "schema": "glm53-w6a8-fc2-scale-result.v1",
        "unit_gmean_nmse": unit_gmean,
        "candidate_gmean_nmse": candidate_gmean,
        "validation_improvement": improvement,
        "validation_expert_wins": wins,
        "experts": len(unit),
        "decision": "material-expand" if material else "fail-retain-unit",
        "plan_sha256": sha256_file(args.plan),
        "scale_map_sha256": sha256_file(args.scale_map),
        "unit_sha256": sha256_file(args.unit),
        "candidate_sha256": sha256_file(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
