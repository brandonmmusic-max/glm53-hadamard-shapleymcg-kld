"""Analyze the frozen two-split REAP W6A8 FC1 scale screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def _gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _index(payload: dict[str, object]) -> dict[float, dict[int, float]]:
    result: dict[float, dict[int, float]] = {}
    for cell in payload["cells"]:  # type: ignore[index]
        result.setdefault(float(cell["a1_gscale"]), {})[int(cell["expert"])] = float(cell["nmse"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    selection = _index(json.loads(args.selection.read_text()))
    validation = _index(json.loads(args.validation.read_text()))
    experts = set(plan["experts"])
    for phase, values in selection.items():
        if set(values) != experts:
            raise RuntimeError(f"selection phase {phase} lacks the frozen expert set")
    selection_gmean = {phase: _gmean(list(values.values())) for phase, values in selection.items()}
    selected_phase = min(selection_gmean, key=lambda phase: (selection_gmean[phase], phase))
    if set(validation) != {1.0, selected_phase}:
        raise RuntimeError("validation must contain only unit and the frozen selected phase")
    for phase, values in validation.items():
        if set(values) != experts:
            raise RuntimeError(f"validation phase {phase} lacks the frozen expert set")
    baseline = validation[1.0]
    candidate = validation[selected_phase]
    baseline_gmean = _gmean(list(baseline.values()))
    candidate_gmean = _gmean(list(candidate.values()))
    improvement = (baseline_gmean - candidate_gmean) / baseline_gmean
    expert_wins = sum(candidate[expert] < baseline[expert] for expert in experts)
    material = improvement >= 0.05 and expert_wins >= 12
    payload = {
        "schema": "glm53-w6a8-reap-scale-result.v1",
        "selected_phase": selected_phase,
        "selection_gmean_nmse": {str(key): value for key, value in selection_gmean.items()},
        "selection_improvement_vs_unit": (
            selection_gmean[1.0] - selection_gmean[selected_phase]
        ) / selection_gmean[1.0],
        "validation_unit_gmean_nmse": baseline_gmean,
        "validation_selected_gmean_nmse": candidate_gmean,
        "validation_improvement_vs_unit": improvement,
        "validation_expert_wins": expert_wins,
        "experts": len(experts),
        "decision": "fc1-scale-material" if material else "fc1-scale-not-dominant",
        "plan_sha256": sha256_file(args.plan),
        "selection_sha256": sha256_file(args.selection),
        "validation_sha256": sha256_file(args.validation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
