"""Select the frozen checkpoint-family ridge ratio."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def _gmean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan_sha = sha256_file(args.plan)
    rows, inputs = [], []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != plan_sha or raw["protected_roles_opened"]:
            raise ValueError("invalid input")
        rows += raw["rows"]
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    controls = {r["expert"]: r["evaluation_full_expert_nmse"] for r in rows if r["arm"] == "down-h16-gptq"}
    candidates = {}
    for row in rows:
        if row["arm"] == "identity-k4-down-h16":
            candidates.setdefault(str(row["ridge_ratio"]), {})[row["expert"]] = row["evaluation_full_expert_nmse"]
    experts = sorted(controls)
    if any(sorted(v) != experts for v in candidates.values()):
        raise ValueError("coverage mismatch")
    control_gmean = _gmean([controls[e] for e in experts])
    grid = {}
    for ridge, values in candidates.items():
        gmean = _gmean([values[e] for e in experts])
        grid[ridge] = {"geometric_mean_nmse": gmean, "relative_improvement_vs_control": 1 - gmean / control_gmean, "wins": sum(values[e] < controls[e] for e in experts)}
    selected = min(grid, key=lambda key: grid[key]["geometric_mean_nmse"])
    passed = grid[selected]["relative_improvement_vs_control"] >= 0.01
    payload = {"schema": "glm53-identity-k4-ridge-tuning-analysis.v1", "plan": {"path": str(args.plan), "sha256": plan_sha}, "inputs": inputs, "experts": experts, "control_geometric_mean_nmse": control_gmean, "grid": grid, "selected_ridge_ratio": float(selected), "decision": "pass-freeze-and-validate" if passed else "fail-stop-identity-k4", "protected_roles_opened": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
