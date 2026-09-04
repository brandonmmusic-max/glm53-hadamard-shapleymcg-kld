"""Freeze projection-wise MCG alphas from all fit-expert raw parts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    rows, inputs = [], []
    for path in args.input:
        raw = json.loads(path.read_text())
        if raw["plan"]["sha256"] != sha256_file(args.plan):
            raise RuntimeError("raw part does not match frozen plan")
        rows.extend(raw["rows"])
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    expected = {(expert, projection, alpha) for expert in plan["fit_experts"] for projection in plan["projections"] for alpha in plan["alpha_grid"]}
    cells = {(int(row["expert"]), row["projection"], float(row["alpha"])): row for row in rows}
    if set(cells) != expected:
        raise RuntimeError("alpha raw coverage mismatch")
    grid, selected = {}, {}
    for projection in plan["projections"]:
        grid[projection] = {}
        for alpha in plan["alpha_grid"]:
            values = [cells[(expert, projection, alpha)]["evaluation_output_nmse"] for expert in plan["fit_experts"]]
            grid[projection][str(alpha)] = {
                "geometric_mean_output_nmse": math.exp(sum(math.log(value) for value in values) / len(values)),
                "wins_vs_alpha2": sum(value < cells[(expert, projection, 2.0)]["evaluation_output_nmse"] for expert, value in zip(plan["fit_experts"], values, strict=True)),
            }
        selected[projection] = min(plan["alpha_grid"], key=lambda alpha: (grid[projection][str(alpha)]["geometric_mean_output_nmse"], alpha))
    payload = {
        "schema": "glm53-p8-mcg-projection-alpha-analysis.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "inputs": inputs,
        "grid": grid,
        "selected_alpha": selected,
        "relative_improvement_vs_alpha2": {projection: 1.0 - grid[projection][str(selected[projection])]["geometric_mean_output_nmse"] / grid[projection]["2.0"]["geometric_mean_output_nmse"] for projection in plan["projections"]},
        "decision": "freeze-selected-alphas-for-expert-disjoint-causal-validation",
        "protected_roles_opened": [],
        "algorithm_exclusion": "no LDLQ or BlockLDLQ used",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
