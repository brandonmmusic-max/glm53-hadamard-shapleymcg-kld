"""Freeze per-expert FC2 phases from the preregistered tuning split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    raw = json.loads(args.tuning.read_text())
    grid = set(float(value) for value in plan["phase_grid"])
    by_expert: dict[int, list[dict[str, object]]] = {}
    for cell in raw["cells"]:
        if float(cell["a1_gscale"]) != 1.0 or float(cell["a2_gscale"]) not in grid:
            raise RuntimeError("tuning cell falls outside the frozen grid")
        by_expert.setdefault(int(cell["expert"]), []).append(cell)
    if set(by_expert) != set(plan["experts"]):
        raise RuntimeError("tuning does not contain the frozen expert set")
    scales: dict[str, dict[str, float]] = {}
    for expert, cells in by_expert.items():
        if {float(cell["a2_gscale"]) for cell in cells} != grid:
            raise RuntimeError(f"expert {expert} does not contain the entire phase grid")
        winner = min(cells, key=lambda cell: (float(cell["nmse"]), float(cell["a2_gscale"])))
        scales[str(expert)] = {"a1_gscale": 1.0, "a2_gscale": float(winner["a2_gscale"])}
    payload = {
        "schema": "glm53-w6a8-fc2-scale-map.v1",
        "plan_sha256": sha256_file(args.plan),
        "tuning_sha256": sha256_file(args.tuning),
        "selection_rule": "minimum fused routed-output NMSE per expert; ties choose smaller phase",
        "scales": scales,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
