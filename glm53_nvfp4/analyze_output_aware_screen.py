"""Apply a frozen output-aware screen decision rule to its held-out rows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .shard_index import sha256_file


def _gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    raw = json.loads(args.input.read_text())
    variants = {}
    for row in raw["rows"]:
        if row["projection"] not in ("gate_proj", "up_proj"):
            continue
        if "validation" not in row:
            continue
        variants.setdefault(row["variant"], []).append(
            row["validation"]["output_nmse"]
        )
    aggregate = {
        variant: {"validation_output_nmse_gmean": _gmean(values), "cells": len(values)}
        for variant, values in variants.items()
    }
    candidate = aggregate["output-aware-hessian-mcg-k4"]["validation_output_nmse_gmean"]
    hessian = aggregate["hessian-mcg-k4"]["validation_output_nmse_gmean"]
    nvfp4 = aggregate["gptq-nvfp4"]["validation_output_nmse_gmean"]
    candidate_vs_hessian = 1.0 - candidate / hessian
    candidate_vs_nvfp4 = 1.0 - candidate / nvfp4
    rows_by_variant = {
        variant: [row for row in raw["rows"] if row["variant"] == variant]
        for variant in ("output-aware-hessian-mcg-k4", "hessian-mcg-k4")
    }
    neither_projection_worse = all(
        candidate_row["validation"]["output_nmse"]
        <= hessian_row["validation"]["output_nmse"]
        for candidate_row, hessian_row in zip(
            rows_by_variant["output-aware-hessian-mcg-k4"],
            rows_by_variant["hessian-mcg-k4"],
            strict=True,
        )
    )
    passed = (
        candidate_vs_hessian >= 0.10
        and candidate_vs_nvfp4 >= 0.10
        and neither_projection_worse
    )
    payload = {
        "schema": "glm53-output-aware-trellis-screen-analysis.v1",
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "aggregate": aggregate,
        "comparisons": {
            "candidate_improvement_vs_hessian_trellis": candidate_vs_hessian,
            "candidate_improvement_vs_gptq_nvfp4": candidate_vs_nvfp4,
            "hessian_trellis_improvement_vs_gptq_nvfp4": 1.0 - hessian / nvfp4,
            "neither_projection_worse_than_hessian_trellis": neither_projection_worse,
        },
        "decision": "advance" if passed else "fail-do-not-advance",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
