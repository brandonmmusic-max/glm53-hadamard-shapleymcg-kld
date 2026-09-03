"""Fail-closed validation and summary for a complete quantized expert layer."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--projections", choices=("all", "gate-up"), default="all")
    args = parser.parse_args()
    seen: dict[int, dict] = {}
    inputs = []
    for path in args.receipt:
        payload = json.loads(path.read_text())
        if payload["layer"] != args.layer:
            raise ValueError(f"wrong layer in {path}")
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for expert, metrics in payload["metrics"].items():
            expert = int(expert)
            if expert in seen:
                raise ValueError(f"duplicate expert {expert}")
            seen[expert] = metrics
    missing = sorted(set(range(288)) - seen.keys())
    projections = {}
    failed = []
    projection_names = ("gate", "up", "down") if args.projections == "all" else ("gate", "up")
    for projection in projection_names:
        values = [seen[expert][projection]["ratio"] for expert in sorted(seen)]
        projections[projection] = {
            "minimum_ratio": min(values),
            "median_ratio": statistics.median(values),
            "mean_ratio": statistics.mean(values),
            "maximum_ratio": max(values),
            "ratios_at_or_above_one": sum(value >= 1.0 for value in values),
        }
        failed.extend((expert, projection, seen[expert][projection]["ratio"]) for expert in seen if seen[expert][projection]["ratio"] >= 1.0)
    sample_counts = [seen[expert]["samples"] for expert in seen]
    status = "pass" if not missing and len(seen) == 288 and not failed and min(sample_counts) >= 16 else "fail"
    output = {
        "schema": "glm53-nvfp4-v2.layer-validation.v1",
        "layer": args.layer,
        "status": status,
        "experts": len(seen),
        "missing_experts": missing,
        "minimum_samples": min(sample_counts) if sample_counts else 0,
        "maximum_samples": max(sample_counts) if sample_counts else 0,
        "projections": projections,
        "failed_metrics": failed,
        "inputs": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"layer": args.layer, "status": status, "projections": projections}, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
