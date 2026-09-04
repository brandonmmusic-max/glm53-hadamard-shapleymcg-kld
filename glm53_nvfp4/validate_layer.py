"""Fail-closed validation and summary for a complete quantized expert layer."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .shard_index import sha256_file


def ratio_fails(method: str, ratio: float) -> bool:
    if method in {"gptq", "mr-gptq", "fpquant-mr-gptq", "trellis-nvfp4", "identity-k4"}:
        return ratio >= 1.0
    if method == "rtn":
        return abs(ratio - 1.0) > 1e-8
    raise ValueError(f"unsupported quantization method {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--projections", choices=("all", "gate-up"), default="all")
    args = parser.parse_args()
    seen: dict[int, dict] = {}
    inputs = []
    methods = set()
    for path in args.receipt:
        payload = json.loads(path.read_text())
        if payload["layer"] != args.layer:
            raise ValueError(f"wrong layer in {path}")
        methods.add(payload["algorithm"].get("quant_method", "gptq"))
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for expert, metrics in payload["metrics"].items():
            expert = int(expert)
            if expert in seen:
                raise ValueError(f"duplicate expert {expert}")
            seen[expert] = metrics
    if len(methods) != 1:
        raise ValueError(f"mixed quantization methods in one layer: {sorted(methods)}")
    method = next(iter(methods))
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
        failed.extend(
            (expert, projection, seen[expert][projection]["ratio"])
            for expert in seen
            if ratio_fails(method, seen[expert][projection]["ratio"])
        )
    sample_counts = [seen[expert]["samples"] for expert in seen]
    status = "pass" if not missing and len(seen) == 288 and not failed and min(sample_counts) >= 16 else "fail"
    output = {
        "schema": "glm53-nvfp4-v11.layer-validation.v1",
        "layer": args.layer,
        "quant_method": method,
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
