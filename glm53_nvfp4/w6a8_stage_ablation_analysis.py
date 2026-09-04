"""Summarize the preregistered real-input W6A8 activation decomposition."""
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    raw = json.loads(args.raw.read_text())
    expected = set(plan["experts"])
    if {int(cell["expert"]) for cell in raw["cells"]} != expected:
        raise RuntimeError("raw result does not contain the frozen experts")
    stage_keys = {
        "fc1": "fc1_activation_quant_only_vs_exact",
        "fc2": "fc2_activation_quant_only_vs_exact",
        "both": "both_activation_quant_vs_exact",
        "fused_vs_both": "fused_vs_both_activation_quant",
    }
    gmean = {
        name: _gmean([float(cell["ablations"][key]["nmse"]) for cell in raw["cells"]])
        for name, key in stage_keys.items()
    }
    gmean["fused_vs_exact"] = _gmean([float(cell["nmse"]) for cell in raw["cells"]])
    closure = gmean["fused_vs_both"] <= 1e-4
    dominant = "activation-quantization" if gmean["both"] >= gmean["fused_vs_both"] else "fused-carrier"
    payload = {
        "schema": "glm53-w6a8-stage-ablation-result.v1",
        "geometric_mean_nmse": gmean,
        "dominant_component": dominant,
        "fc2_to_fc1_nmse_ratio": gmean["fc2"] / gmean["fc1"],
        "activation_to_fused_residual_ratio": gmean["both"] / gmean["fused_vs_both"],
        "fused_closure_at_1e-4": closure,
        "decision": (
            "activation-dominant-closed" if dominant == "activation-quantization" and closure
            else "activation-dominant-near-closure" if dominant == "activation-quantization"
            else "fused-carrier-dominant"
        ),
        "plan_sha256": sha256_file(args.plan),
        "raw_sha256": sha256_file(args.raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
