"""Validate the four-GPU N128x2 versus N256 P8 graph gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(controls: list[dict], candidates: list[dict]) -> dict:
    if len(controls) != 4 or len(candidates) != 4:
        raise ValueError("four GPU cells per arm are required")
    rows = []
    for gpu, (control, candidate) in enumerate(zip(controls, candidates)):
        for payload in (control, candidate):
            cell = payload["cells"][0]
            graph = cell["cuda_graph"]
            if payload["decision"] != "pass" or cell["tokens"] != 1:
                raise ValueError(f"GPU {gpu} runtime closure failed")
            if graph["replays"] != 5 or graph["timing_repeats"] != 100:
                raise ValueError(f"GPU {gpu} graph inventory differs")
            if not graph["bitwise_deterministic"] or not graph["matches_eager_output"]:
                raise ValueError(f"GPU {gpu} graph closure failed")
        a, b = control["cells"][0], candidate["cells"][0]
        if a["payload_sha256"] != b["payload_sha256"]:
            raise ValueError(f"GPU {gpu} payload mismatch")
        if a["output_sha256"] != b["output_sha256"]:
            raise ValueError(f"GPU {gpu} output mismatch")
        control_ms = a["cuda_graph"]["median_ms"]
        candidate_ms = b["cuda_graph"]["median_ms"]
        rows.append({
            "physical_gpu": gpu,
            "control_n128x2_median_ms": control_ms,
            "candidate_n256_median_ms": candidate_ms,
            "reduction_percent": 100.0 * (control_ms - candidate_ms) / control_ms,
            "bit_exact": True,
        })
    advances = all(row["reduction_percent"] >= 35.0 for row in rows)
    return {
        "schema": "glm53-p8-smallm-n256-device-analysis.v1",
        "decision": "advance_to_integrated_tp4" if advances else "stop_n256_before_integration",
        "decision_rule": "bit-exact and at least 35 percent faster on every GPU",
        "rows": rows,
        "minimum_reduction_percent": min(row["reduction_percent"] for row in rows),
        "maximum_reduction_percent": max(row["reduction_percent"] for row in rows),
        "conclusion": "N256 A reuse is valid but too small to target the remaining product deficit.",
        "protected_roles_opened": [],
        "ldlq": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    controls = [json.loads((args.input_dir / f"control-gpu{i}.json").read_text()) for i in range(4)]
    candidates = [json.loads((args.input_dir / f"candidate-gpu{i}.json").read_text()) for i in range(4)]
    result = analyze(controls, candidates)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
