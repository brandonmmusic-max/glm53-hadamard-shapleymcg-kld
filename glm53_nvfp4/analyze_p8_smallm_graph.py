"""Validate the frozen four-GPU captured-graph P8 small-M gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(baselines: list[dict], candidates: list[dict]) -> dict:
    if len(baselines) != 4 or len(candidates) != 4:
        raise ValueError("the frozen inventory requires four GPUs per arm")
    rows = []
    for gpu, (baseline, candidate) in enumerate(zip(baselines, candidates)):
        for payload in (baseline, candidate):
            if payload["decision"] != "pass":
                raise ValueError(f"GPU {gpu} contains a failed runtime closure")
            cell = payload["cells"][0]
            graph = cell["cuda_graph"]
            if cell["tokens"] != 1 or graph is None:
                raise ValueError(f"GPU {gpu} is not an M1 graph cell")
            if graph["replays"] != 5 or graph["timing_repeats"] != 100:
                raise ValueError(f"GPU {gpu} graph inventory differs")
            if not graph["bitwise_deterministic"] or not graph["matches_eager_output"]:
                raise ValueError(f"GPU {gpu} graph closure failed")
        if baseline["mode"] != "monolithic" or baseline["small_m_scheduler"]:
            raise ValueError(f"GPU {gpu} baseline is not monolithic")
        if not candidate["small_m_scheduler"]:
            raise ValueError(f"GPU {gpu} candidate is not small-M")
        base_cell, cand_cell = baseline["cells"][0], candidate["cells"][0]
        if base_cell["payload_sha256"] != cand_cell["payload_sha256"]:
            raise ValueError(f"GPU {gpu} payload differs across arms")
        if base_cell["output_sha256"] != cand_cell["output_sha256"]:
            raise ValueError(f"GPU {gpu} output differs across arms")
        base_ms = base_cell["cuda_graph"]["median_ms"]
        cand_ms = cand_cell["cuda_graph"]["median_ms"]
        reduction = 100.0 * (base_ms - cand_ms) / base_ms
        rows.append({
            "physical_gpu": gpu,
            "baseline_median_ms": base_ms,
            "candidate_median_ms": cand_ms,
            "reduction_percent": reduction,
            "speedup": base_ms / cand_ms,
            "payload_sha256": base_cell["payload_sha256"],
            "output_sha256": base_cell["output_sha256"],
            "bit_exact": True,
        })
    passed = all(row["reduction_percent"] >= 50.0 for row in rows)
    return {
        "schema": "glm53-p8-smallm-graph-analysis.v1",
        "decision": "advance_to_integrated_tp4" if passed else "stop_before_integration",
        "decision_rule": (
            "all four GPU cells must be payload-identical, output-bit-exact, "
            "graph deterministic, and at least 50 percent faster"
        ),
        "rows": rows,
        "minimum_reduction_percent": min(row["reduction_percent"] for row in rows),
        "maximum_reduction_percent": max(row["reduction_percent"] for row in rows),
        "protected_roles_opened": [],
        "ldlq": False,
        "scope": "single-layer M1 CUDA-graph closure; not serving tokens/s or KLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baselines = [json.loads((args.input_dir / f"valid-baseline-gpu{i}.json").read_text()) for i in range(4)]
    candidates = [json.loads((args.input_dir / f"valid-candidate-gpu{i}.json").read_text()) for i in range(4)]
    result = analyze(baselines, candidates)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["decision"] != "advance_to_integrated_tp4":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
