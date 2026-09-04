"""Finalize the sealed W6A8 clipped-SwiGLU diagnosis record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    result = json.loads(args.result.read_text())
    if record["status"] != "executing":
        raise RuntimeError("record must have an execution receipt before finalization")
    if result["analysis_plan_sha256"] != record["analysis_lock"]["analysis_plan_sha256"]:
        raise RuntimeError("result does not match the sealed analysis plan")
    if result["windows"] != 32 or result["decision"] not in {
        "pass-carrier-repaired",
        "fail-ladder-blocked",
        "continue-scale-and-staged-epilogue",
    }:
        raise RuntimeError("invalid clamp result")
    primary = result["clamp_vs_r0"]
    record["qualification"] = {
        "status": "qualified" if result["carrier_repaired"] else "failed",
        "attempt_count": 1,
        "primary_result": {
            "r0_mean_kld": result["arm_mean_kld"]["R0"],
            "c0_mean_kld": result["arm_mean_kld"]["C0"],
            "mean_delta_kld": primary["mean_delta_kld"],
            "delta_ci95_bca": primary["delta_ci95_bca"],
            "clamp_vs_w3": result["clamp_vs_w3"],
        },
        "decision": result["decision"],
        "raw_evidence_sha256": sha256_file(args.result),
        "limitations": [
            "Development conditional-fit role only; no protected confirmation or final role was opened.",
            "One cold execution per arm; paired uncertainty is over fixed document windows.",
        ],
    }
    record["authorization"]["compute"] = "completed"
    record["status"] = "completed"
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record["status"], "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
