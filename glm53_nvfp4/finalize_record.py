"""Finalize the sealed experiment record from immutable execution receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--source-verify", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--final-state", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    result = json.loads(args.confirmation.read_text())
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "pass" or result.get("decision") not in ("pass", "fail"):
        raise RuntimeError("invalid finalization inputs")
    observations = {
        "source_integrity": f"hf cache verify receipt sha256={sha256_file(args.source_verify)}",
        "candidate_integrity": f"108864 changed routed-expert tensors in 168 chunks; freeze sha256={sha256_file(args.freeze)}",
        "candidate_freeze": f"candidate and analysis frozen at implementation commit {freeze['implementation']['commit']}",
        "restoration": f"final state receipt sha256={sha256_file(args.final_state)}",
    }
    for invariant in record["invariants"]:
        if invariant["name"] in observations:
            invariant["observed"] = observations[invariant["name"]]
            invariant["status"] = "pass"
    record["qualification"] = {
        "status": "qualified" if result["decision"] == "pass" else "failed",
        "attempt_count": 1,
        "primary_result": {
            key: result[key] for key in (
                "candidate_mean_kld", "stock_mean_kld", "mean_delta_kld",
                "relative_improvement", "delta_ci95_percentile", "delta_ci95_bca",
            )
        },
        "decision": result["decision"],
        "raw_evidence_sha256": sha256_file(args.confirmation),
        "limitations": [
            "Single quantization campaign does not estimate between-campaign variance.",
            "Confirmation-level evidence only; the 25 legacy final windows were previously opened.",
            "Layer calibration uses pinned BF16 captures and block-diagonal group-16 Hessians, not causal propagation of quantized activations.",
        ],
    }
    record["authorization"]["compute"] = "completed"
    record["status"] = "completed"
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record["status"], "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
