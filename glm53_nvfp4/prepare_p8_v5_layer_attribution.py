"""Freeze exploratory V5 attribution for the supported P8 layer-3/20 subset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--teacher-verification", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--subset-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    roles = json.loads(args.roles.read_text())
    expected = roles["selection_waves"]["1"]
    if len(expected) != 63:
        raise RuntimeError("expected exact V5 n=63 wave")
    teacher = json.loads(args.teacher_verification.read_text())
    if teacher.get("status") != "pass" or teacher.get("file_count") != 63:
        raise RuntimeError("V5 teacher verification did not pass")
    subset = json.loads(args.subset_analysis.read_text())
    if subset.get("decision") != "fail-external-v5":
        raise RuntimeError("attribution is only valid after the frozen subset gate failed")

    candidates = {}
    for path in args.candidate:
        receipt_path = path / "BF16_WEIGHT_OVERLAY_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        if len(receipt.get("layers", [])) != 1 or not receipt.get("config_unchanged"):
            raise RuntimeError(f"candidate is not one BF16-matched layer: {path}")
        layer = int(receipt["layers"][0])
        candidates[str(layer)] = {"model": str(path), "receipt": _file(receipt_path)}
    if set(candidates) != {"3", "20"}:
        raise RuntimeError(f"expected layers 3 and 20, got {sorted(candidates)}")

    baseline_receipt = args.baseline / "BF16_LAYER_RECEIPT.json"
    payload = {
        "schema": "glm53-p8-v5-layer-attribution-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "opened V5 selection wave reused for exploratory attribution",
        "role_limitation": (
            "the combined layer-3/20 contrast has already been observed on this role; "
            "these individual-layer results are developmental and cannot qualify a protected claim"
        ),
        "selection_wave": 1,
        "windows": 63,
        "represented_domains": ["axis1_general", "axis2_legal", "axis3_code_agentic"],
        "baseline": {"model": str(args.baseline), "receipt": _file(baseline_receipt)},
        "candidates": candidates,
        "run_order": ["3", "20"],
        "execution_match": (
            "every arm is a BF16 weight overlay through the identical InstantTensor/framework path; "
            "only one layer's reconstructed values differ"
        ),
        "estimand": "per-layer candidate minus decoded-GPTQ BF16-overlay teacher KLD",
        "decision_rule": (
            "call a layer externally supported for redesign only if mean delta is at most -0.0006 nats, "
            "its Bonferroni-adjusted 97.5 percent paired BCa upper bound is below zero, and all three "
            "domain mean deltas are negative"
        ),
        "multiplicity": "two comparisons; family alpha 0.05, individual two-sided alpha 0.025",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090427},
        "runtime_boundary": "pseudoquant encoder evidence only; native P8 remains blocked by activation-path closure",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is a quality product",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "the 28 reserved confirmation logits are not read",
        "inputs": {
            "roles": _file(args.roles),
            "teacher_verification": _file(args.teacher_verification),
            "failed_subset_analysis": _file(args.subset_analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
