"""Freeze external V5 validation for the fit/CF-selected P8 layer-3/20 subset."""
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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--attribution-analysis", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    expected = roles["selection_waves"]["1"]
    if len(expected) != 63:
        raise RuntimeError("expected exact V5 n=63 wave")
    attribution = json.loads(args.attribution_analysis.read_text())
    if attribution.get("supported_layers") != [3, 20]:
        raise RuntimeError("attribution did not freeze exactly layers 3 and 20")
    candidate_receipt = json.loads((args.candidate / "BF16_WEIGHT_OVERLAY_RECEIPT.json").read_text())
    baseline_receipt = json.loads((args.baseline / "BF16_LAYER_RECEIPT.json").read_text())
    if candidate_receipt.get("layers") != [3, 20] or not candidate_receipt.get("config_unchanged"):
        raise RuntimeError("candidate is not the frozen BF16-matched layer-3/20 subset")
    if baseline_receipt.get("bf16_layers") != [3, 19, 20]:
        raise RuntimeError("baseline is not the shared BF16 GPTQ control carrier")
    teacher = json.loads(args.teacher_verification.read_text())
    if teacher.get("status") != "pass" or teacher.get("file_count") != 63:
        raise RuntimeError("V5 teacher verification did not pass")
    payload = {
        "schema": "glm53-p8-supported-subset-v5-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "selection wave 1 reused as external validation",
        "role_limitation": "the role was previously scored for H16, but this codec subset was selected only from REAP fit and the disjoint conditional-fit panel before its candidate-control contrast was observed here; this is not pristine confirmation",
        "selection_wave": 1,
        "windows": 63,
        "represented_domains": ["axis1_general", "axis2_legal", "axis3_code_agentic"],
        "layers": [3, 20],
        "run_order": ["matched-gptq-bf16-control", "p8-layer3-20-bf16-candidate"],
        "candidate": {
            "model": str(args.candidate),
            "receipt": _file(args.candidate / "BF16_WEIGHT_OVERLAY_RECEIPT.json"),
            "physical_codec_bpw_for_replaced_weights": 4.25,
            "charged_bpw": 4.5,
        },
        "baseline": {
            "model": str(args.baseline),
            "receipt": _file(args.baseline / "BF16_LAYER_RECEIPT.json"),
            "source_format": "decoded full-Hessian GPTQ NVFP4",
            "bpw": 4.5,
        },
        "execution_match": "both arms are BF16 weight overlays through the identical InstantTensor/framework path; only layer-3/20 weight values differ",
        "estimand": "paired candidate minus matched-control mean teacher KLD and relative arithmetic-mean improvement",
        "decision_rule": "provisionally validate the P8 subset only if relative arithmetic-mean KLD improvement is at least 3 percent, the paired BCa upper bound is below zero, and every represented-domain mean delta is negative",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090426},
        "runtime_boundary": "pseudoquant encoder evidence only; native P8 kernel remains blocked by activation-path closure",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is a quality product",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "the 28 reserved confirmation logits are not read",
        "inputs": {
            "roles": _file(args.roles),
            "teacher_verification": _file(args.teacher_verification),
            "attribution_analysis": _file(args.attribution_analysis),
            "runtime_manifest": _file(args.runtime_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
