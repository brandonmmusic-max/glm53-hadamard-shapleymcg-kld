"""Freeze the adaptive n=32 KLD screen for the 3/19/20 P8 composite."""
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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--layer-analysis", type=Path, action="append", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--invalidated-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("composite screen requires exactly 32 conditional-fit windows")
    if any(roles["roles"][role] for role in ("selection", "confirmation", "final")):
        raise RuntimeError("adaptive role manifest must not expose protected roles")
    expected_domains = {
        "axis1_general": 8,
        "axis2_legal": 8,
        "axis3_code_agentic": 8,
        "axis4_reasoning_termination": 8,
    }
    if roles["domains"] != expected_domains:
        raise RuntimeError("conditional-fit role is not domain balanced")

    analyses = []
    layers = []
    for path in args.layer_analysis:
        value = json.loads(path.read_text())
        decision = value.get("decision", "")
        if not decision.startswith("pass-build-layer"):
            raise RuntimeError(f"cross-layer causal screen did not pass: {path}")
        layer = int(decision.split("pass-build-layer", 1)[1].split("-", 1)[0])
        layers.append(layer)
        analyses.append(_file(path))
    if sorted(layers) != [3, 19, 20]:
        raise RuntimeError(f"expected passed layers [3, 19, 20], got {sorted(layers)}")

    candidate_receipt = args.candidate / "BF16_LAYER_RECEIPT.json"
    candidate_overlay = json.loads((args.candidate / "OVERLAY.json").read_text())
    if candidate_overlay.get("bf16_layers") != [3, 19, 20]:
        raise RuntimeError("candidate overlay does not contain exactly layers 3/19/20")
    baseline_overlay = json.loads((args.baseline / "OVERLAY.json").read_text())
    if baseline_overlay.get("bf16_layers") != [3, 19, 20]:
        raise RuntimeError("baseline is not the exact three-layer BF16 control overlay")
    baseline_receipt = args.baseline / "BF16_LAYER_RECEIPT.json"

    payload = {
        "schema": "glm53-hessian-trellis-p8-composite-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit",
        "role_status": "adaptive development role previously used by the one-layer screen",
        "windows": 32,
        "domain_counts": expected_domains,
        "layers": [3, 19, 20],
        "run_order": ["matched-gptq-nvfp4-bf16-control", "p8-bf16-pseudoquant-composite"],
        "candidate": {
            "model": str(args.candidate),
            "index": _file(args.candidate / "model.safetensors.index.json"),
            "receipt": _file(candidate_receipt),
            "physical_codec_bpw_for_replaced_weights": 4.25,
            "charged_bpw_for_gate": 4.5,
            "execution": "dense BF16 pseudoquant overlay; numerical encoder gate only",
        },
        "baseline": {
            "model": str(args.baseline),
            "index": _file(args.baseline / "model.safetensors.index.json"),
            "receipt": _file(baseline_receipt),
            "format": "exact decoded BF16 reconstruction of full-Hessian calibrated ModelOpt NVFP4 at the same three layers",
            "bpw": 4.5,
            "execution": "dense BF16 overlay through the identical framework path as the candidate",
        },
        "primary_estimand": "equal-window mean KLD(candidate BF16 overlay) minus KLD(matched decoded-GPTQ BF16 overlay)",
        "decision_rule": "advance the composite only if mean delta is at most -0.0014 nats and the paired BCa 95 percent upper bound is below zero",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 20000, "seed": 20260902},
        "stopping_rule": "run each arm once on the exact 32 windows; no exclusions, substitutions, or rerolls",
        "runtime_boundary": "a pass is developmental pseudoquant evidence only; P8 kernel closure remains blocked by W6A8 activation-path failure",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is the quality product; P4 is the NVFP4-speed product",
        "protected_boundary": "selection, confirmation, and final are not read by this screen; the 28 reserved confirmation logits remain unopened",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "deviation": (
            {
                "description": "the prior composite comparison used packed NVFP4/Humming for the control and a BF16 framework overlay for the candidate; it is retained but invalid for encoder attribution",
                "invalidated_analysis": _file(args.invalidated_analysis),
            }
            if args.invalidated_analysis is not None
            else None
        ),
        "inputs": {
            "roles": _file(args.roles),
            "layer_analyses": analyses,
            "runtime_manifest": _file(args.runtime_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
