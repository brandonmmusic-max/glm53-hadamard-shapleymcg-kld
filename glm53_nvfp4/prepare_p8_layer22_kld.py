"""Freeze the matched-path layer-22 P8 end-to-end KLD gate."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPECTED_DOMAINS = {
    "axis1_general": 8,
    "axis2_legal": 8,
    "axis3_code_agentic": 8,
    "axis4_reasoning_termination": 8,
}


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _overlay(path: Path) -> dict:
    receipt_path = path / "BF16_LAYER_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("bf16_layers") != [22] or receipt.get("required_load_format") != "instanttensor":
        raise RuntimeError(f"not an exact layer-22 BF16 overlay: {path}")
    if receipt.get("redirected_tensors") != 864:
        raise RuntimeError(f"unexpected redirected tensor count: {path}")
    return {"model": str(path), "receipt": _file(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--causal-analysis", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32 or roles.get("domains") != EXPECTED_DOMAINS:
        raise RuntimeError("KLD gate requires the exact domain-balanced n=32 role")
    if any(roles["roles"][role] for role in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("KLD role file exposes data outside conditional-fit")
    causal = json.loads(args.causal_analysis.read_text())
    if causal.get("decision") != "pass-build-layer22-kld-candidate" or causal.get("ldlq_used"):
        raise RuntimeError("layer-22 causal screen did not qualify the no-LDLQ candidate")

    candidate = _overlay(args.candidate)
    candidate.update({"physical_codec_bpw": 4.25, "charged_bpw": 4.5})
    baseline = _overlay(args.baseline)
    baseline.update({"source_format": "decoded full-Hessian GPTQ NVFP4", "bpw": 4.5})
    payload = {
        "schema": "glm53-p8-layer22-matched-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit developmental gate",
        "windows": 32,
        "domain_counts": EXPECTED_DOMAINS,
        "layer": 22,
        "run_order": ["matched-gptq-bf16-control", "p8-layer22-bf16-candidate"],
        "candidate": candidate,
        "baseline": baseline,
        "execution_match": "both arms replace exactly layer 22 as BF16 through the identical InstantTensor/framework path; only reconstructed weight values differ",
        "primary_estimand": "paired equal-window mean KLD(candidate) minus KLD(decoded-GPTQ control)",
        "decision_rule": "pass only if mean delta is at most -0.0014 nats and the paired BCa 95 percent upper bound is below zero",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090432},
        "stopping_rule": "run each arm once on the exact 32 windows; preserve failures; no exclusions, substitutions, or rerolls",
        "runtime_boundary": "a pass qualifies pseudoquant encoder quality only; P8 device closure remains blocked by the E4M3 activation carrier",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4 and is a quality product; P4 is the NVFP4-speed product",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "roles": _file(args.roles),
            "causal_analysis": _file(args.causal_analysis),
            "runtime_manifest": _file(args.runtime_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
