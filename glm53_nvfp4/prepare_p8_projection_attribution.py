"""Freeze layer-22 P8 per-projection KLD attribution before execution."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--prior-kld-analysis", type=Path, required=True)
    parser.add_argument("--route-analysis", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("requires exact opened conditional-fit n=32")
    if any(roles["roles"][key] for key in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("attribution role file must expose only conditional-fit")
    prior = json.loads(args.prior_kld_analysis.read_text())
    route = json.loads(args.route_analysis.read_text())
    if prior.get("decision") != "fail-pseudoquant-kld":
        raise RuntimeError("expected the failed full-layer P8 gate")
    if route.get("decision") != "invalid-pre-layer-divergence":
        raise RuntimeError("projection attribution follows only an invalid causal route probe")
    candidates = {}
    for path in args.candidate:
        receipt_path = path / "BF16_WEIGHT_OVERLAY_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("layers") != [22] or len(receipt.get("projections", [])) != 1:
            raise RuntimeError(f"not an exact layer-22 single-projection overlay: {path}")
        projection = receipt["projections"][0]
        candidates[projection] = {"model": str(path), "receipt": _file(receipt_path)}
    if set(candidates) != set(PROJECTIONS):
        raise RuntimeError(f"expected exactly {PROJECTIONS}; got {sorted(candidates)}")
    payload = {
        "schema": "glm53-p8-layer22-projection-attribution-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "opened conditional-fit developmental attribution",
        "windows": 32,
        "layer": 22,
        "baseline": {
            "model": str(args.baseline),
            "receipt": _file(args.baseline / "BF16_LAYER_RECEIPT.json"),
        },
        "candidates": candidates,
        "run_order": list(PROJECTIONS),
        "estimand": "single-projection P8 reconstruction minus matched decoded-GPTQ KLD",
        "decision_rule": "a projection is directionally supported only if mean delta is at most -0.0004 nats and its Bonferroni-adjusted 98.333 percent paired BCa upper bound is below zero; this diagnoses the encoder and does not itself pass gate 4a",
        "multiplicity": "three comparisons; family alpha 0.05, individual two-sided alpha 1/60",
        "bootstrap": {
            "method": "paired BCa",
            "unit": "window",
            "replicates": 50000,
            "seed": 2026090433,
        },
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "protected_boundary": "no selection, confirmation, or final data are read; the 28 confirmation logits remain unopened",
        "inputs": {
            "roles": _file(args.roles),
            "prior_kld_analysis": _file(args.prior_kld_analysis),
            "invalid_route_analysis": _file(args.route_analysis),
            "runtime_manifest": _file(args.runtime_manifest),
            "runner": _file(args.runner),
            "analyzer": _file(Path(__file__).with_name("analyze_p8_projection_attribution.py")),
            "preparer": _file(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
