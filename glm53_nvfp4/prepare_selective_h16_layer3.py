"""Freeze identity-inclusive, per-block H16 selection after the forced-bank null."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--prior-full-build", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--conditional-fit-roles", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prior = json.loads(args.prior_analysis.read_text())
    if (
        prior.get("schema") != "glm53-nvfp4-v3.paired-conditional-fit.v1"
        or prior.get("decision") != "fail"
        or prior.get("mean_delta_kld", 0) <= 0
    ):
        raise RuntimeError("prior forced-bank KLD is not the preserved wrong-direction null")
    full = json.loads(args.prior_full_build.read_text())
    if full.get("status") != "pass" or full.get("physical_bpw", 99) > 4.501:
        raise RuntimeError("prior physical build is not closed at the target rate")
    role = json.loads(args.conditional_fit_roles.read_text())
    if len(role["roles"]["conditional-fit"]) != 32 or any(
        role["roles"][name] for name in ("fit", "selection", "confirmation", "final")
    ):
        raise RuntimeError("adaptive role boundary is invalid")
    payload = {
        "schema": "glm53-rotation-v7.selective-h16-layer3-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "role": "adaptive conditional-fit development after forced-H16 all-block null",
        "amendment_reason": (
            "the forced-H16 bank was 0.790 percent worse end-to-end despite positive local gain "
            "versus fixed H16; include identity in the same one-byte bank so each physical block "
            "can decline rotation under the fitted block-Hessian objective"
        ),
        "layer": 3,
        "experts": 288,
        "calibration": {"role": "fit", "strategy": "domain-balanced", "offset": 0,
                        "count": 256, "route_weight_power": 2},
        "candidate": {
            "name": "identity-inclusive-structured-bank256-blockgptq-all288",
            "transform": "independent identity-or-D-P-H16 per expert, projection, and physical K16 block",
            "include_identity": True,
            "variants_per_block": 256,
            "selection_method": "block-gptq-hessian",
            "search_grid": 8,
            "global_scale_refits": 2,
            "quantizer": "full-Hessian GPTQ, static within-native-group act order, slab128",
            "physical_format": "E2M1 packed weights plus E4M3/16 scales and FP32 tensor-family scale",
            "descriptor": "one uint8 bank index per K16 block; index 0 identity; procedural zero-table H16 law",
            "gate_up_scale": "one jointly refit FP32 scalar after separate gate/up selection",
            "down_scale": "independent jointly refit FP32 scalar",
        },
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "pseudoquant_execution": "serve Q(W R_b) R_b.T as BF16 effective weights for KLD linkage only",
        "conditional_fit": {
            "windows": 32,
            "run_order": ["stock-modelopt-nvfp4", "selective-h16-pseudoquant"],
            "stopping_rule": "one complete fresh run per arm; no rerolls, substitutions, or exclusions",
            "primary_estimand": "equal-window mean candidate KLD minus stock KLD",
            "analysis": "paired BCa 95 percent CI over windows",
            "development_advance_rule": "advance iff mean candidate KLD is lower than stock; report CI",
        },
        "claim_boundary": "conditional-fit32 is reused adaptive data and cannot qualify a final claim",
        "protected_boundary": "selection, confirmation, final, and 28 reserved confirmation logits remain unopened",
        "prior": {"analysis": _file(args.prior_analysis), "full_build": _file(args.prior_full_build)},
        "inputs": {
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_manifest),
            "roles": _file(args.roles),
            "conditional_fit_roles": _file(args.conditional_fit_roles),
            "materializer": _file(args.materializer),
            "selector": _file(args.selector),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
