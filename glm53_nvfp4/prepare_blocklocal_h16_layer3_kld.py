"""Freeze a new development hypothesis for all-expert block-local H16 KLD."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-analysis", type=Path, required=True)
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
        prior.get("schema") != "glm53-blocklocal-signed-h16-screen-analysis.v1"
        or prior.get("decision") != "fail-do-not-open-v4-selection"
        or prior.get("protected_roles_opened") != []
    ):
        raise RuntimeError("prior screen is not the preserved failed bank256 analysis")
    roles = json.loads(args.conditional_fit_roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("development role must contain exactly 32 conditional-fit windows")
    if any(roles["roles"][name] for name in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("conditional-fit role exposes a protected partition")
    payload = {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "role": "new adaptive development hypothesis after failed surrogate promotion gate",
        "amendment_reason": (
            "the user explicitly relaxed the intermediate expert-window gate so the "
            "10.5 percent identity-relative block-GPTQ screen signal can receive one "
            "end-to-end conditional-fit linkage test before Shapley allocation"
        ),
        "layer": 3,
        "experts": 288,
        "calibration": {
            "role": "fit",
            "strategy": "domain-balanced",
            "offset": 0,
            "count": 256,
            "route_weight_power": 2,
        },
        "candidate": {
            "name": "structured-bank256-blockgptq-all288",
            "transform": "independent D-P-H16 per expert, projection, and physical K16 block",
            "variants_per_block": 256,
            "selection_method": "block-gptq-hessian",
            "search_grid": 8,
            "global_scale_refits": 2,
            "quantizer": "full-Hessian GPTQ, static within-native-group act order, slab128",
            "physical_format": "E2M1 packed weights plus E4M3/16 scales and FP32 tensor-family scale",
            "descriptor": "one uint8 bank index per physical K16 block; procedural bank uses zero runtime table bytes",
            "gate_up_scale": "one jointly refit FP32 scalar after separate gate/up transform selection",
            "down_scale": "independent jointly refit FP32 scalar",
        },
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "pseudoquant_execution": (
            "serve inverse-rotated BF16 effective weights Q(W R) R^T through the "
            "stock fused MoE; this measures numerical linkage, not compact runtime"
        ),
        "conditional_fit": {
            "windows": 32,
            "run_order": ["stock-modelopt-nvfp4", "blocklocal-h16-pseudoquant"],
            "stopping_rule": "one complete run per arm; no rerolls, substitutions, or exclusions",
            "primary_estimand": "equal-window mean candidate KLD minus stock KLD",
            "analysis": "paired BCa 95 percent CI over windows; report all four domains",
            "development_advance_rule": (
                "advance to a separately frozen unopened selection wave only if mean "
                "candidate KLD is lower than stock; CI exclusion is reported but not "
                "required at this adaptive-development stage"
            ),
        },
        "claim_boundary": (
            "conditional-fit32 is already-open adaptive data; a favorable result is "
            "not protected qualification and does not prove the indexed native prologue"
        ),
        "protected_boundary": "selection, confirmation, final, and the 28 reserved confirmation logits remain unopened",
        "prior": _file(args.prior_analysis),
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
