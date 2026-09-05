"""Freeze the V8 actual-KLD search for a compact shared middle rotation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


GRID = (
    ("m0250", -0.25),
    ("m0125", -0.125),
    ("m00625", -0.0625),
    ("m003125", -0.03125),
    ("zero", 0.0),
    ("p003125", 0.03125),
    ("p00625", 0.0625),
    ("p0125", 0.125),
    ("p0250", 0.25),
)


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return result


def build(args: argparse.Namespace) -> dict:
    prior_v6 = json.loads(args.prior_v6_analysis.read_text())
    prior_v7 = json.loads(args.prior_v7_analysis.read_text())
    if prior_v6.get("decision") != "fail" or prior_v7.get("decision") != "fail":
        raise RuntimeError("V8 is justified only after both V6 and V7 failed")
    train = json.loads(args.train_roles.read_text())
    tune = json.loads(args.tune_roles.read_text())
    if train["counts"]["fit"] != 32 or tune["counts"]["fit"] != 32:
        raise RuntimeError("V8 requires train32 and tune32")
    if {x["id"] for x in train["roles"]["fit"]} & {
        x["id"] for x in tune["roles"]["fit"]
    }:
        raise RuntimeError("train/tune overlap")
    result = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-search-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "role": "adaptive fit-only KLD search followed by already-open conditional-fit32",
        "observation": (
            "V6 forced block-local H16 and V7 identity-selective block H16 both "
            "improved their local surrogate but worsened end-to-end KLD"
        ),
        "hypothesis": (
            "a compact layer-shared partial butterfly on the down input can retain "
            "the down-family signal without incompatible block-specific bases"
        ),
        "falsification": (
            "no nonzero angle beats both the matched zero-angle full-GPTQ control "
            "and stock ModelOpt NVFP4 on tune32 mean causal KLD"
        ),
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "layer": 3,
        "experts": 288,
        "candidate": {
            "scope": "down-only; gate/up remain bitwise carrier NVFP4",
            "rotation": "one layer-shared 16x16 FP32 SO butterfly repeated over 128 middle K16 blocks and 288 experts",
            "parameterization": "four H16-wired Givens stages with one shared scalar angle",
            "identity": "angle_pi=0",
            "hadamard_family_endpoint": "abs(angle_pi)=0.25 gives a dense signed-Hadamard member",
            "stored_table_bytes": 1024,
            "per_block_or_expert_descriptors": 0,
            "runtime_precision": "FP32 table validated, then cast to served BF16 by current Humming hook",
        },
        "encoder": {
            "format": "ModelOpt NVFP4 E2M1 plus E4M3-per-16 and FP32 tensor scale",
            "method": "full-Hessian GPTQ with static native-group order",
            "search_grid": 12,
            "global_scale_refits": 2,
            "objective": "actual packed down operator under routed fit activations",
            "ldlq": False,
        },
        "calibration": {
            "role": "fit",
            "samples_per_expert": 256,
            "sampling_strategy": "domain-balanced",
            "route_weight_power": 2,
        },
        "search": {
            "coarse_grid": [
                {"id": name, "angle_pi": value} for name, value in GRID
            ],
            "training_measurement": "all-vocabulary all-causal-row teacher KLD on train32",
            "selection_rule": "lowest finite equal-window mean KLD on train32; no NMSE substitution",
            "tune_arms": "selected nonzero angle, zero-angle matched control, and stock ModelOpt control",
            "tune_advance_rule": (
                "selected nonzero angle must have lower tune32 mean KLD than both controls, "
                "stock delta <= -0.0004, and no domain mean delta versus stock > +0.0005; "
                "paired BCa is reported as a control but need not exclude zero"
            ),
            "conditional_fit_rule": (
                "after tune pass, freeze one candidate and run conditional-fit32 once; "
                "advance only if means are lower than both controls; report paired BCa"
            ),
            "stopping": "one train32 run per nine fixed grid arms; one tune32 run per declared tune arm; no rerolls or replacements",
        },
        "runtime": {
            "topology": "TP4/EP4/DCP4",
            "moe_backend": "humming",
            "activation_dtype": "BF16",
            "kv_cache": "FP8 MLA",
            "mtp": "disabled",
            "cuda_graphs": "disabled/enforce-eager for KLD",
            "placement": "existing post-SwiGLU Humming mid hook",
            "fusion_status": "not fused; native prologue work is gated on KLD win",
        },
        "controls": {
            "rotation_attribution": "angle_pi=0 encoded by the identical full-GPTQ down-only path",
            "product_relevance": "stock ModelOpt NVFP4",
        },
        "claim_boundary": (
            "train/tune are fit-role adaptive evidence; conditional-fit32 is already-open "
            "development evidence; selection, confirmation, final, and reserved confirmation logits remain unopened"
        ),
        "prior": {
            "v6": _record(args.prior_v6_analysis),
            "v7": _record(args.prior_v7_analysis),
        },
        "inputs": {
            "source_index": _record(args.source_index),
            "capture_manifest": _record(args.capture_manifest),
            "calibration_roles": _record(args.calibration_roles),
            "train_roles": _record(args.train_roles),
            "tune_roles": _record(args.tune_roles),
            "conditional_fit_roles": _record(args.conditional_fit_roles),
            "materializer": _record(args.materializer),
            "rotation_builder": _record(args.rotation_builder),
            "candidate_builder": _record(args.candidate_builder),
            "build_finalizer": _record(args.build_finalizer),
            "layer_validator": _record(args.layer_validator),
            "build_runner": _record(args.build_runner),
            "run_kld": _record(args.run_kld),
            "role_eval": _record(args.role_eval),
            "paired_analysis": _record(args.paired_analysis),
            "runtime_manifest": _record(args.runtime_manifest),
            "plan_builder": _record(Path(__file__)),
        },
        "protected_roles_opened": [],
    }
    if args.prior_plan is not None:
        result["amendment"] = {
            "prior_plan": _record(args.prior_plan),
            "reason": args.amendment_reason,
            "changed_decision_fields": [],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "source_index",
        "capture_manifest",
        "calibration_roles",
        "train_roles",
        "tune_roles",
        "conditional_fit_roles",
        "materializer",
        "rotation_builder",
        "candidate_builder",
        "build_finalizer",
        "layer_validator",
        "build_runner",
        "run_kld",
        "role_eval",
        "paired_analysis",
        "runtime_manifest",
        "prior_v6_analysis",
        "prior_v7_analysis",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path)
    parser.add_argument(
        "--amendment-reason",
        default=(
            "pin the complete build and validation toolchain after infrastructure smoke; "
            "no KLD target or grid result was opened"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
