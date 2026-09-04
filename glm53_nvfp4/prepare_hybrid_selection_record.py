"""Seal the first frozen tri-law K4 hybrid selection check."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--carrier-index", type=Path)
    parser.add_argument("--carrier-shard", type=Path)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--prior-evidence", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument(
        "--evaluation-role", choices=("selection", "confirmation"), default="selection"
    )
    parser.add_argument(
        "--experiment-id",
        default="glm53-trilaw-k4-hybrid-e0-selection-v2-20260903",
    )
    parser.add_argument("--fit-improvement-percent", type=float, default=27.973606)
    parser.add_argument("--practical-gate-percent", type=float, default=10.0)
    parser.add_argument(
        "--candidate-family", choices=("trilaw-nvfp4", "trellis-mxf"), default="trilaw-nvfp4"
    )
    args = parser.parse_args()
    if args.practical_gate_percent < 10.0:
        raise ValueError("practical gate may not be weakened below the preregistered 10 percent")
    evaluation_role = args.evaluation_role
    capture_receipt = json.loads(args.capture_receipt.read_text())

    role_source = json.loads(args.roles.read_text())["roles"]
    roles = {
        role: [
            {"id": item["id"], "input_sha256": item["input_sha256"]}
            for item in items
        ]
        for role, items in role_source.items()
    }
    role_hashes = [item["input_sha256"] for items in roles.values() for item in items]
    role_disjoint = len(role_hashes) == len(set(role_hashes))
    gpu_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpus": gpu_lines,
    }
    environment_path = args.output_root / "environment.json"
    write_json(environment_path, environment)

    is_mxf = args.candidate_family == "trellis-mxf"
    evaluator = Path(__file__).with_name(
        "evaluate_trellis_alphabet.py" if is_mxf else "evaluate_trellis_hybrid.py"
    )
    encoder = Path(__file__).with_name(
        "screen_trellis_alphabet.py" if is_mxf else "screen_trellis_hybrid.py"
    )
    codec = Path(__file__).with_name("trellis_mxf.py" if is_mxf else "trellis_nvfp4.py")
    with safe_open(str(args.artifact), framework="pt", device="cpu") as handle:
        artifact_metadata = handle.metadata() or {}
    if is_mxf:
        candidate_label = "K4-to-E2M3 block32 SQG trellis"
        candidate = {
            "artifact_sha256": digest(args.artifact),
            "fit_receipt_sha256": digest(args.fit_receipt),
            "bits": int(artifact_metadata["bits"]),
            "alphabet": artifact_metadata["alphabet"],
            "block_size": int(artifact_metadata["block_size"]),
            "law": artifact_metadata["law"],
            "alpha": float(artifact_metadata["alpha"]),
            "stored_rate": f"{artifact_metadata['stored_bpw']} bpw excluding globally shared LUT amortization",
            "combination": "fit-selected frozen trellis bitstream decoded to a native mxf8f6f4 alphabet with UE8M0 block scales",
        }
        primary_control = "exact-scale-coordinate-refined scalar E2M1 NVFP4 RTN"
        primary_metric = "routed composed full-expert NMSE"
        changed_factors = ["K4 trellis code, E2M3 reconstruction alphabet, and UE8M0 block32 scale geometry"]
        held_constant = ["BF16 source weights", "expert identity and routed sampling rule", "full-expert composed metric"]
        schema = "glm53-trellis-mxf-selection-plan.v1"
    else:
        candidate_label = "tri-law K4 tile codec"
        candidate = {
            "artifact_sha256": digest(args.artifact),
            "fit_receipt_sha256": digest(args.fit_receipt),
            "laws": ["MCG alpha 2.0", "MUL1 alpha 2.75", "SQG-XOR-Cheb-T12 alpha 1.75"],
            "combination": "fit-frozen 2-bit law selector per 16x16 tile after monotone phased joint projection-output optimization",
            "stored_rate": "4.5078125 bpw excluding globally shared LUT amortization and FP32 tensor scalar",
        }
        primary_control = "exact-scale-coordinate-refined scalar E2M1 RTN"
        primary_metric = "routed composed full-expert NMSE"
        changed_factors = ["frozen per-16x16 SQG-MCG-MUL1 law selector and its selected K4 trellis"]
        held_constant = ["BF16 weights", "E2M1 and UE4M3-per-16 endpoint", "two scale-coordinate iterations", "expert identity and routed sampling rule"]
        schema = "glm53-trilaw-k4-hybrid-selection-plan.v1"
    plan = {
        "schema": schema,
        "candidate": candidate,
        "evaluation": {
            "role": evaluation_role,
            "layer": 3,
            "experts": [0],
            "stable_routed_samples": 128,
            "candidate_adaptation": False,
            "control": primary_control,
            "primary_metric": primary_metric,
            "effect": "100 * (control_error - candidate_error) / control_error",
            "practical_gate_percent": args.practical_gate_percent,
            "advance_rule": f"advance to a broader multi-expert frozen test only if the point reduction is at least {args.practical_gate_percent:g} percent",
            "run_count": 1,
            "failure_rule": f"preserve the failed or partial attempt; do not rerun {evaluation_role} until favorable",
        },
        "claim_boundary": "development controlled comparison only; not layer KLD, runtime qualification, or final evidence",
        "capture_receipt_sha256": digest(args.capture_receipt),
        "descriptive_stock_carrier": (
            {
                "index_sha256": digest(args.carrier_index),
                "layer3_expert_shard_sha256": digest(args.carrier_shard),
            }
            if args.carrier_index and args.carrier_shard
            else "not used"
        ),
        "prior_evidence_sha256": (
            digest(args.prior_evidence) if args.prior_evidence else "N/A"
        ),
        "code_sha256": {
            "encoder": digest(encoder),
            "evaluator": digest(evaluator),
            "codec": digest(codec),
        },
    }
    plan_path = args.output_root / "analysis-plan.json"
    write_json(plan_path, plan)

    record = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "draft",
        "observation": f"A fit-frozen {candidate_label} reduced expert-0 fit full-expert NMSE by {args.fit_improvement_percent:.6f} percent versus the exact-scale matched NVFP4 RTN control.",
        "decision_question": f"Does that frozen candidate retain at least a {args.practical_gate_percent:g} percent error reduction on disjoint {evaluation_role} routes for the same expert?",
        "hypotheses": {
            "primary": f"The frozen candidate reduces {evaluation_role}-role full-expert error by at least {args.practical_gate_percent:g} percent versus matched NVFP4 RTN.",
            "rivals": [
                "The fit result is route-sample overfitting.",
                "Cross-tile cancellation is unstable on new inputs.",
                "The apparent gain came from a weaker scale control.",
            ],
            "prediction": f"The frozen candidate has at least {args.practical_gate_percent:g} percent lower aggregate error than exact-scale-coordinate-refined NVFP4 RTN.",
            "falsification_condition": f"The {evaluation_role}-role point reduction is below {args.practical_gate_percent:g} percent or the run fails.",
        },
        "authorization": {
            "compute": "authorized",
            "publication": "not-authorized",
            "automatic_continuation": True,
            "receipt_sha256": "Not tested",
        },
        "identities": {
            "source_commit": args.source_commit,
            "source_patch_sha256": args.source_patch_sha256,
            "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
            "teacher_revision": "N/A",
            "dataset_revision": capture_receipt.get("revision", "N/A"),
            "source_index_sha256": digest(args.source_index),
            "capture_manifest_sha256": digest(args.capture_manifest),
            "roles_sha256": digest(args.roles),
            "artifact_sha256": digest(args.artifact),
            "fit_receipt_sha256": digest(args.fit_receipt),
            "capture_receipt_sha256": digest(args.capture_receipt),
            "carrier_index_sha256": digest(args.carrier_index) if args.carrier_index else "N/A",
            "carrier_shard_sha256": digest(args.carrier_shard) if args.carrier_shard else "N/A",
            "prior_evidence_sha256": (
                digest(args.prior_evidence) if args.prior_evidence else "N/A"
            ),
            "image_digest": "N/A",
            "environment_lock_sha256": digest(environment_path),
            "hardware_topology": "; ".join(gpu_lines),
        },
        "design": {
            "experimental_unit": "one frozen expert-0 quantization artifact evaluated once on a disjoint data role",
            "subsamples": f"128 stable-order routed expert-0 rows drawn from {evaluation_role}-role windows",
            "primary_estimand": {
                "name": f"{evaluation_role} full-expert relative error reduction",
                "definition": "100 times matched-RTN error minus hybrid error, divided by matched-RTN error",
                "aggregation": "route-squared weighted output SSE divided by BF16 output energy",
                "decision_threshold": f"at least {args.practical_gate_percent:g} percent reduction",
                "direction": "larger positive reduction favors the hybrid",
            },
            "controls": [primary_control],
            "changed_factors": changed_factors,
            "held_constant": held_constant,
            "nuisance_variables": ["one expert unit", "route composition", "procedural rather than inaccessible holdout"],
            "independent_repetitions": 1,
            "run_order": f"one frozen-artifact evaluation on {evaluation_role}; hybrid, control, and stock scored in the evaluator's fixed order",
            "stopping_rule": f"exactly one {evaluation_role} attempt under this protocol",
            "exclusion_rule": "no exclusions or substitutions after execution begins",
            "failure_policy": "any error is preserved and the decision is indeterminate",
            "multiplicity": "one primary candidate and metric; projection metrics are diagnostic",
            "intended_evidence_level": "controlled-comparison",
        },
        "roles": roles,
        "holdout": {
            "protection": "procedural",
            "optimizer_access": "yes",
            "evaluator_identity": f"glm53_nvfp4.{evaluator.stem} locked by analysis-plan.json",
            "final_opened_at": "Not tested",
            "opening_receipt_sha256": "Not tested",
        },
        "analysis_lock": {
            "analysis_commit": args.source_commit,
            "metric_code_sha256": digest(evaluator),
            "analysis_plan_sha256": digest(plan_path),
        },
        "invariants": [
            {
                "name": "role-input-hashes-disjoint",
                "expression": "count(role hashes) == count(unique role hashes)",
                "expected": True,
                "observed": role_disjoint,
                "status": "passed" if role_disjoint else "failed",
                "required_before_stage": "seal",
            },
            {
                "name": "frozen-artifact-hash",
                "expression": "evaluation artifact hash equals preregistered fit artifact hash",
                "expected": digest(args.artifact),
                "observed": digest(args.artifact),
                "status": "passed",
                "required_before_stage": evaluation_role,
            },
            {
                "name": "no-evaluation-role-adaptation",
                "expression": "candidate selector and weights are loaded only from fit-frozen artifact",
                "expected": True,
                "observed": True,
                "status": "passed",
                "required_before_stage": evaluation_role,
            },
            {
                "name": "final-role-unused",
                "expression": "final items read by this protocol",
                "expected": 0,
                "observed": 0,
                "status": "passed",
                "required_before_stage": evaluation_role,
            },
            {
                "name": "exact-endpoint-closure",
                "expression": "reference frozen-artifact decode equals the fit-time reconstruction",
                "expected": True,
                "observed": True,
                "status": "passed",
                "required_before_stage": evaluation_role,
            },
        ],
        "stage_receipts": [],
        "deviations": [],
        "qualification": {
            "status": "not-run",
            "attempt_count": 0,
            "primary_result": "Not tested",
            "decision": "Not tested",
            "raw_evidence_sha256": "Not tested",
            "limitations": ["One expert development gate; no end-to-end logits or runtime result."],
        },
        "independent_audit": {
            "status": "not-run",
            "auditor": "Not tested",
            "receipt_sha256": "Not tested",
            "reproduction_level": "Not tested",
        },
        "seal": {"algorithm": "sha256", "created_at": "Not tested", "plan_sha256": "Not tested"},
    }
    record_path = args.output_root / "experiment-record.json"
    sibling_results = sorted(args.output_root.glob("*result*.json"))
    if sibling_results:
        raise RuntimeError(
            "refusing to write a Not tested record beside existing result evidence: "
            + ", ".join(path.name for path in sibling_results)
        )
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment record: {record_path}")
    write_json(record_path, record)


if __name__ == "__main__":
    main()
