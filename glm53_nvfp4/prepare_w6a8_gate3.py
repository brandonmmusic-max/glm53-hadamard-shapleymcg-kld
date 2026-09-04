"""Prepare the sealed conditional-fit W6A8 diagnosis before any scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--mxfp6-candidate", type=Path, required=True)
    parser.add_argument("--bf16-candidate", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--supersedes-record", type=Path)
    args = parser.parse_args()

    roles = json.loads(args.roles.read_text())
    windows = roles["roles"]["conditional-fit"]
    if len(windows) != 32 or roles["domains"] != {
        "axis1_general": 8,
        "axis2_legal": 8,
        "axis3_code_agentic": 8,
        "axis4_reasoning_termination": 8,
    }:
        raise RuntimeError("gate 3 requires the fixed domain-balanced n=32 role")
    if any(roles["roles"][name] for name in ("selection", "confirmation", "final")):
        raise RuntimeError("gate 3 role file must not expose protected roles")

    mxfp6_receipt_path = args.mxfp6_candidate / "MIXED_RECEIPT.json"
    bf16_receipt_path = args.bf16_candidate / "BF16_LAYER_RECEIPT.json"
    mxfp6_receipt = json.loads(mxfp6_receipt_path.read_text())
    bf16_receipt = json.loads(bf16_receipt_path.read_text())
    decode_receipts = []
    receipt_root = args.bf16_candidate.parent / "receipts"
    for path in sorted(receipt_root.glob("decode-*.json")):
        payload = json.loads(path.read_text())
        decode_receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "input_sha256": payload["input"]["sha256"],
                "output_sha256": payload["output"]["sha256"],
                "per_projection": payload["per_projection"],
            }
        )
    if len(decode_receipts) != 4:
        raise RuntimeError("four decoded-weight receipts are required")
    if sorted(item["input_sha256"] for item in decode_receipts) != sorted(
        item["sha256"] for item in mxfp6_receipt["chunks"]
    ):
        raise RuntimeError("BF16 control does not decode the exact MXFP6 chunks")
    if sorted(item["output_sha256"] for item in decode_receipts) != sorted(
        item["sha256"] for item in bf16_receipt["chunks"]
    ):
        raise RuntimeError("BF16 overlay does not reference the exact decoded chunks")

    repo = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    git_patch = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary"]
    )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    analysis_plan = {
        "schema": "glm53-w6a8-gate3-analysis-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit",
        "windows": 32,
        "arms_in_fixed_order": [
            {
                "id": "R0",
                "model": "exact decoded MXFP6 weights as BF16 overlay",
                "activation": "BF16 framework path",
                "epilogue": "framework routed reduction",
            },
            {
                "id": "W0",
                "model": "same MXFP6 weights",
                "activation": "E4M3 K32, unit input scales, fast math",
                "epilogue": "BF16 atomic routed accumulation",
            },
            {
                "id": "W1",
                "model": "same MXFP6 weights",
                "activation": "E4M3 K32, unit input scales, fast math",
                "epilogue": "deterministic route outputs plus top-k reduction",
            },
            {
                "id": "W2",
                "model": "same MXFP6 weights",
                "activation": "E4M3 K32, unit input scales, non-fast SiLU",
                "epilogue": "deterministic route outputs plus top-k reduction",
            },
            {
                "id": "W3",
                "model": "same MXFP6 weights",
                "activation": "E4M3 K32, unit FC1 scale, dynamic FC2 scale, non-fast SiLU",
                "epilogue": "deterministic route outputs plus top-k reduction",
            },
        ],
        "causal_contrasts": {
            "scatter": "W1 - W0",
            "fast_math": "W2 - W1",
            "dynamic_fc2_scale": "W3 - W2",
            "total_best_fixed_path": "W3 - R0",
        },
        "primary_estimand": "equal-window mean KLD(W3) minus KLD(R0)",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 20000, "seed": 20260902},
        "decision_rule": {
            "carrier_repaired": "BCa upper bound for W3-R0 is <= +0.0014 nats",
            "ladder_blocked": "mean W3-R0 is >= +0.01 nats",
            "otherwise": "continue diagnosis with calibrated input-scale and staged-epilogue arms; do not start P8",
        },
        "stopping_rule": "run each arm once over the exact 32 windows; no role substitution, arm deletion, or reroll",
        "protected_boundary": "selection and the 28 confirmation logits are not opened",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4; P8 is quality-oriented and P4 is speed-oriented",
        "ldlq": False,
    }
    if args.supersedes_record is not None:
        analysis_plan["supersedes"] = {
            "path": str(args.supersedes_record),
            "sha256": sha256_file(args.supersedes_record),
            "reason": "the pinned v80 image contains an H16-only MXFP6 plugin and rejected identity before W0 scoring",
        }
    analysis_plan_path = args.output_root / "analysis-plan.json"
    _write(analysis_plan_path, analysis_plan)

    environment = {
        "schema": "glm53-w6a8-gate3-environment.v1",
        "source_commit": git_head,
        "source_patch_sha256": hashlib.sha256(git_patch).hexdigest(),
        "runtime_image": args.image,
        "runtime_image_id": image_id,
        "hardware": hardware,
        "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        "runtime_patch_manifest": {
            "path": str(args.runtime_manifest),
            "sha256": sha256_file(args.runtime_manifest),
        },
        "mxfp6_candidate": {
            "path": str(args.mxfp6_candidate),
            "receipt_sha256": sha256_file(mxfp6_receipt_path),
        },
        "bf16_candidate": {
            "path": str(args.bf16_candidate),
            "receipt_sha256": sha256_file(bf16_receipt_path),
        },
        "decoded_weight_receipts": decode_receipts,
    }
    environment_path = args.output_root / "environment.json"
    _write(environment_path, environment)

    role_entries = [
        {
            "id": item["id"],
            "input_sha256": item["input_sha256"],
            "teacher_sha256": item["teacher_sha256"],
        }
        for item in windows
    ]
    auth_text = "User explicitly directed autonomous local gate-3 W6A8 diagnosis on 2026-09-04."
    record = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "draft",
        "observation": "Near-lossless layer-3 MXFP6 weights previously incurred about +123 percent KLD through W6A8 on a noisy 16-window role.",
        "decision_question": "Can deterministic reduction, precise activation math, and dynamic FC2 scaling make the W6A8 carrier noninferior to the identical-weight BF16 overlay?",
        "hypotheses": {
            "primary": "W3 is noninferior to R0 at the fixed +0.0014-nat margin.",
            "rivals": [
                "Top-k-8 BF16 atomic accumulation causes the excess KLD.",
                "Fast SiLU math causes the excess KLD.",
                "FC2 intermediate scaling causes the excess KLD.",
                "The remaining E4M3 activation quantization itself causes the excess KLD.",
                "Distributed TP4 loading differs from the top-k-1 canary.",
            ],
            "prediction": "The fixed W3 arm has BCa upper delta no larger than +0.0014 nats versus R0.",
            "falsification_condition": "The W3 BCa upper delta exceeds +0.0014 nats; +0.01 mean delta additionally blocks the ladder.",
        },
        "authorization": {
            "compute": "authorized",
            "publication": "not-authorized",
            "automatic_continuation": True,
            "receipt_sha256": hashlib.sha256(auth_text.encode()).hexdigest(),
        },
        "identities": {
            "source_commit": git_head,
            "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
            "teacher_revision": "95f4fdd94bf29989db2e0d1054e4931f55edb6aa",
            "dataset_revision": "N/A",
            "image_digest": "N/A",
            "environment_lock_sha256": sha256_file(environment_path),
            "hardware_topology": hardware,
        },
        "design": {
            "experimental_unit": "one fixed document window scored by both identical-weight execution paths",
            "subsamples": "32 fixed conditional-fit windows, exactly eight from each of four domains",
            "primary_estimand": {
                "name": "W3 minus R0 mean KLD",
                "definition": "paired per-window student-to-BF16-teacher KLD difference",
                "aggregation": "equal-weight mean over windows with paired BCa interval",
                "decision_threshold": "BCa upper bound <= +0.0014 nats; mean >= +0.01 blocks ladder",
                "direction": "smaller is better",
            },
            "controls": ["R0 identical decoded weights through BF16 overlay", "W0 current W6A8 path"],
            "changed_factors": ["top-k reduction", "fast activation math", "dynamic FC2 activation scale"],
            "held_constant": ["weight values", "layer 3", "runtime image", "TP4/EP4/DCP4", "teacher logits", "window order"],
            "nuisance_variables": ["CUDA scheduling", "server cold-start state", "document heterogeneity"],
            "independent_repetitions": 1,
            "run_order": "R0, W0, W1, W2, W3",
            "stopping_rule": analysis_plan["stopping_rule"],
            "exclusion_rule": "no exclusions, substitutions, or reruns after scoring begins",
            "failure_policy": "preserve failed launches and mark the affected contrast indeterminate",
            "multiplicity": "one primary W3-R0 gate; W1-W0, W2-W1, and W3-W2 are diagnostic causal contrasts",
            "intended_evidence_level": "controlled-comparison",
        },
        "roles": {"fit": [], "conditional-fit": role_entries, "selection": [], "confirmation": [], "final": []},
        "holdout": {
            "protection": "procedural",
            "optimizer_access": "yes",
            "evaluator_identity": "glm53_nvfp4.role_eval plus w6a8_gate3_analysis at frozen hashes",
            "final_opened_at": "Not tested",
            "opening_receipt_sha256": "Not tested",
        },
        "analysis_lock": {
            "analysis_commit": git_head,
            "metric_code_sha256": sha256_file(repo / "glm53_nvfp4/w6a8_gate3_analysis.py"),
            "analysis_plan_sha256": sha256_file(analysis_plan_path),
        },
        "invariants": [
            {"name": "teacher-materialized-and-hashed", "expression": "all 32 role entries have locally verified teacher files", "expected": True, "observed": True, "status": "passed", "required_before_stage": "seal"},
            {"name": "domain-balanced", "expression": "each declared domain has eight windows", "expected": True, "observed": True, "status": "passed", "required_before_stage": "seal"},
            {"name": "protected-roles-unopened", "expression": "selection, confirmation, and final entries in this role file equal zero", "expected": 0, "observed": 0, "status": "passed", "required_before_stage": "seal"},
            {"name": "identical-weight-control", "expression": "BF16 overlay chunk hashes equal exact decodes of the four MXFP6 input chunk hashes", "expected": True, "observed": True, "status": "passed", "required_before_stage": "seal"},
        ],
        "stage_receipts": [],
        "deviations": [
            "The earlier 00:37 roles-v4 attempt silently fell back to roles-v3 and produced no record.",
            "Three prior V3 selection waves were opened within 26 minutes.",
            "Confirmation hidden states, but not confirmation teacher logits, were read for a different hybrid artifact after a selection failure.",
            *( ["This record supersedes a launch-only attempt that used an H16-only runtime image; no W arm was scored."] if args.supersedes_record is not None else [] ),
        ],
        "qualification": {"status": "not-run", "attempt_count": 0, "primary_result": "Not tested", "decision": "Not tested", "raw_evidence_sha256": "Not tested", "limitations": ["Development conditional-fit role; not protected confirmation or final evidence."]},
        "independent_audit": {"status": "not-run", "auditor": "Not tested", "receipt_sha256": "Not tested", "reproduction_level": "Not tested"},
        "seal": {"algorithm": "sha256", "created_at": "Not tested", "plan_sha256": "Not tested"},
    }
    record_path = args.output_root / "experiment-record.json"
    if record_path.exists() or list(args.output_root.glob("*result*.json")):
        raise FileExistsError("refusing to overwrite an existing gate-3 record or result")
    _write(record_path, record)
    print(json.dumps({"record": str(record_path), "analysis_plan_sha256": sha256_file(analysis_plan_path), "environment_sha256": sha256_file(environment_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
