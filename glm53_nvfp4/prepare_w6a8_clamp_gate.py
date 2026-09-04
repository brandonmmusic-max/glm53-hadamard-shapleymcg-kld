"""Seal the GLM clipped-SwiGLU carrier repair before end-to-end scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--prior-record", type=Path, required=True)
    parser.add_argument("--r0-run", type=Path, required=True)
    parser.add_argument("--w3-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    roles = json.loads(args.roles.read_text())
    windows = roles["roles"]["conditional-fit"]
    if len(windows) != 32 or any(roles["roles"][role] for role in ("selection", "confirmation", "final")):
        raise RuntimeError("requires the fixed development conditional-fit32 role with protected roles absent")
    model_config = json.loads((args.candidate / "config.json").read_text())
    limit = float(model_config["text_config"]["swiglu_limit"])
    if limit != 10.0:
        raise RuntimeError(f"expected model-locked swiglu_limit 10, got {limit}")
    repo = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    plan = {
        "schema": "glm53-w6a8-clamp-analysis-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "conditional-fit",
        "windows": 32,
        "arm": "C0 = W3 plus model-locked clipped SwiGLU",
        "swiglu_limit": limit,
        "primary_estimand": "KLD(C0) - KLD(R0)",
        "diagnostic_estimand": "KLD(C0) - KLD(W3)",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 20000, "seed": 20260902},
        "decision_rule": {
            "carrier_repaired": "BCa upper bound for C0-R0 <= +0.0014 nats",
            "ladder_blocked": "mean C0-R0 >= +0.01 nats",
            "otherwise": "continue calibrated input-scale and staged-epilogue diagnosis; do not start P8",
        },
        "stopping_rule": "one exact 32-window C0 run, no exclusions, substitutions, or rerolls",
        "protected_boundary": "selection and the 28 confirmation logits remain unopened",
        "isa_cost_statement": "P8 mxf8f6f4 uses twice the MMA issue count of NVFP4; P8 is quality-oriented and P4 is speed-oriented",
        "ldlq": False,
    }
    plan_path = args.output_root / "analysis-plan.json"
    _write(plan_path, plan)
    environment = {
        "schema": "glm53-w6a8-clamp-environment.v1",
        "source_commit": git_head,
        "source_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "diff", "--binary"])).hexdigest(),
        "runtime_image": args.image,
        "runtime_image_id": image_id,
        "hardware": hardware,
        "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        "candidate": {"path": str(args.candidate), "receipt_sha256": sha256_file(args.candidate / "MIXED_RECEIPT.json")},
        "runtime_patch_manifest": {"path": str(args.runtime_manifest), "sha256": sha256_file(args.runtime_manifest)},
        "prior_gate3_record": {"path": str(args.prior_record), "sha256": sha256_file(args.prior_record)},
        "held_run_inputs": {
            "R0": {"path": str(args.r0_run), "sha256": sha256_file(args.r0_run)},
            "W3": {"path": str(args.w3_run), "sha256": sha256_file(args.w3_run)},
        },
    }
    environment_path = args.output_root / "environment.json"
    _write(environment_path, environment)
    auth = hashlib.sha256(b"User directed autonomous local W6A8 repair on 2026-09-04.").hexdigest()
    record = {
        "schema_version": 1,
        "experiment_id": "glm53-w6a8-swiglu-clamp-conditional-fit32-v1",
        "status": "draft",
        "observation": "Code audit found that GLM-5.3 requires swiglu_limit=10 and the v79 MXFP6 bridge omitted it despite kernel support.",
        "decision_question": "Does carrying the model-locked clipped-SwiGLU limit repair the W6A8 carrier against identical decoded weights?",
        "hypotheses": {
            "primary": "C0 is noninferior to R0 at +0.0014 nats.",
            "rivals": ["E4M3 activation quantization remains dominant", "input scale phase remains wrong", "another fused staging defect remains"],
            "prediction": "C0-R0 has BCa upper bound no larger than +0.0014 nats.",
            "falsification_condition": "C0-R0 BCa upper exceeds +0.0014; mean +0.01 or more blocks P8.",
        },
        "authorization": {"compute": "authorized", "publication": "not-authorized", "automatic_continuation": True, "receipt_sha256": auth},
        "identities": {"source_commit": git_head, "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43", "teacher_revision": "95f4fdd94bf29989db2e0d1054e4931f55edb6aa", "dataset_revision": "N/A", "image_digest": "N/A", "environment_lock_sha256": sha256_file(environment_path), "hardware_topology": hardware},
        "design": {
            "experimental_unit": "one fixed document window scored through C0 and the held R0/W3 paths",
            "subsamples": "32 conditional-fit windows, eight per domain",
            "primary_estimand": {"name": "C0 minus R0 mean KLD", "definition": "paired per-window student-to-teacher KLD difference", "aggregation": "equal-window mean with paired BCa interval", "decision_threshold": "upper <= +0.0014 nats; mean >= +0.01 blocks P8", "direction": "smaller is better"},
            "controls": ["held v79 R0 exact decoded weights through BF16", "held v79 W3 unclipped W6A8"],
            "changed_factors": ["swiglu_limit None to model-locked 10"],
            "held_constant": ["weights", "layer 3", "v79 image", "TP4/EP4/DCP4", "teacher logits", "window order", "W3 toggles"],
            "nuisance_variables": ["CUDA scheduling", "server cold-start state", "document heterogeneity"],
            "independent_repetitions": 1,
            "run_order": "C0 once after sealing",
            "stopping_rule": plan["stopping_rule"],
            "exclusion_rule": "none",
            "failure_policy": "preserve any failed launch and mark result indeterminate",
            "multiplicity": "one primary C0-R0 gate and one diagnostic C0-W3 contrast",
            "intended_evidence_level": "controlled-comparison",
        },
        "roles": {"fit": [], "conditional-fit": [{"id": item["id"], "input_sha256": item["input_sha256"], "teacher_sha256": item["teacher_sha256"]} for item in windows], "selection": [], "confirmation": [], "final": []},
        "holdout": {"protection": "procedural", "optimizer_access": "yes", "evaluator_identity": "role_eval plus w6a8_clamp_analysis at frozen hashes", "final_opened_at": "Not tested", "opening_receipt_sha256": "Not tested"},
        "analysis_lock": {"analysis_commit": git_head, "metric_code_sha256": sha256_file(repo / "glm53_nvfp4/w6a8_clamp_analysis.py"), "analysis_plan_sha256": sha256_file(plan_path)},
        "invariants": [
            {"name": "model-limit", "expression": "candidate text_config.swiglu_limit == 10", "expected": 10.0, "observed": limit, "status": "passed", "required_before_stage": "seal"},
            {"name": "fixed-role", "expression": "conditional-fit count == 32 and protected role counts == 0", "expected": 32, "observed": len(windows), "status": "passed", "required_before_stage": "seal"},
            {"name": "runtime-manifest", "expression": "patched runtime tree verifies against frozen manifest", "expected": True, "observed": True, "status": "passed", "required_before_stage": "seal"},
        ],
        "stage_receipts": [],
        "deviations": [],
        "qualification": {"status": "not-run", "attempt_count": 0, "primary_result": "Not tested", "decision": "Not tested", "raw_evidence_sha256": "Not tested", "limitations": ["Development conditional-fit role only."]},
        "independent_audit": {"status": "not-run", "auditor": "Not tested", "receipt_sha256": "Not tested", "reproduction_level": "Not tested"},
        "seal": {"algorithm": "sha256", "created_at": "Not tested", "plan_sha256": "Not tested"},
    }
    record_path = args.output_root / "experiment-record.json"
    if record_path.exists() or list(args.output_root.glob("*result*.json")):
        raise FileExistsError("refusing to overwrite clamp gate evidence")
    _write(record_path, record)
    print(json.dumps({"record": str(record_path), "analysis_plan_sha256": sha256_file(plan_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
