"""Freeze a V8 three-arm tune32 or conditional-fit32 KLD gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune32", "conditional-fit32"), required=True)
    parser.add_argument("--search-plan", type=Path, required=True)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--teacher-subset-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--selected-arm", required=True)
    parser.add_argument("--run-kld", type=Path, required=True)
    parser.add_argument("--role-eval", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    search = json.loads(args.search_plan.read_text())
    prior = json.loads(args.prior_analysis.read_text())
    if prior.get("selected_arm") != args.selected_arm:
        raise RuntimeError("selected arm differs from prior analysis")
    if args.stage == "conditional-fit32" and prior.get("decision") != "pass":
        raise RuntimeError("conditional-fit requires a passing tune analysis")
    allowed = {row["id"] for row in search["search"]["coarse_grid"]}
    if args.selected_arm == "zero" or args.selected_arm not in allowed:
        raise RuntimeError("gate requires one selected nonzero grid arm")
    roles = json.loads(args.roles.read_text())
    runtime_role = "fit" if args.stage == "tune32" else "conditional-fit"
    if len(roles["roles"][runtime_role]) != 32:
        raise RuntimeError("gate roles must contain exactly 32 windows")
    zero_root = args.root / "arms/zero"
    candidate_root = args.root / "arms" / args.selected_arm
    for arm_root in (zero_root, candidate_root):
        build = json.loads((arm_root / "FULL_BUILD.json").read_text())
        if build.get("status") != "pass" or build.get("physical_payload_bpw") > 4.50001:
            raise RuntimeError(f"invalid physical candidate: {arm_root}")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    run_ids = {
        arm: f"rotation-v8-shared-mid-butterfly-l3-{args.selected_arm}-{arm}-{args.stage}-v1"
        for arm in ("stock", "zero", "candidate")
    }
    threshold = {
        "maximum_stock_delta": -0.0004 if args.stage == "tune32" else 0.0,
        "maximum_domain_delta": 0.0005 if args.stage == "tune32" else 1.0e9,
        "bca_ci_blocks": False,
    }
    result = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-gate-execution.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "stage": args.stage,
        "runtime_role": runtime_role,
        "analysis_role": f"{runtime_role}/{args.stage}",
        "selected_arm": args.selected_arm,
        "run_order": ["stock", "zero", "candidate"],
        "run_ids": run_ids,
        "models": {
            "stock": str(args.stock.resolve()),
            "zero": str((zero_root / "candidate").resolve()),
            "candidate": str((candidate_root / "candidate").resolve()),
        },
        "rotations": {
            "zero": _record(args.root / "rotations/zero.safetensors"),
            "candidate": _record(args.root / "rotations" / f"{args.selected_arm}.safetensors"),
        },
        "physical_builds": {
            "zero": _record(zero_root / "FULL_BUILD.json"),
            "candidate": _record(candidate_root / "FULL_BUILD.json"),
        },
        "inputs": {
            "search_plan": _record(args.search_plan),
            "prior_analysis": _record(args.prior_analysis),
            "roles": _record(args.roles),
            "teacher_subset_receipt": _record(args.teacher_subset_receipt),
            "stock_config": _record(args.stock / "config.json"),
            "stock_index": _record(args.stock / "model.safetensors.index.json"),
            "run_kld": _record(args.run_kld),
            "role_eval": _record(args.role_eval),
            "runner": _record(args.runner),
            "analyzer": _record(args.analyzer),
            "runtime_manifest": _record(args.runtime_manifest),
            "preparer": _record(Path(__file__)),
            "analysis_python": _record(Path("/usr/bin/python3.12")),
        },
        "threshold": threshold,
        "runtime": {
            "image": args.image,
            "image_id": image_id,
            "topology": "TP4/EP4/DCP4",
            "moe_backend": "humming",
            "load_format": "instanttensor",
            "activation_dtype": "BF16",
            "kv_cache": "FP8 MLA",
            "cuda_graphs": "disabled/enforce-eager",
            "mtp": "disabled",
            "rotation_scope": "mid-only",
            "rotation_placement": "humming-inner post-SwiGLU hook; not fused",
            "model_aux_mount_root": str(args.root.resolve()),
        },
        "outputs": {
            "run_root": "/media/brandonmusic/klcstore/bmxfp4-glm53/kld-v3",
            "analysis": str((args.root / "results" / f"{args.stage}-analysis-v1.json").resolve()),
            "execution_receipt": str((args.root / "results" / f"{args.stage}-execution-v1.json").resolve()),
        },
        "protected_roles_opened": [],
        "claim_boundary": "adaptive development evidence; protected roles remain unopened",
        "ldlq": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
