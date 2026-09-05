"""Freeze the V8 train32 physical-runtime KLD execution."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


RUN_ORDER = (
    "stock",
    "zero",
    "m003125",
    "p003125",
    "m00625",
    "p00625",
    "m0125",
    "p0125",
    "m0250",
    "p0250",
)


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
    parser.add_argument("--search-plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
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
    search_plan = json.loads(args.search_plan.read_text())
    search_sha = sha256_file(args.search_plan)
    if (
        search_plan.get("decision_before_result") is not True
        or search_plan.get("protected_roles_opened") != []
        or [item["id"] for item in search_plan["search"]["coarse_grid"]]
        != [arm for arm in ("m0250", "m0125", "m00625", "m003125", "zero", "p003125", "p00625", "p0125", "p0250")]
    ):
        raise RuntimeError("invalid V8 search plan")
    roles = json.loads(args.roles.read_text())
    if roles.get("counts", {}).get("fit") != 32:
        raise RuntimeError("execution requires the sealed train32 role")
    candidates = {}
    for arm in RUN_ORDER[1:]:
        arm_root = args.root / "arms" / arm
        build_path = arm_root / "FULL_BUILD.json"
        build = json.loads(build_path.read_text())
        if (
            build.get("status") != "pass"
            or build.get("plan_sha256") != search_sha
            or build.get("angle_pi")
            != next(row["angle_pi"] for row in search_plan["search"]["coarse_grid"] if row["id"] == arm)
            or build.get("physical_payload_bpw") > 4.50001
        ):
            raise RuntimeError(f"candidate build is not sealed: {arm}")
        candidates[arm] = {
            "model": str((arm_root / "candidate").resolve()),
            "rotation": _record(args.root / "rotations" / f"{arm}.safetensors"),
            "build": _record(build_path),
            "config": _record(arm_root / "candidate" / "config.json"),
            "index": _record(arm_root / "candidate" / "model.safetensors.index.json"),
        }
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    run_ids = {
        arm: f"rotation-v8-shared-mid-butterfly-l3-{arm}-train32-v1"
        for arm in RUN_ORDER
    }
    result = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-train32-execution.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "search_plan": _record(args.search_plan),
        "role": "fit/train32 adaptive search",
        "run_order": list(RUN_ORDER),
        "grid_arms": list(RUN_ORDER[1:]),
        "run_ids": run_ids,
        "models": {
            "stock": str(args.stock.resolve()),
            **{arm: candidates[arm]["model"] for arm in RUN_ORDER[1:]},
        },
        "candidates": candidates,
        "inputs": {
            "roles": _record(args.roles),
            "stock_config": _record(args.stock / "config.json"),
            "stock_index": _record(args.stock / "model.safetensors.index.json"),
            "run_kld": _record(args.run_kld),
            "role_eval": _record(args.role_eval),
            "runner": _record(args.runner),
            "analyzer": _record(args.analyzer),
            "runtime_manifest": _record(args.runtime_manifest),
            "analysis_python": _record(Path("/usr/bin/python3.12")),
        },
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
        "selection_rule": search_plan["search"]["selection_rule"],
        "stopping_rule": search_plan["search"]["stopping"],
        "outputs": {
            "run_root": "/media/brandonmusic/klcstore/bmxfp4-glm53/kld-v3",
            "analysis": str((args.root / "results" / "train32-analysis.json").resolve()),
            "execution_receipt": str((args.root / "results" / "train32-execution.json").resolve()),
        },
        "protected_roles_opened": [],
        "claim_boundary": "adaptive fit-role evidence only; selection, confirmation, final, and reserved confirmation logits remain unopened",
        "ldlq": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "image_id": image_id}, sort_keys=True))


if __name__ == "__main__":
    main()
