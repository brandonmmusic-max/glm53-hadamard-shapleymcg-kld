"""Freeze the exact matched conditional-fit32 execution before serving either arm."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    roles = json.loads(args.roles.read_text())
    if len(roles["roles"]["conditional-fit"]) != 32:
        raise RuntimeError("execution role must contain exactly 32 conditional-fit windows")
    if any(roles["roles"][name] for name in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("execution role exposes a protected/non-conditional partition")

    build_receipt = json.loads(args.full_build.read_text())
    if (
        build_receipt.get("schema")
        not in {
            "glm53-rotation-v6.blocklocal-h16-layer3-full-build.v1",
            "glm53-rotation-v6.blocklocal-h16-layer3-full-build.v2",
        }
        or build_receipt.get("status") != "pass"
        or build_receipt.get("experts") != 288
        or build_receipt.get("protected_roles_opened") != []
    ):
        raise RuntimeError("all-expert candidate build is not closed")
    candidate_receipt_path = args.candidate / "BF16_LAYER_RECEIPT.json"
    candidate_receipt = json.loads(candidate_receipt_path.read_text())
    mount_root = args.model_aux_mount_root.resolve()
    if (
        sha256_file(candidate_receipt_path) != build_receipt.get("candidate_receipt_sha256")
        or candidate_receipt.get("bf16_layers") != [3]
        or candidate_receipt.get("redirected_tensors") != 864
        or Path(candidate_receipt.get("carrier", "")).resolve() != args.stock.resolve()
        or candidate_receipt.get("required_load_format") != "instanttensor"
        or not all(
            Path(item["path"]).resolve().is_relative_to(mount_root)
            for item in candidate_receipt.get("chunks", [])
        )
    ):
        raise RuntimeError("candidate overlay does not close against the all-expert build")

    return {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-cf32-execution.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "run_order": ["stock", "candidate"],
        "run_ids": {
            "stock": "rotation-v6-blocklocal-h16-l3-stock-cf32-v3",
            "candidate": "rotation-v6-blocklocal-h16-l3-candidate-cf32-v3",
        },
        "stopping_rule": "one fresh complete run per arm; no resume, reroll, exclusion, or substitution",
        "decision_rule": (
            "adaptive advance iff equal-window mean candidate KLD is below stock; "
            "paired BCa 95 percent CI is reported but need not exclude zero"
        ),
        "claim_boundary": (
            "already-open conditional-fit32 is adaptive evidence only; this BF16 effective-weight "
            "overlay does not establish native compact-runtime closure"
        ),
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "runtime": {
            "image": args.runtime_image,
            "image_id": args.image_id,
            "load_format": "instanttensor",
            "topology": "TP4/EP4/DCP4",
            "moe_backend": "humming",
            "activation_dtype": "BF16",
            "kv_cache": "FP8 MLA",
            "cuda_graphs": "disabled/enforce-eager",
            "mtp": "disabled",
            "rotation_runtime": "identity; inverse rotation is baked into BF16 effective weights",
            "model_aux_mount_root": str(args.model_aux_mount_root.resolve()),
        },
        "models": {
            "stock": str(args.stock.resolve()),
            "candidate": str(args.candidate.resolve()),
        },
        "outputs": {
            "run_root": str(args.run_root.resolve()),
            "execution_receipt": str(args.execution_receipt.resolve()),
            "paired_analysis": str(args.paired_analysis.resolve()),
        },
        "inputs": {
            "hypothesis_plan": _file(args.hypothesis_plan),
            "full_build": _file(args.full_build),
            "candidate_receipt": _file(candidate_receipt_path),
            "candidate_index": _file(args.candidate / "model.safetensors.index.json"),
            "candidate_config": _file(args.candidate / "config.json"),
            "stock_index": _file(args.stock / "model.safetensors.index.json"),
            "stock_config": _file(args.stock / "config.json"),
            "roles": _file(args.roles),
            "runtime_manifest": _file(args.runtime_manifest),
            "runner": _file(args.runner),
            "run_kld": _file(args.run_kld),
            "role_eval": _file(args.role_eval),
            "paired_analysis_code": _file(args.paired_analysis_code),
        },
        "protected_roles_opened": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis-plan", type=Path, required=True)
    parser.add_argument("--full-build", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model-aux-mount-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--run-kld", type=Path, required=True)
    parser.add_argument("--role-eval", type=Path, required=True)
    parser.add_argument("--paired-analysis-code", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--paired-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
