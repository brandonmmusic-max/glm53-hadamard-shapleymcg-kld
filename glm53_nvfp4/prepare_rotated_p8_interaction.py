"""Freeze the selected shared butterfly in one TrellisMX-P8 interaction."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


EXPECTED_DOMAINS = {
    "axis1_general": 8,
    "axis2_legal": 8,
    "axis3_code_agentic": 8,
    "axis4_reasoning_termination": 8,
}


def _file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-p8-design", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--train-analysis", type=Path, required=True)
    parser.add_argument("--tune-analysis", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--prior-preflight-failure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 3 <= args.layer <= 44:
        raise ValueError("layer must be 3..44")

    base = json.loads(args.base_p8_design.read_text())
    if (
        base.get("schema") != "glm53-p8.direct-kld-native-shapley-design.v2"
        or base.get("ldlq") is not False
        or base.get("encoder_contract", {}).get("ldlq") is not False
    ):
        raise RuntimeError("base design is not the frozen no-LDLQ P8 v2 design")
    roles = json.loads(args.roles.read_text())
    rows = roles.get("roles", {}).get("fit", [])
    counts = {domain: 0 for domain in EXPECTED_DOMAINS}
    for row in rows:
        if row.get("domain") not in counts:
            raise RuntimeError("tune role contains an unexpected domain")
        counts[row["domain"]] += 1
    if len(rows) != 32 or counts != EXPECTED_DOMAINS:
        raise RuntimeError(f"interaction role is not balanced tune32: {counts}")

    train = json.loads(args.train_analysis.read_text())
    tune = json.loads(args.tune_analysis.read_text())
    if (
        train.get("schema")
        != "glm53-rotation-v8.shared-mid-butterfly-train32-analysis.v1"
        or train.get("selected_arm") != "p00625"
        or tune.get("schema")
        != "glm53-rotation-v8.shared-mid-butterfly-gate-analysis.v1"
        or tune.get("stage") != "tune32"
        or tune.get("selected_arm") != "p00625"
        or tune.get("role") != "fit/tune32"
    ):
        raise RuntimeError("rotation analyses do not identify the frozen p00625 arm")
    with safe_open(args.rotation, framework="pt", device="cpu") as src:
        metadata = src.metadata() or {}
        if (
            metadata.get("schema") != "glm53-rotation-v8.shared-butterfly16.v1"
            or metadata.get("scope") != "mid-only"
            or metadata.get("layer") != str(args.layer)
            or float(metadata.get("angle_pi", "nan")) != 0.0625
            or set(src.keys()) != {f"layer_{args.layer:03d}_mid"}
        ):
            raise RuntimeError("rotation is not the selected canonical p00625 payload")

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip():
        raise RuntimeError("freeze the integration from a clean committed worktree")
    inputs = {
        "encoder": _file(repo / "glm53_nvfp4/quantize_rotated_hessian_trellis_layer.py"),
        "block_rotation": _file(repo / "glm53_nvfp4/block_rotation.py"),
        "trellis_codec": _file(repo / "glm53_nvfp4/trellis_mxf.py"),
        "run_kld": _file(repo / "scripts/run_kld_v3.sh"),
        "runtime_patch": _file(repo / "runtime_patch/sitecustomize.py"),
        "train_analysis": _file(args.train_analysis),
        "tune_analysis": _file(args.tune_analysis),
        "roles": _file(args.roles),
    }
    prior_failure = None
    if args.prior_preflight_failure is not None:
        logs = sorted(args.prior_preflight_failure.glob("logs/chunk-*.log"))
        if len(logs) != 4:
            raise RuntimeError("prior preflight failure must contain four chunk logs")
        prior_failure = {
            "root": str(args.prior_preflight_failure.resolve()),
            "logs": [_file(path) for path in logs],
            "classification": "pre-output deterministic dtype mismatch in diagnostic metric; no codec, dense chunk, receipt, or KLD result written",
        }
    payload = {
        "schema": "glm53-trellismx-p8.shared-mid-butterfly-design.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "decision_record": "DECISIONS_GLM53.md#21",
        "git_commit": commit,
        "layers": [args.layer],
        "rotation": {
            "id": "p00625",
            "angle_pi": 0.0625,
            **_file(args.rotation),
            "selection": (
                "p00625 fixed by the completed layer-3 fit analysis before the "
                "three-layer P8 encoder test; no per-layer angle search"
            ),
            "tune_decision_control": tune["decision"],
        },
        "base_p8_design": _file(args.base_p8_design),
        "encoder": {
            "gate_up": "K4 procedural-MCG E4M3 UE8M0-K32, full-Hessian GPTQ feedback",
            "down": "encode W2 @ R with K4 procedural-MCG E4M3 UE8M0-K32, full-Hessian GPTQ feedback",
            "activation_order": "static within each native 16-column trellis group",
            "scale_refits": 2,
            "rotation_arithmetic": "post-SwiGLU BF16 input; four FP32 Givens stages; final BF16 round; E4M3 UE8M0-K32 quantization",
            "stored_weight_bpw": 4.25,
            "rotation_table_bytes": 0,
            "ldlq": False,
        },
        "pseudoquant_gate": {
            "role": "fit/tune32",
            "windows": 32,
            "domain_counts": EXPECTED_DOMAINS,
            "arms": ["fresh-stock-nvfp4", "unrotated-p8", "rotated-p8-p00625"],
            "run_order": ["fresh-stock-nvfp4", "unrotated-p8", "rotated-p8-p00625"],
            "decision_rule": "rotated P8 mean KLD below unrotated P8 and no greater than stock",
            "paired_bca": "reported as nonblocking development control",
            "stopping": "one run per arm; no exclusions, substitutions, rerolls, or replacement angles",
        },
        "device_closure_gate": {
            "comparison": "fused rotated P8 kernel minus exact staged rotated P8 pseudoquant",
            "decision_rule": "absolute paired mean delta <= 0.0014 and paired BCa 95 percent interval contains zero",
            "determinism": "five bitwise-identical runs",
        },
        "advance": "only a pseudoquant pass advances the selected rotation into the fused device prologue and all-layer build",
        "isa_cost_statement": "P8 uses mxf8f6f4 at twice NVFP4 MMA issue count; it is the quality product, not the P4 speed product",
        "protected_roles_opened": [],
        "protected_boundary": "selection, confirmation, final, and 28 reserved confirmation logits remain unopened",
        "ldlq": False,
        "inputs": inputs,
        "prior_preflight_failure": prior_failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
