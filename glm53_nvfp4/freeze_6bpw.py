"""Fail-closed freeze for the Shapley allocation and its confirmation controls."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .shard_index import sha256_file


EXPECTED_MXFP6_LAYERS = 35
EXPECTED_NVFP4_LAYERS = 7
MAX_BPW = 6.0
RUNTIME_IMAGE = "klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v80-mxfp6-h16-all"
RUNTIME_IMAGE_ID = "sha256:8e6856039c449aa202f113c3834af09e4e1dcb4c9c9b7e7eb009608f3772627a"


def _file(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git(root: Path) -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"repository is dirty: {root}")
    return {"root": str(root.resolve()), "commit": head, "clean": True}


def _model(path: Path, *, mixed: bool) -> dict:
    result = {
        "root": str(path.resolve()),
        "index": _file(path / "model.safetensors.index.json"),
        "config": _file(path / "config.json"),
    }
    if mixed:
        result["mixed_receipt"] = _file(path / "MIXED_RECEIPT.json")
    if (path / "OVERLAY.json").is_file():
        overlay = json.loads((path / "OVERLAY.json").read_text())
        if overlay.get("required_load_format") != "instanttensor":
            raise RuntimeError(f"{path}: sparse overlay is not bound to instanttensor")
        result["overlay"] = _file(path / "OVERLAY.json")
    return result


def _validate_equal_cost(path: Path, expected_layers: set[int]) -> dict:
    receipt_path = path / "MIXED_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    mxfp6 = set(receipt["mxfp6_layers"])
    nvfp4 = set(receipt["nvfp4_layers"])
    if mxfp6 != expected_layers:
        raise RuntimeError(f"{path}: allocation does not match frozen layer set")
    if len(mxfp6) != EXPECTED_MXFP6_LAYERS or len(nvfp4) != EXPECTED_NVFP4_LAYERS:
        raise RuntimeError(f"{path}: allocation is not 35 MXFP6 / 7 NVFP4")
    if mxfp6 & nvfp4 or mxfp6 | nvfp4 != set(range(3, 45)):
        raise RuntimeError(f"{path}: routed-layer partition is invalid")
    if not receipt["payload_bpw"] <= MAX_BPW:
        raise RuntimeError(f"{path}: exact payload exceeds {MAX_BPW} bpw")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--uniform-depth-control", type=Path, required=True)
    parser.add_argument("--uniform-rotated", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--full-mxfp6", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--control-design", type=Path, required=True)
    parser.add_argument("--rotation-winner", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    harness = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval")
    implementation = _git(root)
    harness_identity = _git(harness)
    runtime_image_id = subprocess.check_output(
        ["docker", "image", "inspect", RUNTIME_IMAGE, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if runtime_image_id != RUNTIME_IMAGE_ID:
        raise RuntimeError(
            f"runtime image mismatch: expected {RUNTIME_IMAGE_ID}, got {runtime_image_id}"
        )

    analysis = json.loads(args.analysis.read_text())
    design = json.loads(args.design.read_text())
    control_design = json.loads(args.control_design.read_text())
    winner = json.loads(args.rotation_winner.read_text())
    if winner.get("status") != "pass" or not (
        winner.get("winner") is not None or winner.get("model") is not None
    ):
        raise RuntimeError("rotation winner is not a passing frozen endpoint")
    if analysis.get("design_sha256") != sha256_file(args.design):
        raise RuntimeError("Shapley analysis/design identity mismatch")
    if len(analysis.get("evidence", [])) != design.get("unique_coalitions"):
        raise RuntimeError("Shapley analysis does not contain every coalition endpoint")

    selected = set(analysis["selected_mxfp6_layers"])
    control_selected = set(control_design["mxfp6_layers"])
    candidate_receipt = _validate_equal_cost(args.candidate, selected)
    control_receipt = _validate_equal_cost(args.uniform_depth_control, control_selected)
    if candidate_receipt["payload_bytes"] != control_receipt["payload_bytes"]:
        raise RuntimeError("Shapley and uniform-depth controls are not exactly equal-cost")

    full_receipt = json.loads((args.full_mxfp6 / "MIXED_RECEIPT.json").read_text())
    if set(full_receipt["mxfp6_layers"]) != set(range(3, 45)):
        raise RuntimeError("full MXFP6 endpoint is incomplete")
    if abs(full_receipt["payload_bpw"] - 6.250007629394531) > 1e-9:
        raise RuntimeError("full MXFP6 endpoint has unexpected exact bpw")

    # Confirmation must still be unopened at the moment of authorization.
    opened = sorted(args.run_root.glob("run-*confirmation*.json"))
    if opened:
        raise RuntimeError(f"confirmation already opened: {opened}")

    payload = {
        "schema": "glm53-nvfp4-v3.shapley-6bpw-freeze.v1",
        "status": "pass",
        "confirmation_authorized": True,
        "claim_boundary": "confirmation-level; no fresh final holdout exists",
        "allocation": {
            "mxfp6_layers": sorted(selected),
            "nvfp4_layers": sorted(set(range(3, 45)) - selected),
            "payload_bytes": candidate_receipt["payload_bytes"],
            "logical_elements": candidate_receipt["logical_elements"],
            "payload_bpw": candidate_receipt["payload_bpw"],
            "budget_bpw": MAX_BPW,
        },
        "primary_comparison": "shapley_6bpw_minus_uniform_rotated_nvfp4",
        "equal_cost_comparison": "shapley_6bpw_minus_uniform_depth_6bpw",
        "secondary_comparison": "shapley_6bpw_minus_stock_nvfp4",
        "diagnostic_comparison": "full_mxfp6_6.25bpw_minus_uniform_rotated_nvfp4",
        "models": {
            "shapley_6bpw": _model(args.candidate, mixed=True),
            "uniform_depth_6bpw": _model(args.uniform_depth_control, mixed=True),
            "uniform_rotated_nvfp4": _model(args.uniform_rotated, mixed=False),
            "stock_nvfp4": _model(args.stock, mixed=False),
            "full_mxfp6_6.25bpw": _model(args.full_mxfp6, mixed=True),
        },
        "evidence": {
            "analysis": _file(args.analysis),
            "design": _file(args.design),
            "candidate_set": _file(args.candidate_set),
            "control_design": _file(args.control_design),
            "rotation_winner": _file(args.rotation_winner),
            "roles": _file(args.roles),
        },
        "implementation": implementation,
        "analysis_harness": harness_identity,
        "runtime_image": {
            "tag": RUNTIME_IMAGE,
            "id": runtime_image_id,
        },
        "analysis_files": [
            _file(root / "glm53_nvfp4/role_eval.py"),
            _file(root / "glm53_nvfp4/paired_role_analysis.py"),
            _file(root / "glm53_nvfp4/shapley_analysis.py"),
            _file(root / "runtime_patch/sitecustomize.py"),
            _file(root / "runtime_patch/b12x_h16/Dockerfile"),
            _file(root / "runtime_patch/b12x_h16/b12x/integration/vllm/plugin.py"),
            _file(root / "runtime_patch/b12x_h16/b12x/integration/vllm/fp6_serving.py"),
            _file(root / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"),
            _file(root / "scripts/run_kld_v3.sh"),
            _file(root / "scripts/run_candidate_canary.sh"),
        ],
        "candidate_build_files": [
            _file(root / "glm53_nvfp4/mxfp6_layer.py"),
            _file(root / "glm53_nvfp4/mixed_candidate.py"),
            _file(root / "glm53_nvfp4/prepare_shapley_candidates.py"),
            _file(root / "glm53_nvfp4/shapley_design.py"),
            _file(root / "glm53_nvfp4/freeze_shapley_inputs.py"),
            _file(root / "glm53_nvfp4/freeze_6bpw.py"),
            _file(root / "scripts/run_mxfp6_layer.sh"),
            _file(root / "scripts/run_mxfp6_all.sh"),
            _file(root / "scripts/run_shapley_v10.sh"),
            _file(root / "scripts/run_shapley_6bpw_campaign_v10.sh"),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "payload_bpw": candidate_receipt["payload_bpw"]}))


if __name__ == "__main__":
    main()
