"""Freeze the complete coalition endpoint set before target KLD scoring."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .shard_index import sha256_file


RUNTIME_IMAGE = "klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v80-mxfp6-h16-all"


def _file(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(path: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError(f"repository is dirty: {path}")
    return {"root": str(path.resolve()), "commit": commit, "clean": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--control-design", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    manifest = json.loads(args.manifest.read_text())
    winner = json.loads(args.winner.read_text())
    if winner.get("status") != "pass" or winner.get("rotation_scope") != "all":
        raise RuntimeError("rotation winner is not the corrected all-projection endpoint")
    rows = manifest.get("candidates", [])
    if len(rows) != design.get("unique_coalitions") or {r["coalition_id"] for r in rows} != set(design["coalitions"]):
        raise RuntimeError("candidate manifest does not cover the sealed design")
    for row in rows:
        path = Path(row["path"])
        if row.get("carrier"):
            if not (path / "OVERLAY.json").is_file():
                raise RuntimeError("empty-coalition rotated carrier is missing its overlay")
        elif sha256_file(path / "MIXED_RECEIPT.json") != row.get("receipt_sha256"):
            raise RuntimeError(f"coalition receipt mismatch: {row['coalition_id']}")
    root = Path(__file__).resolve().parent.parent
    harness = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval")
    payload = {
        "schema": "glm53-nvfp4-v10.shapley-input-freeze.v1",
        "status": "pass",
        "conditional_fit_authorized": True,
        "coalitions": len(rows),
        "design": _file(args.design),
        "control_design": _file(args.control_design),
        "candidate_set": _file(args.manifest),
        "roles": _file(args.roles),
        "rotation_winner": _file(args.winner),
        "implementation": _git(root),
        "analysis_harness": _git(harness),
        "runtime_image": {
            "tag": RUNTIME_IMAGE,
            "id": subprocess.check_output(
                ["docker", "image", "inspect", RUNTIME_IMAGE, "--format", "{{.Id}}"],
                text=True,
            ).strip(),
        },
        "analysis_files": [
            _file(root / "glm53_nvfp4/role_eval.py"),
            _file(root / "glm53_nvfp4/shapley_analysis.py"),
            _file(root / "scripts/run_kld_v3.sh"),
            _file(root / "runtime_patch/sitecustomize.py"),
            _file(root / "runtime_patch/b12x_h16/Dockerfile"),
            _file(root / "runtime_patch/b12x_h16/b12x/integration/vllm/plugin.py"),
            _file(root / "runtime_patch/b12x_h16/b12x/integration/vllm/fp6_serving.py"),
            _file(root / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "coalitions": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
