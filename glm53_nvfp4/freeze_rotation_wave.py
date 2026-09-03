"""Freeze and verify the complete V3 H16/learned gate-up rotation candidate set."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(root: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    return {"root": str(root), "commit": commit, "clean": not dirty}


def _expected_gate_up(weight_map: dict[str, str]) -> set[str]:
    result = set()
    for layer in range(3, 45):
        for expert in range(288):
            for projection in ("gate_proj", "up_proj"):
                stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    name = f"{stem}.{suffix}"
                    if name not in weight_map:
                        raise KeyError(name)
                    result.add(name)
    return result


def _candidate(candidate: Path, stock_map: dict[str, str], expected: set[str]) -> dict:
    index_path = candidate / "model.safetensors.index.json"
    overlay_path = candidate / "OVERLAY.json"
    index = json.loads(index_path.read_text())
    overlay = json.loads(overlay_path.read_text())
    changed = {name for name in stock_map if index["weight_map"][name] != stock_map[name]}
    if changed != expected:
        raise RuntimeError(f"{candidate}: changed tensor mismatch {len(changed)} != {len(expected)}")
    if overlay["redirected_tensors"] != len(expected):
        raise RuntimeError(f"{candidate}: overlay count mismatch")
    chunks = sorted({index["weight_map"][name] for name in expected})
    if len(chunks) != 168:
        raise RuntimeError(f"{candidate}: expected 168 gate/up chunks, found {len(chunks)}")
    tensor_count = 0
    files = []
    for chunk in chunks:
        path = (candidate / chunk).resolve()
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensor_count += len(list(handle.keys()))
        files.append(_file(path))
    if tensor_count != len(expected):
        raise RuntimeError(f"{candidate}: chunk tensor count mismatch")
    return {"root": str(candidate.resolve()), "index": _file(index_path), "overlay": _file(overlay_path), "changed_tensors": len(changed), "chunks": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--had16", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--learned-rotations", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    implementation = _git(root)
    harness = _git(Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval"))
    if not implementation["clean"] or not harness["clean"]:
        raise RuntimeError("implementation and analysis repositories must be clean")
    stock_index = json.loads((args.stock / "model.safetensors.index.json").read_text())
    expected = _expected_gate_up(stock_index["weight_map"])
    validations = {"had16": [], "learned": []}
    for layer in range(3, 45):
        for mode in validations:
            path = args.campaign / f"evidence-v3/{mode}/{mode}-layer-{layer:03d}-validation.json"
            value = json.loads(path.read_text())
            if value.get("status") != "pass" or value.get("experts") != 288:
                raise RuntimeError(f"failed validation: {path}")
            validations[mode].append(_file(path))
    payload = {
        "schema": "glm53-nvfp4-v3.rotation-wave-freeze.v1",
        "status": "pass",
        "conditional_fit_authorized": True,
        "selection_wave_1_authorized": True,
        "stock": {"root": str(args.stock.resolve()), "index": _file(args.stock / "model.safetensors.index.json")},
        "candidates": {"had16": _candidate(args.had16, stock_index["weight_map"], expected), "learned": _candidate(args.learned, stock_index["weight_map"], expected)},
        "learned_rotations": _file(args.learned_rotations),
        "roles": _file(args.roles),
        "implementation": implementation,
        "analysis_harness": harness,
        "analysis_files": [
            _file(root / "glm53_nvfp4/role_eval.py"),
            _file(root / "glm53_nvfp4/paired_role_analysis.py"),
            _file(root / "runtime_patch/sitecustomize.py"),
            _file(root / "scripts/run_kld_v3.sh"),
        ],
        "validations": validations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "changed_tensors_per_candidate": len(expected), "implementation_commit": implementation["commit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
