"""Freeze the complete candidate and pre-confirmation evidence inventory."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file

HARNESS = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval")
HARNESS_COMMIT = "565aca8ded8f015eeecb7f9e2aa99e1da965e5e7"


def git_identity(root: Path) -> dict:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    return {"root": str(root), "commit": head, "clean": not bool(dirty)}


def file_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--selection-candidate", type=Path, required=True)
    parser.add_argument("--selection-stock", type=Path, required=True)
    parser.add_argument("--selection-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    implementation = git_identity(Path(__file__).resolve().parent.parent)
    harness = git_identity(HARNESS)
    if not implementation["clean"] or not harness["clean"] or harness["commit"] != HARNESS_COMMIT:
        raise RuntimeError("implementation and analysis repositories must be clean and pinned")

    roles = json.loads(args.roles.read_text())
    expected_selection = {item["id"] for item in roles["roles"]["selection"]}
    selection_manifests = []
    for path in (args.selection_candidate, args.selection_stock):
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete" or manifest.get("role") != "selection":
            raise RuntimeError(f"incomplete selection manifest: {path}")
        if set(manifest["windows"]) != expected_selection:
            raise RuntimeError(f"selection window mismatch: {path}")
        selection_manifests.append(file_receipt(path))
    selection_analysis = json.loads(args.selection_analysis.read_text())
    if selection_analysis.get("role") != "selection" or selection_analysis.get("decision") not in ("continue", "pass"):
        raise RuntimeError("selection analysis did not authorize candidate freeze")

    original = json.loads((args.carrier / "model.safetensors.index.json").read_text())
    candidate = json.loads((args.candidate / "model.safetensors.index.json").read_text())
    overlay = json.loads((args.candidate / "OVERLAY.json").read_text())
    target_names = {
        name for name in original["weight_map"]
        if ".mlp.experts." in name and name.endswith((".weight", ".weight_scale", ".weight_scale_2"))
        and not name.endswith(".input_scale")
    }
    expected_target_count = 42 * 288 * 3 * 3
    if len(target_names) != expected_target_count:
        raise RuntimeError(f"unexpected carrier target inventory: {len(target_names)}")
    changed = {name for name in original["weight_map"] if original["weight_map"][name] != candidate["weight_map"][name]}
    if changed != target_names or overlay["redirected_tensors"] != expected_target_count:
        raise RuntimeError("candidate index does not change exactly the routed-expert payload")
    chunks = sorted({candidate["weight_map"][name] for name in target_names})
    if len(chunks) != 42 * 4:
        raise RuntimeError(f"expected 168 candidate chunks, found {len(chunks)}")
    chunk_receipts = []
    tensor_count = 0
    for name in chunks:
        path = args.candidate / name
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensor_count += len(list(handle.keys()))
        chunk_receipts.append(file_receipt(path.resolve()))
    if tensor_count != expected_target_count:
        raise RuntimeError(f"candidate chunks contain {tensor_count} tensors")
    layer_validations = []
    for layer in range(3, 45):
        path = args.campaign / f"evidence/layer-{layer:03d}-validation.json"
        validation = json.loads(path.read_text())
        if validation.get("status") != "pass" or validation.get("layer") != layer:
            raise RuntimeError(f"layer validation failed: {layer}")
        layer_validations.append(file_receipt(path))

    payload = {
        "schema": "glm53-nvfp4-v2.candidate-freeze.v1",
        "status": "pass",
        "confirmation_authorized": True,
        "candidate": {
            "root": str(args.candidate.resolve()),
            "index": file_receipt(args.candidate / "model.safetensors.index.json"),
            "overlay": file_receipt(args.candidate / "OVERLAY.json"),
            "changed_tensors": len(changed),
            "chunks": chunk_receipts,
        },
        "carrier_index": file_receipt(args.carrier / "model.safetensors.index.json"),
        "implementation": implementation,
        "analysis_harness": harness,
        "analysis_files": [
            file_receipt(HARNESS / "kld_eval/kld/core.py"),
            file_receipt(HARNESS / "kld_eval/kld/records.py"),
            file_receipt(HARNESS / "kld_eval/adapters/vllm_b12x.py"),
            file_receipt(Path(__file__).with_name("role_eval.py")),
            file_receipt(Path(__file__).with_name("paired_role_analysis.py")),
        ],
        "roles": file_receipt(args.roles),
        "selection_manifests": selection_manifests,
        "selection_analysis": file_receipt(args.selection_analysis),
        "layer_validations": layer_validations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "changed_tensors": len(changed), "chunks": len(chunks), "implementation_commit": implementation["commit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
