"""Seal a preregistered selection stop without opening confirmation evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def file_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def candidate_inventory(candidate: Path, carrier: Path) -> dict:
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    index_path = candidate / "model.safetensors.index.json"
    candidate_index = json.loads(index_path.read_text())
    overlay_path = candidate / "OVERLAY.json"
    overlay = json.loads(overlay_path.read_text())
    targets = {
        name
        for name in original["weight_map"]
        if ".mlp.experts." in name
        and name.endswith((".weight", ".weight_scale", ".weight_scale_2"))
        and not name.endswith(".input_scale")
    }
    expected = 42 * 288 * 3 * 3
    if len(targets) != expected:
        raise RuntimeError(f"unexpected carrier target inventory: {len(targets)}")
    changed = {name for name in original["weight_map"] if original["weight_map"][name] != candidate_index["weight_map"][name]}
    chunks = sorted({candidate_index["weight_map"][name] for name in targets})
    if changed != targets or overlay.get("redirected_tensors") != expected or len(chunks) != 168:
        raise RuntimeError("stopped candidate does not contain the exact complete routed-expert overlay")
    missing = [name for name in chunks if not (candidate / name).exists()]
    if missing:
        raise RuntimeError(f"missing candidate chunks: {missing[:3]}")
    return {
        "root": str(candidate.resolve()),
        "index": file_receipt(index_path),
        "overlay": file_receipt(overlay_path),
        "changed_tensors": len(changed),
        "chunks": len(chunks),
        "chunk_names_sha256": sha256_file_list(chunks),
    }


def sha256_file_list(values: list[str]) -> str:
    import hashlib

    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def update_record(record: dict, *, selection: dict, terminal_receipt: Path, source_verify: Path, final_state: Path) -> None:
    observations = {
        "source_integrity": (
            f"complete pinned source verified after streamed generation; receipt sha256={sha256_file(source_verify)}; "
            "timing deviation recorded"
        ),
        "pilot_export_load": "passed corrected layer-3 packed/reload/runtime canary before the full campaign",
        "candidate_freeze": "not applicable: the preregistered selection stop prohibited candidate freeze and confirmation access",
        "restoration": f"final state receipt sha256={sha256_file(final_state)}",
    }
    for invariant in record["invariants"]:
        name = invariant["name"]
        if name not in observations:
            continue
        invariant["observed"] = observations[name]
        invariant["status"] = "pending" if name == "candidate_freeze" else "passed"
    if not any(item.get("stage") == "source-integrity-timing" for item in record["deviations"]):
        record["deviations"].append(
            {
                "stage": "source-integrity-timing",
                "classification": "late full-repository verification",
                "description": (
                    "The 642.7 GB repository-wide Hugging Face cache verification completed after the streamed layer campaign, "
                    "rather than before the one-layer pilot as stated by the invariant. Each consumed shard arrived through the "
                    "pinned Hub downloader; the complete local repository subsequently passed strict verification."
                ),
                "decision_impact": (
                    "This is a protocol-timing deviation. It does not rescue the candidate: the candidate was stopped on the "
                    "protected selection role, and confirmation remained unopened."
                ),
            }
        )
    record["qualification"] = {
        "status": "failed",
        "attempt_count": 0,
        "primary_result": {
            key: selection[key]
            for key in (
                "candidate_mean_kld",
                "stock_mean_kld",
                "mean_delta_kld",
                "relative_improvement",
                "delta_ci95_percentile",
                "delta_ci95_bca",
            )
        },
        "decision": "selection-stop",
        "raw_evidence_sha256": sha256_file(terminal_receipt),
        "limitations": [
            "Confirmation was not opened because the preregistered selection rule stopped the candidate.",
            "Performance, Estonia, and LAVD benchmarks were not run because the quality gate failed.",
            "Single quantization campaign does not estimate between-campaign variance.",
            "Layer calibration used pinned BF16 captures and group-16 block Hessians, not causal propagation of quantized activations.",
        ],
    }
    record["authorization"]["compute"] = "completed"
    record["status"] = "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--source-verify", type=Path, required=True)
    parser.add_argument("--teacher-verify", type=Path, required=True)
    parser.add_argument("--final-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    selection_path = root / "evidence/selection-analysis.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("role") != "selection" or selection.get("decision") != "stop" or selection.get("windows") != 16:
        raise RuntimeError("selection evidence does not authorize the stopped terminal path")
    roles = json.loads((root / "roles/roles.json").read_text())
    expected_windows = {item["id"] for item in roles["roles"]["selection"]}
    manifests = []
    for run_id in ("selection-stock", "selection-candidate"):
        path = root / f"kld/run-{run_id}.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete" or set(manifest.get("windows", [])) != expected_windows:
            raise RuntimeError(f"incomplete selection manifest: {path}")
        manifests.append(file_receipt(path))
    if (root / "evidence/candidate-freeze.json").exists() or (root / "evidence/confirmation-analysis.json").exists():
        raise RuntimeError("confirmation boundary was crossed despite selection stop")
    if any((root / "benchmarks").glob("*/summary.json")):
        raise RuntimeError("performance benchmark evidence exists despite selection stop")
    validations = []
    for layer in range(3, 45):
        path = root / f"evidence/layer-{layer:03d}-validation.json"
        result = json.loads(path.read_text())
        if result.get("layer") != layer or result.get("status") != "pass":
            raise RuntimeError(f"invalid layer validation: {layer}")
        validations.append(file_receipt(path))
    subprocess.run(["git", "cat-file", "-e", f"{args.candidate_commit}^{{commit}}"], check=True)
    source = json.loads(args.source_verify.read_text())
    if source.get("repo_id") != "zai-org/GLM-5.3-Flash-BF16" or source.get("checked", 0) < 130:
        raise RuntimeError("full pinned source verification receipt is invalid")
    teacher = json.loads(args.teacher_verify.read_text())
    if teacher.get("repo_id") != "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits" or teacher.get("checked", 0) < 75:
        raise RuntimeError("role-approved teacher verification receipt is invalid")
    final_state = json.loads(args.final_state.read_text())
    service = final_state["commands"]["service"]
    backend = final_state["commands"]["backend_service"]
    timer = final_state["commands"]["model_stack_timer"]
    production = final_state["commands"]["production_container"]
    port = final_state["commands"]["port_8000"]
    if (
        service.get("output") != "inactive"
        or backend.get("output") != "active"
        or timer.get("output") != "inactive"
        or production.get("output") != "false"
        or port.get("output", "").strip().splitlines()[1:]
    ):
        raise RuntimeError("pre-run model service/port state was not restored")
    candidate = candidate_inventory(root / "candidates/uniform-gptq", args.carrier)
    candidate_session = root / "kld/sessions/selection-candidate"
    models = json.loads((candidate_session / "models.json").read_text())
    served_ids = {item.get("id") for item in models.get("data", [])}
    backend_proof = (candidate_session / "backend-proof.log").read_text()
    image_digests = (candidate_session / "image-digests.json").read_text()
    if "GLM-5.3-Flash-NVFP4-V2" not in served_ids:
        raise RuntimeError("selection runtime did not serve the complete candidate identity")
    if "FLASHINFER_MLA_SPARSE_SM120" not in backend_proof or not any(
        marker in backend_proof.lower() for marker in ("nvfp4", "modelopt")
    ):
        raise RuntimeError("selection runtime backend proof is incomplete")
    if "sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe" not in image_digests:
        raise RuntimeError("selection runtime image digest mismatch")
    payload = {
        "schema": "glm53-nvfp4-v2.selection-stop.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stopped",
        "confirmation_authorized": False,
        "decision": selection,
        "candidate": candidate,
        "candidate_generation_commit": args.candidate_commit,
        "roles": file_receipt(root / "roles/roles.json"),
        "source_verify": file_receipt(args.source_verify),
        "teacher_verify": file_receipt(args.teacher_verify),
        "selection_manifests": manifests,
        "selection_analysis": file_receipt(selection_path),
        "candidate_runtime": {
            "models": file_receipt(candidate_session / "models.json"),
            "backend_proof": file_receipt(candidate_session / "backend-proof.log"),
            "image_digests": file_receipt(candidate_session / "image-digests.json"),
        },
        "layer_validations": validations,
        "restoration": file_receipt(args.final_state),
        "confirmation": "not opened",
        "benchmarks": "skipped by preregistered quality gate",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    record = json.loads(args.record.read_text())
    update_record(
        record,
        selection=selection,
        terminal_receipt=args.output,
        source_verify=args.source_verify,
        final_state=args.final_state,
    )
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "completed", "decision": "selection-stop", "changed_tensors": candidate["changed_tensors"]}, sort_keys=True))


if __name__ == "__main__":
    main()
