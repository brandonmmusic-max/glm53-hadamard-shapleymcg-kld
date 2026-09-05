"""Role-safe BF16-teacher KLD capture using the pinned local harness kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .shard_index import sha256_file

HARNESS = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval")
HARNESS_COMMIT = "565aca8ded8f015eeecb7f9e2aa99e1da965e5e7"
TEACHER_REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
FULL_PANEL_SOURCE_REVISION = "7c378d5f17dba158c4c803eff27c346dd0615660"
MODEL_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
METRIC_CODE_SHA256 = "752af6740595791f300cf47f15ef81f8ca85245d5282fc600ef26ee57eae41e6"
VOCAB_LIMIT = 154880
ALIGNMENT_BAND = (0.20, 0.995)


def _verify_harness() -> None:
    head = subprocess.check_output(["git", "-C", str(HARNESS), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(HARNESS), "status", "--porcelain"], text=True).strip()
    if head != HARNESS_COMMIT or dirty:
        raise RuntimeError(f"analysis harness identity changed: head={head} dirty={bool(dirty)}")


def _hub_expected(local_root: Path, relative: str) -> tuple[str, str]:
    metadata = local_root / ".cache/huggingface/download" / f"{relative}.metadata"
    lines = metadata.read_text().splitlines()
    if len(lines) < 2 or lines[0] != TEACHER_REVISION:
        raise RuntimeError(f"teacher Hub metadata revision mismatch: {metadata}")
    return lines[0], lines[1]


def _load_teacher(path: Path, window: dict) -> np.ndarray:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensor = handle.get_slice("logits")
        if tuple(tensor.get_shape()) != (window["prediction_positions"], VOCAB_LIMIT):
            raise RuntimeError(f"{window['id']}: teacher logits geometry mismatch")
        if tensor.get_dtype() != "F32":
            raise RuntimeError(f"{window['id']}: teacher logits are not FP32")
        if metadata.get("window_id") != window["id"] or metadata.get("model_revision") != MODEL_REVISION:
            raise RuntimeError(f"{window['id']}: teacher safetensors metadata mismatch")
        if metadata.get("token_ids_sha256") != window["input_sha256"]:
            raise RuntimeError(f"{window['id']}: teacher/token identity mismatch")
        return tensor[:].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("fit", "conditional-fit", "selection", "confirmation"),
        required=True,
    )
    parser.add_argument("--selection-wave", type=int)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    _verify_harness()
    roles = json.loads(args.roles.read_text())
    windows = roles["roles"][args.role]
    if args.selection_wave is not None:
        if args.role != "selection":
            raise RuntimeError("--selection-wave is valid only for selection")
        wave = str(args.selection_wave)
        if wave not in roles["selection_waves"]:
            raise RuntimeError(f"selection wave {wave} is not declared")
        wave_ids = set(roles["selection_waves"][wave])
        windows = [window for window in windows if window["id"] in wave_ids]
        if len(windows) != len(wave_ids):
            raise RuntimeError("selection wave ids do not resolve exactly")
    teacher_manifest_path = args.teacher_root / "logits/full-panel/full-panel-manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    if (
        teacher_manifest.get("repo_id") != "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
        or teacher_manifest.get("source_hub_revision") != FULL_PANEL_SOURCE_REVISION
        or teacher_manifest.get("model_revision") != MODEL_REVISION
        or teacher_manifest.get("logits_dtype") != "float32"
        or teacher_manifest.get("vocab_size") != VOCAB_LIMIT
    ):
        raise RuntimeError("full-panel teacher manifest identity mismatch")
    teacher_files = {item["path"]: item for item in teacher_manifest["logit_files"]}
    if args.role == "confirmation":
        if args.freeze_receipt is None:
            raise RuntimeError("confirmation requires a frozen-candidate receipt")
        freeze = json.loads(args.freeze_receipt.read_text())
        if freeze.get("status") != "pass" or freeze.get("confirmation_authorized") is not True:
            raise RuntimeError("candidate freeze receipt does not authorize confirmation")

    sys.path.insert(0, str(HARNESS))
    from kld_eval.adapters.vllm_b12x import VllmB12xCaptureAdapter
    from kld_eval.kld.core import score_window
    from kld_eval.kld.records import write_window_records

    args.run_root.mkdir(parents=True, exist_ok=True)
    records = args.run_root / "records" / args.run_id
    manifest_path = args.run_root / f"run-{args.run_id}.json"
    manifest = {
        "schema": "glm53-nvfp4-v2.role-kld-run.v1",
        "run_id": args.run_id,
        "role": args.role,
        "config_id": args.config_id,
        "roles_sha256": sha256_file(args.roles),
        "teacher_revision": TEACHER_REVISION,
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "harness_commit": HARNESS_COMMIT,
        "metric_code_sha256": METRIC_CODE_SHA256,
        "freeze_receipt_sha256": sha256_file(args.freeze_receipt) if args.freeze_receipt else None,
        "started_unix": time.time(),
        "windows": {},
    }
    if args.resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        for key in ("role", "config_id", "roles_sha256", "teacher_revision", "teacher_manifest_sha256", "harness_commit", "metric_code_sha256", "freeze_receipt_sha256"):
            if prior[key] != manifest[key]:
                raise RuntimeError(f"resume identity mismatch: {key}")
        manifest = prior
    adapter = VllmB12xCaptureAdapter(
        url=args.url, model_name=args.model_name, capture_root=args.capture_root,
        container=args.container, keep_payload=False,
    )
    manifest["adapter"] = adapter.describe()
    for window in windows:
        if args.resume and window["id"] in manifest["windows"]:
            continue
        started = time.monotonic()
        token_path = Path(window["token_path"])
        if sha256_file(token_path) != window["input_sha256"]:
            raise RuntimeError(f"{window['id']}: token container hash mismatch")
        tokens = np.load(token_path, allow_pickle=False)
        if tokens.shape != (window["prediction_positions"] + 1,):
            raise RuntimeError(f"{window['id']}: token geometry mismatch")
        teacher_path = args.teacher_root / window["teacher_path"]
        teacher_entry = teacher_files[window["teacher_path"]]
        expected_manifest_role = window.get(
            "teacher_source_role",
            "conditional-fit" if args.role == "conditional-fit" else args.role,
        )
        if (
            teacher_entry["window_id"] != window["id"]
            or teacher_entry["role"] != expected_manifest_role
            or teacher_entry["domain"] != window["domain"]
            or teacher_entry["token_ids_sha256"] != window["input_sha256"]
            or teacher_entry["prediction_positions"] != window["prediction_positions"]
            or teacher_path.stat().st_size != teacher_entry["bytes"]
        ):
            raise RuntimeError(f"{window['id']}: role and teacher manifest disagree")
        _, expected_teacher_sha = _hub_expected(args.teacher_root, window["teacher_path"])
        if expected_teacher_sha != teacher_entry["sha256"]:
            raise RuntimeError(f"{window['id']}: Hub and teacher manifest hashes disagree")
        teacher_sha = sha256_file(teacher_path)
        if teacher_sha != expected_teacher_sha:
            raise RuntimeError(f"{window['id']}: teacher file hash mismatch")
        teacher = _load_teacher(teacher_path, window)
        self_score = score_window(teacher, teacher, tokens, VOCAB_LIMIT)
        if not np.all(self_score.kld == 0.0):
            raise RuntimeError(f"{window['id']}: teacher self-KLD is nonzero")
        student = adapter.score(tokens)
        scores = score_window(teacher, student.logits, tokens, VOCAB_LIMIT)
        teacher_alignment = float(np.mean(scores.teacher_top1 == scores.realized_token))
        if not ALIGNMENT_BAND[0] <= teacher_alignment <= ALIGNMENT_BAND[1]:
            raise RuntimeError(f"{window['id']}: alignment guard failed: {teacher_alignment}")
        record_path = write_window_records(
            records, run_id=args.run_id, config_id=args.config_id,
            window_id=window["id"], document_id=teacher_entry["document_id"], domain=window["domain"],
            context_len=window["prediction_positions"] + 1, scores=scores,
            protocol_sha256=roles["source_panel_sha256"], student_logits_sha256=student.logits_sha256,
            extra_meta={"role": args.role, "teacher_sha256": teacher_sha, "roles_sha256": manifest["roles_sha256"]},
        )
        manifest["windows"][window["id"]] = {
            "domain": window["domain"],
            "teacher_sha256": teacher_sha,
            "student_logits_sha256": student.logits_sha256,
            "record_path": str(record_path),
            "record_sha256": sha256_file(record_path),
            "mean_kld": float(scores.kld.mean()),
            "top1_agreement": float(np.mean(scores.teacher_top1 == scores.student_top1)),
            "teacher_top1_realized_agreement": teacher_alignment,
            "elapsed_seconds": time.monotonic() - started,
            "request": student.meta,
        }
        manifest["finished_unix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"window": window["id"], "mean_kld": manifest["windows"][window["id"]]["mean_kld"]}), flush=True)
        del teacher, student, scores, self_score
    if set(manifest["windows"]) != {window["id"] for window in windows}:
        raise RuntimeError("run did not complete the exact sealed role")
    manifest["status"] = "complete"
    manifest["mean_of_window_means"] = float(np.mean([item["mean_kld"] for item in manifest["windows"].values()]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": args.run_id, "windows": len(windows), "mean_of_window_means": manifest["mean_of_window_means"]}, indent=2))


if __name__ == "__main__":
    main()
