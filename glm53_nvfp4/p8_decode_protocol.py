"""Role and causal-alignment gates for full-window P8 single-row closure.

Importing this module neither opens teacher logits nor launches GPU work.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from .role_eval import (
    HARNESS, METRIC_CODE_SHA256, MODEL_REVISION, VOCAB_LIMIT,
    FULL_PANEL_SOURCE_REVISION, _verify_harness, _hub_expected,
)

ROLES_SHA256 = "b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14"
DOMAINS = {"axis1_general", "axis2_legal", "axis3_code_agentic", "axis4_reasoning_termination"}
ROWS = 2047


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokens_sha(tokens):
    """Canonical value hash, distinct from the existing .npy container hash."""
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def validate_role_payload(payload):
    if (payload.get("schema") != "glm53-codec.conditional-fit32-development-role.v1"
            or payload.get("teacher_repo_id") != "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
            or payload.get("teacher_revision") != FULL_PANEL_SOURCE_REVISION):
        raise ValueError("role provenance differs")
    roles = payload.get("roles", {})
    if set(roles) != {"fit", "conditional-fit", "selection", "confirmation", "final"}:
        raise ValueError("role inventory differs")
    if any(roles[name] for name in roles if name != "conditional-fit"):
        raise ValueError("only the existing conditional-fit role may be opened")
    windows = roles["conditional-fit"]
    if len(windows) != 32 or len({w["id"] for w in windows}) != 32:
        raise ValueError("exactly 32 unique windows required")
    counts = Counter(w["domain"] for w in windows)
    if set(counts) != DOMAINS or set(counts.values()) != {8}:
        raise ValueError("role must retain eight windows per domain")
    for window in windows:
        window_id = window["id"]
        if (not re.fullmatch(r"conditional-fit-\d{4}", window_id)
                or window.get("prediction_positions") != ROWS
                or window.get("teacher_source_role") != "conditional-fit"
                or window.get("teacher_path") != f"logits/full-panel/conditional-fit/{window_id}.safetensors"):
            raise ValueError("window role, path or causal geometry differs")
    return windows


def load_role_inputs(roles_path, teacher_root, *, verify_teacher_bytes=True):
    """Read only the frozen 32-window inputs; never enumerate protected logits."""
    from safetensors import safe_open

    roles_path, teacher_root = Path(roles_path), Path(teacher_root)
    if sha(roles_path) != ROLES_SHA256:
        raise ValueError("fixed conditional-fit32 role hash differs")
    payload = json.loads(roles_path.read_text())
    windows = validate_role_payload(payload)
    manifest_path = teacher_root / "logits/full-panel/full-panel-manifest.json"
    if sha(manifest_path) != payload["teacher_manifest_sha256"]:
        raise ValueError("teacher manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("vocab_size") != VOCAB_LIMIT
            or manifest.get("logits_dtype") != "float32"
            or manifest.get("source_hub_revision") != FULL_PANEL_SOURCE_REVISION):
        raise ValueError("teacher precision or revision differs")
    entries = {entry["path"]: entry for entry in manifest["logit_files"]}
    verified = []
    for window in windows:
        token_path = Path(window["token_path"])
        if sha(token_path) != window["input_sha256"]:
            raise ValueError("token file identity differs")
        tokens = np.load(token_path, allow_pickle=False)
        validate_tokens(tokens, ROWS + 1)
        teacher = teacher_root / window["teacher_path"]
        if teacher.resolve() != teacher or not teacher.is_relative_to(teacher_root):
            raise ValueError("teacher path is not canonical and in the approved root")
        entry = entries[window["teacher_path"]]
        if (entry["window_id"] != window["id"] or entry["role"] != "conditional-fit"
                or entry["domain"] != window["domain"]
                or entry["prediction_positions"] != ROWS
                or entry["token_ids_sha256"] != window["input_sha256"]
                or entry["sha256"] != window["teacher_sha256"]
                or entry["bytes"] != teacher.stat().st_size):
            raise ValueError("teacher entry does not match the fixed role")
        _, hub_sha = _hub_expected(teacher_root, window["teacher_path"])
        if hub_sha != entry["sha256"]:
            raise ValueError("Hub teacher identity differs")
        if verify_teacher_bytes and sha(teacher) != entry["sha256"]:
            raise ValueError("teacher bytes differ")
        with safe_open(str(teacher), framework="pt", device="cpu") as handle:
            tensor = handle.get_slice("logits")
            metadata = handle.metadata() or {}
            if (tuple(tensor.get_shape()) != (ROWS, VOCAB_LIMIT) or tensor.get_dtype() != "F32"
                    or metadata.get("window_id") != window["id"]
                    or metadata.get("model_revision") != MODEL_REVISION
                    or metadata.get("token_ids_sha256") != window["input_sha256"]):
                raise ValueError("teacher stored tensor/header differs")
        verified.append({**window, "token_values_sha256": tokens_sha(tokens),
                         "teacher_bytes_verified": verify_teacher_bytes})
    return verified


def validate_tokens(tokens, expected_length):
    if (not isinstance(tokens, np.ndarray) or tokens.ndim != 1
            or tokens.size != expected_length or tokens.dtype.kind not in "iu"
            or np.any(tokens < 0) or np.any(tokens >= VOCAB_LIMIT)):
        raise ValueError("invalid causal token IDs")


def full_window_contract(tokens):
    validate_tokens(tokens, ROWS + 1)
    return {"prompt_token_ids": tokens[:1].tolist(),
            "forced_token_ids": tokens[1:].tolist(),
            "prediction_rows": ROWS, "one_token_prefill_rows": 1,
            "true_decode_rows": ROWS - 1,
            "teacher_row_start": 0, "teacher_row_stop": ROWS,
            "token_values_sha256": tokens_sha(tokens)}


def completion_request(tokens, window_id, model):
    """Force the complete original history; never compare detokenized text."""
    if not re.fullmatch(r"conditional-fit-\d{4}", window_id):
        raise ValueError("request must belong to the fixed conditional-fit role")
    contract = full_window_contract(tokens)
    prompt, forced = contract["prompt_token_ids"], contract["forced_token_ids"]
    return {"model": model, "prompt": prompt, "max_tokens": len(forced),
            "temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0,
            "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
            "ignore_eos": True, "stop": [], "stop_token_ids": [],
            "add_special_tokens": False, "skip_special_tokens": False,
            "return_token_ids": True, "stream": False, "seed": 20260905,
            "vllm_xargs": {
                "p8_decode_capture_window_id": window_id,
                "p8_decode_capture_forced_token_ids": forced,
                "p8_decode_capture_start_output_len": 0,
                "p8_decode_capture_prompt_token_ids_sha256": tokens_sha(prompt),
                "p8_decode_capture_forced_token_ids_sha256": tokens_sha(forced),
            }}


def verify_response(response, request):
    choices = response.get("choices", [])
    forced = request["vllm_xargs"]["p8_decode_capture_forced_token_ids"]
    if (response.get("model") != request["model"] or len(choices) != 1
            or choices[0].get("token_ids") != forced
            or choices[0].get("finish_reason") != "length"
            or response.get("usage", {}).get("prompt_tokens") != 1
            or response.get("usage", {}).get("completion_tokens") != len(forced)):
        raise ValueError("server did not consume and return the exact forced sequence")
    return {"forced_token_ids_verified": True, "prompt_tokens": 1,
            "completion_tokens": len(forced), "prediction_positions": len(forced)}


def load_capture(root, window_id, tokens):
    """Validate a completed full-window raw capture before returning a read-only map."""
    root = Path(root)
    request = completion_request(tokens, window_id, "validation-only")
    metadata_path = root / f"{window_id}.capture.json"
    raw_path = root / f"{window_id}.logits.f32"
    if metadata_path.resolve() != metadata_path or raw_path.resolve() != raw_path:
        raise ValueError("capture paths must be canonical, without symlinks")
    metadata = json.loads(metadata_path.read_text())
    xargs = request["vllm_xargs"]
    required = {
        "schema": "glm53-p8.forced-decode-logits.v1", "status": "complete",
        "window_id": window_id, "dtype": "<f4", "shape": [ROWS, VOCAB_LIMIT],
        "real_vocab_size": VOCAB_LIMIT, "rows_completed": ROWS,
        "capture_start_output_len": 0, "captured_output_len_range": [0, ROWS - 1],
        "causal_teacher_row_range": [0, ROWS - 1], "one_token_prefill_row_range": [0, 0],
        "true_decode_row_range": [1, ROWS - 1], "prompt_token_count": 1,
        "forced_token_count": ROWS,
        "prompt_token_ids_sha256": xargs["p8_decode_capture_prompt_token_ids_sha256"],
        "forced_token_ids_sha256": xargs["p8_decode_capture_forced_token_ids_sha256"],
        "sequence_token_ids_sha256": tokens_sha(tokens), "raw_file": raw_path.name,
        "raw_bytes": ROWS * VOCAB_LIMIT * 4, "pre_mask_raw_fp32": True,
        "finite_real_vocab": True, "tp_rank": 0,
    }
    if (any(metadata.get(key) != value for key, value in required.items())
            or metadata.get("original_logit_dtype") not in ("torch.float32", "torch.float16", "torch.bfloat16")
            or metadata.get("original_logit_width", 0) < VOCAB_LIMIT
            or metadata.get("request_extra_arg_keys") != sorted(xargs)
            or metadata.get("completed_unix_ns", 0) <= metadata.get("started_unix_ns", 0)
            or raw_path.stat().st_size != required["raw_bytes"]
            or sha(raw_path) != metadata.get("raw_sha256")):
        raise ValueError("capture identity, rows, precision or raw hash differs")
    for suffix in ("capture.failed.json", "capture.inprogress.json", "logits.f32.partial"):
        if (root / f"{window_id}.{suffix}").exists():
            raise ValueError("capture has contradictory incomplete/failure artifacts")
    return np.memmap(raw_path, mode="r", dtype="<f4", shape=(ROWS, VOCAB_LIMIT)), metadata


def exact_logits(reference, candidate, *, chunk_rows=32):
    """Bitwise real-vocabulary closure, with bounded-memory failure diagnostics."""
    if (reference.shape != candidate.shape or reference.ndim != 2
            or reference.dtype != np.dtype("<f4") or candidate.dtype != np.dtype("<f4")
            or reference.shape[1] != VOCAB_LIMIT or chunk_rows < 1):
        raise ValueError("raw logit dtype/geometry differs")
    mismatches = 0
    differing_rows = []
    maximum = 0.0
    for start in range(0, len(reference), chunk_rows):
        a = np.ascontiguousarray(reference[start:start + chunk_rows])
        b = np.ascontiguousarray(candidate[start:start + chunk_rows])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite logits cannot qualify closure")
        unequal = a.view(np.uint32) != b.view(np.uint32)
        mismatches += int(unequal.sum())
        differing_rows.extend((np.flatnonzero(unequal.any(axis=1)) + start).tolist())
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b))))
    return {"exact": mismatches == 0, "unequal_values": mismatches,
            "differing_rows": differing_rows, "maximum_absolute_difference": maximum,
            "rows": len(reference), "vocabulary": VOCAB_LIMIT}


def score_aligned(teacher, student, tokens):
    """Use the existing pinned CPU FP64 metric; no CUDA scoring or row shift."""
    _verify_harness()
    metric = HARNESS / "kld_eval/kld/core.py"
    if sha(metric) != METRIC_CODE_SHA256:
        raise ValueError("causal KLD metric code changed")
    sys.path.insert(0, str(HARNESS))
    from kld_eval.kld.core import score_window
    validate_tokens(tokens, len(teacher) + 1)
    scores = score_window(teacher, student, tokens, VOCAB_LIMIT, chunk_rows=16)
    if not np.isfinite(scores.kld).all():
        raise ValueError("nonfinite KLD")
    return scores
