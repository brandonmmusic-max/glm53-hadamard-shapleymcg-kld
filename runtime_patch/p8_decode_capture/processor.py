"""Forced-sequence, pre-mask FP32 logit capture for vLLM 7f1e92bec.

This module is intentionally separate from the serving patch. Load it only in
the dedicated correctness service with::

    --logits-processors \
      p8_decode_capture.processor:ForcedDecodeCaptureLogitsProcessor

The request supplies token identities, never a path. The server operator owns
the output root and exact window allowlist through environment variables. A
final ``*.capture.json`` is written only after every expected row is durable;
partial and failure artifacts remain visibly incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor


SCHEMA = "glm53-p8.forced-decode-logits.v1"
PINNED_VLLM_COMMIT = "7f1e92bec13a05170ff78fd102d03c42487e4836"
REAL_VOCAB_SIZE = 154_880
WINDOW_RE = re.compile(r"conditional-fit-[0-9]{4}\Z")
ROOT_ENV = "GLM53_P8_DECODE_CAPTURE_ROOT"
ALLOWLIST_ENV = "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS"
EXPECTED_OUTPUTS_ENV = "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS"

WINDOW_KEY = "p8_decode_capture_window_id"
FORCED_KEY = "p8_decode_capture_forced_token_ids"
START_KEY = "p8_decode_capture_start_output_len"
PROMPT_HASH_KEY = "p8_decode_capture_prompt_token_ids_sha256"
FORCED_HASH_KEY = "p8_decode_capture_forced_token_ids_sha256"
REQUEST_KEYS = frozenset(
    {WINDOW_KEY, FORCED_KEY, START_KEY, PROMPT_HASH_KEY, FORCED_HASH_KEY}
)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs as a canonical, contiguous little-endian int64 array."""
    values = np.asarray([int(token) for token in token_ids], dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _require_plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VLLMValidationError(f"{name} must be an integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise VLLMValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _expected_outputs() -> int:
    raw = os.environ.get(EXPECTED_OUTPUTS_ENV, "2047")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid {EXPECTED_OUTPUTS_ENV}={raw!r}") from error
    if not 1 <= value <= 2047:
        raise RuntimeError(f"{EXPECTED_OUTPUTS_ENV} must be in [1, 2047]")
    return value


def _allowed_window_ids() -> frozenset[str]:
    raw = os.environ.get(ALLOWLIST_ENV, "")
    values = raw.split(",") if raw else []
    if not values or any(not WINDOW_RE.fullmatch(value) for value in values):
        raise RuntimeError(
            f"{ALLOWLIST_ENV} must be a nonempty comma-separated list of "
            "conditional-fit-NNNN identifiers"
        )
    if len(values) != len(set(values)):
        raise RuntimeError(f"{ALLOWLIST_ENV} contains duplicate identifiers")
    return frozenset(values)


def _capture_root() -> Path:
    raw = os.environ.get(ROOT_ENV, "")
    if not raw or not Path(raw).is_absolute():
        raise RuntimeError(f"{ROOT_ENV} must name an absolute pre-existing directory")
    root = Path(raw).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"{ROOT_ENV} is not a directory: {root}")
    return root


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write while recording capture metadata")
        offset += written


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        _write_all(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class _RequestSpec:
    window_id: str
    forced_token_ids: tuple[int, ...]
    capture_start_output_len: int
    prompt_sha256: str
    forced_sha256: str


def _parse_request(params: SamplingParams) -> _RequestSpec:
    extra = params.extra_args
    if not isinstance(extra, dict) or set(extra) != REQUEST_KEYS:
        present = sorted(extra) if isinstance(extra, dict) else type(extra).__name__
        raise VLLMValidationError(
            f"decode capture requires exactly {sorted(REQUEST_KEYS)}; got {present}"
        )

    window_id = extra[WINDOW_KEY]
    if not isinstance(window_id, str) or not WINDOW_RE.fullmatch(window_id):
        raise VLLMValidationError(f"invalid {WINDOW_KEY}")
    try:
        allowed = _allowed_window_ids()
    except RuntimeError as error:
        raise VLLMValidationError(str(error)) from error
    if window_id not in allowed:
        raise VLLMValidationError(f"{window_id} is not in the server capture allowlist")

    forced_raw = extra[FORCED_KEY]
    if not isinstance(forced_raw, list):
        raise VLLMValidationError(f"{FORCED_KEY} must be a list of integers")
    forced = tuple(
        _require_plain_int(token, f"{FORCED_KEY}[{index}]")
        for index, token in enumerate(forced_raw)
    )
    expected = _expected_outputs()
    if len(forced) != expected:
        raise VLLMValidationError(
            f"{FORCED_KEY} must contain exactly {expected} tokens, got {len(forced)}"
        )
    if any(token < 0 or token >= REAL_VOCAB_SIZE for token in forced):
        raise VLLMValidationError(f"{FORCED_KEY} contains a token outside real vocabulary")

    start = _require_plain_int(extra[START_KEY], START_KEY)
    if not 0 <= start < len(forced):
        raise VLLMValidationError(f"{START_KEY} must be in [0, {len(forced) - 1}]")
    prompt_hash = _require_sha256(extra[PROMPT_HASH_KEY], PROMPT_HASH_KEY)
    forced_hash = _require_sha256(extra[FORCED_HASH_KEY], FORCED_HASH_KEY)
    actual_forced_hash = token_ids_sha256(forced)
    if forced_hash != actual_forced_hash:
        raise VLLMValidationError(f"{FORCED_HASH_KEY} does not match forced tokens")

    required_fields = {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "ignore_eos": True,
    }
    for name, expected_value in required_fields.items():
        if getattr(params, name, None) != expected_value:
            raise VLLMValidationError(
                f"decode capture requires {name}={expected_value!r}"
            )
    if getattr(params, "top_k", None) not in (0, -1):
        raise VLLMValidationError("decode capture requires disabled top_k")
    if params.max_tokens != len(forced):
        raise VLLMValidationError("max_tokens must equal forced token count")
    for name in ("logprobs", "prompt_logprobs", "logprob_token_ids"):
        if getattr(params, name, None) is not None:
            raise VLLMValidationError(f"decode capture requires {name}=None")
    for name in ("stop", "stop_token_ids", "bad_words", "allowed_token_ids", "logit_bias"):
        if getattr(params, name, None):
            raise VLLMValidationError(f"decode capture requires empty {name}")
    if getattr(params, "structured_outputs", None) is not None:
        raise VLLMValidationError("decode capture forbids structured outputs")
    if getattr(params, "thinking_token_budget", None) is not None:
        raise VLLMValidationError("decode capture forbids thinking_token_budget")

    return _RequestSpec(
        window_id=window_id,
        forced_token_ids=forced,
        capture_start_output_len=start,
        prompt_sha256=prompt_hash,
        forced_sha256=forced_hash,
    )


class _CaptureArtifact:
    def __init__(
        self,
        root: Path,
        spec: _RequestSpec,
        prompt_ids: tuple[int, ...],
        tp_rank: int,
    ) -> None:
        self.root = root
        self.spec = spec
        self.prompt_ids = prompt_ids
        self.tp_rank = tp_rank
        self.rows = len(spec.forced_token_ids) - spec.capture_start_output_len
        self.shape = (self.rows, REAL_VOCAB_SIZE)
        stem = spec.window_id
        self.raw_partial = root / f"{stem}.logits.f32.partial"
        self.raw_final = root / f"{stem}.logits.f32"
        self.inprogress = root / f"{stem}.capture.inprogress.json"
        self.metadata_final = root / f"{stem}.capture.json"
        self.metadata_partial = root / f"{stem}.capture.json.partial"
        self.failed = root / f"{stem}.capture.failed.json"
        for path in (
            self.raw_partial,
            self.raw_final,
            self.inprogress,
            self.metadata_final,
            self.metadata_partial,
            self.failed,
        ):
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"refusing to overwrite decode capture artifact: {path}")

        started = time.time_ns()
        _write_json_exclusive(
            self.inprogress,
            {
                "schema": SCHEMA,
                "status": "in-progress",
                "window_id": spec.window_id,
                "started_unix_ns": started,
                "dtype": "<f4",
                "shape": list(self.shape),
                "capture_start_output_len": spec.capture_start_output_len,
                "prompt_token_ids_sha256": spec.prompt_sha256,
                "forced_token_ids_sha256": spec.forced_sha256,
                "tp_rank": tp_rank,
                "pinned_vllm_interface_commit": PINNED_VLLM_COMMIT,
            },
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.raw_partial, flags, 0o600)
        try:
            os.ftruncate(fd, self.rows * REAL_VOCAB_SIZE * 4)
        finally:
            os.close(fd)
        self.array = np.memmap(
            self.raw_partial, mode="r+", dtype="<f4", shape=self.shape, order="C"
        )
        self.started_unix_ns = started
        self.rows_written = 0
        self.original_logit_width: int | None = None
        self.original_logit_dtype: str | None = None
        self.raw_hash = hashlib.sha256()
        self.complete = False
        _fsync_directory(root)

    def write_row(
        self,
        capture_row: int,
        row: np.ndarray,
        original_width: int,
        original_dtype: str = "torch.float32",
    ) -> None:
        if self.complete:
            raise RuntimeError("capture is already complete")
        if capture_row != self.rows_written:
            raise RuntimeError(
                f"nonsequential or duplicate capture row {capture_row}; "
                f"expected {self.rows_written}"
            )
        if row.shape != (REAL_VOCAB_SIZE,) or row.dtype != np.dtype("float32"):
            raise RuntimeError(f"invalid capture row geometry/dtype: {row.shape} {row.dtype}")
        if not np.isfinite(row).all():
            raise RuntimeError(f"nonfinite values in capture row {capture_row}")
        if self.original_logit_width is None:
            self.original_logit_width = original_width
        elif self.original_logit_width != original_width:
            raise RuntimeError("incoming logit width changed during capture")
        if self.original_logit_dtype is None:
            self.original_logit_dtype = original_dtype
        elif self.original_logit_dtype != original_dtype:
            raise RuntimeError("incoming logit dtype changed during capture")
        canonical = np.ascontiguousarray(row, dtype="<f4")
        self.array[capture_row] = canonical
        self.raw_hash.update(canonical.tobytes(order="C"))
        self.rows_written += 1

    def finalize(self) -> None:
        if self.complete:
            raise RuntimeError("capture finalized more than once")
        if (
            self.rows_written != self.rows
            or self.original_logit_width is None
            or self.original_logit_dtype is None
        ):
            raise RuntimeError(
                f"cannot finalize incomplete capture: {self.rows_written}/{self.rows} rows"
            )
        self.array.flush()
        self.array._mmap.close()  # NumPy has no public close API for memmap.
        fd = os.open(self.raw_partial, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(self.raw_partial, self.raw_final)
        os.unlink(self.raw_partial)
        _fsync_directory(self.root)

        start = self.spec.capture_start_output_len
        first_decode = max(start, 1)
        metadata = {
            "schema": SCHEMA,
            "status": "complete",
            "window_id": self.spec.window_id,
            "started_unix_ns": self.started_unix_ns,
            "completed_unix_ns": time.time_ns(),
            "dtype": "<f4",
            "shape": list(self.shape),
            "real_vocab_size": REAL_VOCAB_SIZE,
            "original_logit_width": self.original_logit_width,
            "original_logit_dtype": self.original_logit_dtype,
            "rows_completed": self.rows_written,
            "capture_start_output_len": start,
            "captured_output_len_range": [start, len(self.spec.forced_token_ids) - 1],
            "causal_teacher_row_range": [start, len(self.spec.forced_token_ids) - 1],
            "one_token_prefill_row_range": [0, 0] if start == 0 else None,
            "true_decode_row_range": (
                [first_decode, len(self.spec.forced_token_ids) - 1]
                if first_decode < len(self.spec.forced_token_ids)
                else None
            ),
            "prompt_token_count": len(self.prompt_ids),
            "forced_token_count": len(self.spec.forced_token_ids),
            "prompt_token_ids_sha256": self.spec.prompt_sha256,
            "forced_token_ids_sha256": self.spec.forced_sha256,
            "sequence_token_ids_sha256": token_ids_sha256(
                (*self.prompt_ids, *self.spec.forced_token_ids)
            ),
            "raw_file": self.raw_final.name,
            "raw_bytes": self.rows * REAL_VOCAB_SIZE * 4,
            "raw_sha256": self.raw_hash.hexdigest(),
            "pre_mask_raw_fp32": True,
            "finite_real_vocab": True,
            "tp_rank": self.tp_rank,
            "request_extra_arg_keys": sorted(REQUEST_KEYS),
            "pinned_vllm_interface_commit": PINNED_VLLM_COMMIT,
        }
        _write_json_exclusive(self.metadata_partial, metadata)
        os.link(self.metadata_partial, self.metadata_final)
        os.unlink(self.metadata_partial)
        os.unlink(self.inprogress)
        _fsync_directory(self.root)
        self.complete = True

    def record_failure(self, error: BaseException) -> None:
        try:
            self.array.flush()
        except BaseException:
            pass
        payload = {
            "schema": SCHEMA,
            "status": "failed",
            "window_id": self.spec.window_id,
            "failed_unix_ns": time.time_ns(),
            "rows_completed": self.rows_written,
            "rows_expected": self.rows,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            _write_json_exclusive(self.failed, payload)
            _fsync_directory(self.root)
        except FileExistsError:
            pass


@dataclass
class _ActiveRequest:
    spec: _RequestSpec
    prompt_ids: tuple[int, ...]
    output_ids: list[int]
    artifact: _CaptureArtifact | None
    next_output_len: int = 0
    complete: bool = False


class ForcedDecodeCaptureLogitsProcessor(LogitsProcessor):
    """Capture unmodified real-vocabulary logits, then force the next token."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        _parse_request(sampling_params)

    def __init__(
        self, vllm_config: Any, device: torch.device, is_pin_memory: bool
    ) -> None:
        del device, is_pin_memory
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise RuntimeError("decode capture requires server max_num_seqs=1")
        if vllm_config.speculative_config is not None:
            raise RuntimeError("decode capture forbids speculative decoding")
        model_vocab = int(vllm_config.model_config.get_vocab_size())
        if model_vocab < REAL_VOCAB_SIZE:
            raise RuntimeError(
                f"model vocabulary {model_vocab} is narrower than {REAL_VOCAB_SIZE}"
            )
        self.root = _capture_root()
        self.allowed_window_ids = _allowed_window_ids()
        self.expected_outputs = _expected_outputs()
        self.tp_rank = int(get_tensor_model_parallel_rank())
        self.capture_on_this_rank = self.tp_rank == 0
        self.active: dict[int, _ActiveRequest] = {}
        self.batch_size = 0

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        if batch_update.batch_size > 1:
            raise RuntimeError("decode capture observed more than one active request")
        if batch_update.moved:
            raise RuntimeError("decode capture forbids request row moves")

        for index in batch_update.removed:
            state = self.active.get(index)
            if state is None:
                raise RuntimeError(f"decode capture removed unknown request row {index}")
            if not state.complete:
                error = RuntimeError(
                    f"decode capture request removed before completion at output "
                    f"length {len(state.output_ids)}"
                )
                if state.artifact is not None:
                    state.artifact.record_failure(error)
                raise error
            del self.active[index]

        if len(batch_update.added) > 1:
            raise RuntimeError("decode capture received multiple added requests")
        for index, params, prompt_ids, output_ids in batch_update.added:
            if index != 0 or self.active:
                raise RuntimeError("decode capture requires one request fixed at batch row zero")
            spec = _parse_request(params)
            if spec.window_id not in self.allowed_window_ids:
                raise RuntimeError("window allowlist changed between validation and execution")
            if len(spec.forced_token_ids) != self.expected_outputs:
                raise RuntimeError("expected output count changed between validation and execution")
            if prompt_ids is None or len(prompt_ids) != 1:
                raise RuntimeError("decode capture requires exactly one pretokenized prompt ID")
            prompt = tuple(int(token) for token in prompt_ids)
            if prompt[0] < 0 or prompt[0] >= REAL_VOCAB_SIZE:
                raise RuntimeError("prompt token is outside real vocabulary")
            if token_ids_sha256(prompt) != spec.prompt_sha256:
                raise RuntimeError("prompt token hash does not match request declaration")
            if output_ids:
                raise RuntimeError("new decode capture request already has output tokens")
            artifact = (
                _CaptureArtifact(self.root, spec, prompt, self.tp_rank)
                if self.capture_on_this_rank
                else None
            )
            self.active[index] = _ActiveRequest(
                spec=spec,
                prompt_ids=prompt,
                output_ids=output_ids,
                artifact=artifact,
            )
        self.batch_size = batch_update.batch_size
        if len(self.active) != self.batch_size:
            raise RuntimeError(
                f"decode capture state/batch mismatch: {len(self.active)} vs {self.batch_size}"
            )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.batch_size != 1 or set(self.active) != {0}:
            raise RuntimeError("decode capture apply requires exactly one active request")
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"decode capture expected logits [1,V], got {tuple(logits.shape)}")
        if logits.dtype != torch.float32:
            raise RuntimeError(f"decode capture expected incoming FP32 logits, got {logits.dtype}")
        width = int(logits.shape[1])
        if width < REAL_VOCAB_SIZE:
            raise RuntimeError(f"incoming logit width {width} is below real vocabulary")

        state = self.active[0]
        try:
            output_len = len(state.output_ids)
            if output_len != state.next_output_len:
                raise RuntimeError(
                    f"duplicate/skipped apply at output length {output_len}; "
                    f"expected {state.next_output_len}"
                )
            if output_len >= len(state.spec.forced_token_ids):
                raise RuntimeError("decode capture received logits after forced sequence ended")
            if tuple(state.output_ids) != state.spec.forced_token_ids[:output_len]:
                raise RuntimeError("generated output token prefix diverged from forced sequence")

            # Capture occurs before the first mutation of logits. The synchronous
            # D2H copy is intentional in this correctness-only service.
            if (
                state.artifact is not None
                and output_len >= state.spec.capture_start_output_len
            ):
                row_tensor = logits[0, :REAL_VOCAB_SIZE].detach().to(device="cpu")
                row = row_tensor.contiguous().numpy()
                state.artifact.write_row(
                    output_len - state.spec.capture_start_output_len,
                    row,
                    width,
                    str(logits.dtype),
                )

            target = state.spec.forced_token_ids[output_len]
            target_value = logits[0, target].clone()
            logits.fill_(float("-inf"))
            logits[0, target] = target_value
            state.next_output_len += 1
            if state.next_output_len == len(state.spec.forced_token_ids):
                if state.artifact is not None:
                    state.artifact.finalize()
                state.complete = True
            return logits
        except BaseException as error:
            if state.artifact is not None:
                state.artifact.record_failure(error)
            raise
