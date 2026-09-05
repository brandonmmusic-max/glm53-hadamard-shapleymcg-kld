"""Forced decode capture seam for vLLM Model Runner V2 at 7f1e92bec.

Unlike :mod:`p8_decode_capture.processor`, this helper is not a custom logits
processor. The pinned vLLM revision rejects those under Model Runner V2. A
three-call source patch in ``sampler-v2-hook.patch`` owns this helper directly
from the V2 sampler, outside the model CUDA graph.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.sampling_params import SamplingParams

from .processor import (
    REAL_VOCAB_SIZE,
    _CaptureArtifact,
    _RequestSpec,
    _allowed_window_ids,
    _capture_root,
    _expected_outputs,
    _parse_request,
    token_ids_sha256,
)


ENABLE_ENV = "GLM53_P8_DECODE_CAPTURE_V2"
ALLOW_NONZERO_START_ENV = "GLM53_P8_DECODE_CAPTURE_ALLOW_NONZERO_START"
_DUMMY_REQ_RE = re.compile(r"req_[0-9]+_[0-9a-f-]+\Z")


def _enabled() -> bool:
    raw = os.environ.get(ENABLE_ENV, "").strip().lower()
    if not raw:
        return False
    if raw not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid {ENABLE_ENV}={raw!r}")
    return True


def _allow_nonzero_start() -> bool:
    raw = os.environ.get(ALLOW_NONZERO_START_ENV, "").strip().lower()
    if not raw:
        return False
    if raw not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid {ALLOW_NONZERO_START_ENV}={raw!r}")
    return True


@dataclass
class _V2ActiveRequest:
    req_idx: int
    spec: _RequestSpec
    artifact: _CaptureArtifact | None = None
    prompt_ids: tuple[int, ...] | None = None
    next_position: int = 0
    complete: bool = False


@dataclass
class _PinnedWarmupState:
    expected_prompt_len: int
    registrations: int = 0
    samples: int = 0


class V2ForcedDecodeCapture:
    """Capture and force one real request while allowing only proven dummy runs."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        vocab_size: int,
        num_speculative_tokens: int,
        req_states: Any,
    ) -> None:
        self.enabled = _enabled()
        self.req_states = req_states
        self.states: dict[int, _V2ActiveRequest] = {}
        self._pinned_warmup: _PinnedWarmupState | None = None
        self._pinned_warmup_terminal: str | None = None
        if not self.enabled:
            return
        if max_num_reqs != 1:
            raise RuntimeError("V2 decode capture requires max_num_seqs=1")
        if vocab_size < REAL_VOCAB_SIZE:
            raise RuntimeError(
                f"model vocabulary {vocab_size} is narrower than {REAL_VOCAB_SIZE}"
            )
        if num_speculative_tokens != 1:
            raise RuntimeError("V2 decode capture forbids speculative or multi-token decode")
        self.root = _capture_root()
        self.allowed_window_ids = _allowed_window_ids()
        self.expected_outputs = _expected_outputs()
        self.tp_rank = int(get_tensor_model_parallel_rank())
        self.capture_on_this_rank = self.tp_rank == 0
        self.allow_nonzero_start = _allow_nonzero_start()
        print(
            "GLM53_P8_DECODE_CAPTURE_V2_READY "
            f"tp_rank={self.tp_rank} max_num_reqs={max_num_reqs} "
            f"real_vocab={REAL_VOCAB_SIZE} expected_outputs={self.expected_outputs}",
            flush=True,
        )

    @contextmanager
    def pinned_startup_warmup_scope(
        self, *, expected_prompt_len: int
    ) -> Iterator[None]:
        """Permit only the pinned scheduler-realistic startup warmup.

        The scope is entered lexically by the patched ``warmup_kernels``
        function. Request names cannot enable it, and every real-request guard
        remains unchanged outside this context.
        """

        if not self.enabled:
            yield
            return
        if self._pinned_warmup is not None:
            raise RuntimeError("nested V2 pinned warmup scope")
        if self._pinned_warmup_terminal is not None:
            raise RuntimeError(
                "V2 pinned warmup scope cannot be replayed after "
                f"{self._pinned_warmup_terminal}"
            )
        if expected_prompt_len < 2:
            raise RuntimeError("pinned V2 warmup prompt must contain at least two tokens")
        if self.states:
            raise RuntimeError("V2 pinned warmup started with capture state")
        if (
            getattr(self.req_states, "num_reqs", None) != 0
            or getattr(self.req_states, "req_id_to_index", None) != {}
            or getattr(self.req_states, "index_to_req_id", None) != {}
        ):
            raise RuntimeError("V2 pinned warmup started with nonempty request state")

        state = _PinnedWarmupState(expected_prompt_len=expected_prompt_len)
        self._pinned_warmup = state
        try:
            yield
            if state.registrations != 1 or state.samples != 2:
                raise RuntimeError(
                    "pinned V2 warmup lifecycle mismatch: "
                    f"registrations={state.registrations} samples={state.samples}"
                )
            if self.states:
                raise RuntimeError("V2 pinned warmup created capture state")
            if (
                getattr(self.req_states, "num_reqs", None) != 0
                or getattr(self.req_states, "req_id_to_index", None) != {}
                or getattr(self.req_states, "index_to_req_id", None) != {}
            ):
                raise RuntimeError("V2 pinned warmup left request state registered")
        except BaseException:
            self._pinned_warmup_terminal = "failure"
            raise
        else:
            self._pinned_warmup_terminal = "complete"
            print(
                "GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED "
                f"tp_rank={self.tp_rank} registrations=1 samples=2",
                flush=True,
            )
        finally:
            self._pinned_warmup = None

    @staticmethod
    def _is_pinned_warmup_sampling_params(sampling_params: SamplingParams) -> bool:
        expected = {
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "repetition_penalty": 1.2,
            "min_tokens": 2,
            "logit_bias": {0: -1.0, 1: 0.5},
            "_bad_words_token_ids": [[0], [1, 2]],
            "logprobs": 5,
            "prompt_logprobs": 1,
        }
        return getattr(sampling_params, "extra_args", None) is None and all(
            getattr(sampling_params, name, None) == value
            for name, value in expected.items()
        )

    def _register_pinned_warmup(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        state = self._pinned_warmup
        if state is None:
            raise AssertionError("pinned warmup registration outside scope")
        if state.registrations != 0 or self.states:
            raise RuntimeError("pinned V2 warmup registered more than one request")
        if req_idx != 0 or prompt_len != state.expected_prompt_len:
            raise RuntimeError(
                "pinned V2 warmup request geometry mismatch: "
                f"req_idx={req_idx} prompt_len={prompt_len}"
            )
        if (
            getattr(self.req_states, "num_reqs", None) != 1
            or getattr(self.req_states, "req_id_to_index", None)
            != {"_warmup_0_": 0}
            or getattr(self.req_states, "index_to_req_id", None)
            != {0: "_warmup_0_"}
        ):
            raise RuntimeError("pinned V2 warmup request identity mismatch")
        if not self._is_pinned_warmup_sampling_params(sampling_params):
            raise RuntimeError("pinned V2 warmup sampling parameters changed")
        state.registrations += 1

    def _sample_pinned_warmup(
        self, logits: torch.Tensor, input_batch: Any
    ) -> torch.Tensor:
        state = self._pinned_warmup
        if state is None:
            raise AssertionError("pinned warmup sample outside scope")
        if state.registrations != 1 or state.samples >= 2 or self.states:
            raise RuntimeError("pinned V2 warmup sampler lifecycle mismatch")
        if (
            getattr(self.req_states, "num_reqs", None) != 1
            or getattr(self.req_states, "req_id_to_index", None)
            != {"_warmup_0_": 0}
            or getattr(self.req_states, "index_to_req_id", None)
            != {0: "_warmup_0_"}
            or getattr(input_batch, "req_ids", None) != ["_warmup_0_"]
            or getattr(input_batch, "num_reqs", None) != 1
            or getattr(input_batch, "num_draft_tokens", None) != 0
            or logits.ndim != 2
            or logits.shape[0] != 1
            or int(logits.shape[1]) < REAL_VOCAB_SIZE
        ):
            raise RuntimeError("pinned V2 warmup sample identity/geometry mismatch")
        mapping = np.asarray(input_batch.idx_mapping_np)
        if mapping.shape != (1,) or int(mapping[0]) != 0:
            raise RuntimeError("pinned V2 warmup sample mapping mismatch")
        indices = input_batch.logits_indices
        if indices.numel() != 1 or bool(input_batch.is_padding[indices][0].item()):
            raise RuntimeError("pinned V2 warmup selected an invalid logit row")

        expected_query_len = state.expected_prompt_len if state.samples == 0 else 1
        expected_position = state.expected_prompt_len - 1 + state.samples
        position = int(input_batch.positions[indices][0].item())
        if (
            getattr(input_batch, "max_query_len", None) != expected_query_len
            or getattr(input_batch, "num_tokens", None) != expected_query_len
            or position != expected_position
        ):
            raise RuntimeError(
                "pinned V2 warmup sample sequence mismatch: "
                f"max_query_len={getattr(input_batch, 'max_query_len', None)} "
                f"num_tokens={getattr(input_batch, 'num_tokens', None)} "
                f"position={position}"
            )
        if state.samples == 0:
            input_token = int(input_batch.input_ids[indices][0].item())
            if input_token != state.expected_prompt_len - 1:
                raise RuntimeError("pinned V2 warmup prompt token sequence changed")
        state.samples += 1
        return logits

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        if not self.enabled:
            return
        if self._pinned_warmup_terminal == "failure":
            raise RuntimeError("V2 pinned warmup failure poisoned capture helper")
        if self._pinned_warmup is not None:
            self._register_pinned_warmup(req_idx, prompt_len, sampling_params)
            return
        if req_idx != 0 or prompt_len != 1:
            raise RuntimeError("V2 decode capture requires one prompt token at request row zero")
        prior = self.states.get(req_idx)
        if prior is not None and not prior.complete:
            error = RuntimeError("new V2 request replaced an incomplete decode capture")
            if prior.artifact is not None:
                prior.artifact.record_failure(error)
            raise error
        spec = _parse_request(sampling_params)
        if spec.window_id not in self.allowed_window_ids:
            raise RuntimeError("window allowlist changed between validation and V2 execution")
        if len(spec.forced_token_ids) != self.expected_outputs:
            raise RuntimeError("expected output count changed before V2 execution")
        if spec.capture_start_output_len != 0 and not self.allow_nonzero_start:
            raise RuntimeError(
                f"nonzero capture start requires explicit {ALLOW_NONZERO_START_ENV}=1"
            )
        self.states[req_idx] = _V2ActiveRequest(req_idx=req_idx, spec=spec)

    def _proven_dummy_batch(self, input_batch: Any) -> bool:
        if getattr(self.req_states, "num_reqs", None) != 0:
            return False
        req_ids = getattr(input_batch, "req_ids", None)
        num_reqs = getattr(input_batch, "num_reqs", None)
        if (
            not isinstance(req_ids, list)
            or not isinstance(num_reqs, int)
            or num_reqs < 1
            or len(req_ids) != num_reqs
            or not all(_DUMMY_REQ_RE.fullmatch(str(req_id)) for req_id in req_ids)
        ):
            return False
        num_tokens = getattr(input_batch, "num_tokens", None)
        is_padding = getattr(input_batch, "is_padding", None)
        if not isinstance(num_tokens, int) or num_tokens < num_reqs or is_padding is None:
            return False
        return bool(torch.all(is_padding[:num_tokens]).item())

    def capture_and_force(self, logits: torch.Tensor, input_batch: Any) -> torch.Tensor:
        if not self.enabled:
            return logits
        if self._pinned_warmup_terminal == "failure":
            raise RuntimeError("V2 pinned warmup failure poisoned capture helper")
        if self._pinned_warmup is not None:
            return self._sample_pinned_warmup(logits, input_batch)
        if not self.states:
            if self._proven_dummy_batch(input_batch):
                return logits
            raise RuntimeError("V2 sampler ran without a registered capture request")
        if (
            getattr(input_batch, "num_reqs", None) != 1
            or getattr(input_batch, "num_tokens", None) != 1
            or getattr(input_batch, "max_query_len", None) != 1
            or getattr(input_batch, "num_draft_tokens", None) != 0
            or getattr(input_batch, "has_structured_output_reqs", None) is not False
            or logits.ndim != 2
            or logits.shape[0] != 1
            or int(logits.shape[1]) < REAL_VOCAB_SIZE
        ):
            raise RuntimeError(
                f"invalid V2 capture batch/logit geometry: "
                f"num_reqs={getattr(input_batch, 'num_reqs', None)} "
                f"draft={getattr(input_batch, 'num_draft_tokens', None)} "
                f"logits={tuple(logits.shape)}"
            )
        mapping = np.asarray(input_batch.idx_mapping_np)
        if mapping.shape != (1,):
            raise RuntimeError(f"invalid V2 request mapping shape {mapping.shape}")
        req_idx = int(mapping[0])
        if set(self.states) != {req_idx}:
            raise RuntimeError("V2 sampler request mapping does not match capture state")
        req_ids = getattr(input_batch, "req_ids", None)
        if not isinstance(req_ids, list) or len(req_ids) != 1:
            raise RuntimeError("V2 capture requires exactly one concrete request ID")
        req_id = req_ids[0]
        if (
            getattr(self.req_states, "req_id_to_index", {}).get(req_id) != req_idx
            or getattr(self.req_states, "index_to_req_id", {}).get(req_idx) != req_id
        ):
            raise RuntimeError("V2 request identity maps do not match the sampled row")

        state = self.states[req_idx]
        try:
            indices = input_batch.logits_indices
            if indices.numel() != 1:
                raise RuntimeError("V2 capture requires exactly one sampled logit index")
            if bool(input_batch.is_padding[indices][0].item()):
                raise RuntimeError("V2 capture selected a padded logit row")
            position = int(input_batch.positions[indices][0].item())
            input_token = int(input_batch.input_ids[indices][0].item())
            if position != state.next_position:
                raise RuntimeError(
                    f"duplicate/skipped V2 causal position {position}; "
                    f"expected {state.next_position}"
                )
            if position >= len(state.spec.forced_token_ids):
                raise RuntimeError("V2 capture received logits after forced sequence ended")

            if position == 0:
                prompt = (input_token,)
                if token_ids_sha256(prompt) != state.spec.prompt_sha256:
                    raise RuntimeError("V2 prompt token hash does not match declaration")
                state.prompt_ids = prompt
                if self.capture_on_this_rank:
                    state.artifact = _CaptureArtifact(
                        self.root, state.spec, prompt, self.tp_rank
                    )
            else:
                expected_input = state.spec.forced_token_ids[position - 1]
                if input_token != expected_input:
                    raise RuntimeError(
                        f"V2 consumed token {input_token} at position {position}; "
                        f"expected forced token {expected_input}"
                    )

            if state.prompt_ids is None:
                raise RuntimeError("V2 capture has no validated prompt identity")
            if (
                state.artifact is not None
                and position >= state.spec.capture_start_output_len
            ):
                # Store the FP32 representation of the untouched model logits.
                # D2H synchronization is confined to this correctness endpoint.
                row = (
                    logits[0, :REAL_VOCAB_SIZE]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .numpy()
                )
                state.artifact.write_row(
                    position - state.spec.capture_start_output_len,
                    row,
                    int(logits.shape[1]),
                    str(logits.dtype),
                )

            target = state.spec.forced_token_ids[position]
            target_value = logits[0, target].clone()
            if not bool(torch.isfinite(target_value).item()):
                raise RuntimeError(f"nonfinite forced-token logit at V2 position {position}")
            logits.fill_(float("-inf"))
            logits[0, target] = target_value
            state.next_position += 1
            if state.next_position == len(state.spec.forced_token_ids):
                if state.artifact is not None:
                    state.artifact.finalize()
                state.complete = True
                if self.capture_on_this_rank:
                    print(
                        "GLM53_P8_DECODE_CAPTURE_V2_COMPLETE "
                        f"window={state.spec.window_id} rows="
                        f"{len(state.spec.forced_token_ids) - state.spec.capture_start_output_len} "
                        f"tp_rank={self.tp_rank}",
                        flush=True,
                    )
            return logits
        except BaseException as error:
            if state.artifact is not None:
                state.artifact.record_failure(error)
            raise
