"""CPU-only contract tests for the isolated forced-decode capture processor."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"


class FakeValidationError(ValueError):
    pass


class FakeLogitsProcessor:
    pass


@dataclass(frozen=True)
class FakeBatchUpdate:
    batch_size: int
    removed: tuple[int, ...] = ()
    added: tuple[tuple, ...] = ()
    moved: tuple[tuple, ...] = ()


class FakeSamplingParams:
    def __init__(self, extra_args, max_tokens):
        self.extra_args = extra_args
        self.max_tokens = max_tokens
        self.n = 1
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 0
        self.min_p = 0.0
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.repetition_penalty = 1.0
        self.min_tokens = 0
        self.ignore_eos = True
        self.logprobs = None
        self.prompt_logprobs = None
        self.logprob_token_ids = None
        self.stop = []
        self.stop_token_ids = []
        self.bad_words = []
        self.allowed_token_ids = None
        self.logit_bias = None
        self.structured_outputs = None
        self.thinking_token_budget = None


def _install_vllm_stubs(monkeypatch):
    names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.exceptions",
        "vllm.sampling_params",
        "vllm.v1",
        "vllm.v1.sample",
        "vllm.v1.sample.logits_processor",
        "vllm.v1.sample.logits_processor.interface",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.distributed.parallel_state"].get_tensor_model_parallel_rank = lambda: 0
    modules["vllm.exceptions"].VLLMValidationError = FakeValidationError
    modules["vllm.sampling_params"].SamplingParams = FakeSamplingParams
    interface = modules["vllm.v1.sample.logits_processor.interface"]
    interface.BatchUpdate = FakeBatchUpdate
    interface.LogitsProcessor = FakeLogitsProcessor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(PATCH))
    for name in list(sys.modules):
        if name == "p8_decode_capture" or name.startswith("p8_decode_capture."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("p8_decode_capture.processor")


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS", "conditional-fit-0003"
    )
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS", "3")
    return _install_vllm_stubs(monkeypatch), tmp_path


def _params(module, *, start=0, forced=(2, 3, 4), window="conditional-fit-0003"):
    prompt = (1,)
    extra = {
        module.WINDOW_KEY: window,
        module.FORCED_KEY: list(forced),
        module.START_KEY: start,
        module.PROMPT_HASH_KEY: module.token_ids_sha256(prompt),
        module.FORCED_HASH_KEY: module.token_ids_sha256(forced),
    }
    return FakeSamplingParams(extra, max_tokens=len(forced)), prompt


def _config(max_num_seqs=1, speculative=None, vocab=154_880):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative,
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab),
    )


def _run_three_rows(module, processor, params, prompt, output_ids):
    processor.update_state(
        FakeBatchUpdate(1, added=((0, params, list(prompt), output_ids),))
    )
    originals = []
    for step, target in enumerate((2, 3, 4)):
        logits = torch.arange(154_882, dtype=torch.float32).reshape(1, -1) + step
        originals.append(logits[0, :154_880].clone().numpy())
        returned = processor.apply(logits)
        assert returned[0, target].isfinite()
        assert int(torch.isfinite(returned).sum()) == 1
        output_ids.append(target)
    return originals


def test_full_capture_is_pre_mask_complete_and_atomic(loaded):
    module, output = loaded
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    output_ids = []
    originals = _run_three_rows(module, processor, params, prompt, output_ids)

    raw = output / "conditional-fit-0003.logits.f32"
    metadata_path = output / "conditional-fit-0003.capture.json"
    assert raw.is_file() and metadata_path.is_file()
    assert not (output / "conditional-fit-0003.logits.f32.partial").exists()
    assert not (output / "conditional-fit-0003.capture.inprogress.json").exists()
    captured = np.memmap(raw, mode="r", dtype="<f4", shape=(3, 154_880))
    np.testing.assert_array_equal(captured, np.stack(originals))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete"
    assert metadata["shape"] == [3, 154_880]
    assert metadata["original_logit_width"] == 154_882
    assert metadata["original_logit_dtype"] == "torch.float32"
    assert metadata["causal_teacher_row_range"] == [0, 2]
    assert metadata["one_token_prefill_row_range"] == [0, 0]
    assert metadata["true_decode_row_range"] == [1, 2]
    assert metadata["raw_sha256"] == module.hashlib.sha256(raw.read_bytes()).hexdigest()
    assert metadata["sequence_token_ids_sha256"] == module.token_ids_sha256((1, 2, 3, 4))


def test_capture_start_writes_only_requested_tail_but_forces_all_rows(loaded):
    module, output = loaded
    params, prompt = _params(module, start=2)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    output_ids = []
    originals = _run_three_rows(module, processor, params, prompt, output_ids)
    raw = output / "conditional-fit-0003.logits.f32"
    captured = np.memmap(raw, mode="r", dtype="<f4", shape=(1, 154_880))
    np.testing.assert_array_equal(captured[0], originals[2])
    metadata = json.loads((output / "conditional-fit-0003.capture.json").read_text())
    assert metadata["shape"] == [1, 154_880]
    assert metadata["causal_teacher_row_range"] == [2, 2]
    assert metadata["one_token_prefill_row_range"] is None
    assert metadata["true_decode_row_range"] == [2, 2]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.extra_args.__setitem__("extra", 1), "requires exactly"),
        (lambda p: p.extra_args.__setitem__("p8_decode_capture_window_id", "../x"), "invalid"),
        (lambda p: p.extra_args.__setitem__("p8_decode_capture_window_id", "conditional-fit-9999"), "allowlist"),
        (lambda p: p.extra_args.__setitem__("p8_decode_capture_forced_token_ids", [2, 3]), "exactly 3"),
        (lambda p: p.extra_args.__setitem__("p8_decode_capture_forced_token_ids_sha256", "0" * 64), "does not match"),
        (lambda p: setattr(p, "max_tokens", 2), "max_tokens"),
        (lambda p: setattr(p, "ignore_eos", False), "ignore_eos"),
        (lambda p: setattr(p, "logprobs", -1), "logprobs"),
        (lambda p: setattr(p, "presence_penalty", 0.1), "presence_penalty"),
        (lambda p: setattr(p, "thinking_token_budget", 4), "thinking_token_budget"),
    ],
)
def test_request_validation_fails_closed(loaded, mutate, match):
    module, _ = loaded
    params, _ = _params(module)
    mutate(params)
    with pytest.raises(FakeValidationError, match=match):
        module.ForcedDecodeCaptureLogitsProcessor.validate_params(params)


@pytest.mark.parametrize(
    "config,match",
    [
        (_config(max_num_seqs=2), "max_num_seqs=1"),
        (_config(speculative=object()), "speculative"),
        (_config(vocab=154_879), "narrower"),
    ],
)
def test_server_configuration_fails_closed(loaded, config, match):
    module, _ = loaded
    with pytest.raises(RuntimeError, match=match):
        module.ForcedDecodeCaptureLogitsProcessor(config, torch.device("cpu"), False)


def test_prompt_hash_duplicate_apply_and_nonfinite_leave_failure_receipt(loaded):
    module, output = loaded
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    bad_prompt = [prompt[0] + 1]
    with pytest.raises(RuntimeError, match="prompt token hash"):
        processor.update_state(FakeBatchUpdate(1, added=((0, params, bad_prompt, []),)))

    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    output_ids = []
    processor.update_state(FakeBatchUpdate(1, added=((0, params, list(prompt), output_ids),)))
    first = torch.zeros((1, 154_880), dtype=torch.float32)
    processor.apply(first)
    with pytest.raises(RuntimeError, match="duplicate/skipped"):
        processor.apply(torch.zeros_like(first))
    failed = json.loads((output / "conditional-fit-0003.capture.failed.json").read_text())
    assert failed["status"] == "failed"
    assert failed["rows_completed"] == 1


def test_nonfinite_and_wrong_geometry_are_rejected_before_mutation(loaded):
    module, _ = loaded
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    processor.update_state(FakeBatchUpdate(1, added=((0, params, list(prompt), []),)))
    logits = torch.zeros((1, 154_880), dtype=torch.float32)
    logits[0, 7] = float("nan")
    before = logits.clone()
    with pytest.raises(RuntimeError, match="nonfinite"):
        processor.apply(logits)
    torch.testing.assert_close(logits, before, equal_nan=True)


@pytest.mark.parametrize(
    "logits,match",
    [
        (torch.zeros((2, 154_880), dtype=torch.float32), r"\[1,V\]"),
        (torch.zeros((1, 154_879), dtype=torch.float32), "below real vocabulary"),
        (torch.zeros((1, 154_880), dtype=torch.float64), "incoming FP32"),
    ],
)
def test_logit_geometry_and_dtype_fail_closed(loaded, logits, match):
    module, _ = loaded
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    processor.update_state(FakeBatchUpdate(1, added=((0, params, list(prompt), []),)))
    before = logits.clone()
    with pytest.raises(RuntimeError, match=match):
        processor.apply(logits)
    torch.testing.assert_close(logits, before)


def test_incomplete_removal_preserves_visible_failure(loaded):
    module, output = loaded
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    output_ids = []
    processor.update_state(FakeBatchUpdate(1, added=((0, params, list(prompt), output_ids),)))
    processor.apply(torch.zeros((1, 154_880), dtype=torch.float32))
    output_ids.append(2)
    with pytest.raises(RuntimeError, match="removed before completion"):
        processor.update_state(FakeBatchUpdate(0, removed=(0,)))
    failure = json.loads((output / "conditional-fit-0003.capture.failed.json").read_text())
    assert failure["rows_completed"] == 1 and failure["rows_expected"] == 3
    assert (output / "conditional-fit-0003.logits.f32.partial").is_file()
    assert not (output / "conditional-fit-0003.capture.json").exists()


def test_nonzero_tp_rank_forces_without_writing(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS", "conditional-fit-0003"
    )
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS", "3")
    module = _install_vllm_stubs(monkeypatch)
    module.get_tensor_model_parallel_rank = lambda: 1
    params, prompt = _params(module)
    processor = module.ForcedDecodeCaptureLogitsProcessor(
        _config(), torch.device("cpu"), False
    )
    output_ids = []
    _run_three_rows(module, processor, params, prompt, output_ids)
    assert list(tmp_path.iterdir()) == []
