"""CPU tests for the isolated Model Runner V2 capture seam."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "runtime_patch"


class FakeValidationError(ValueError):
    pass


class FakeLogitsProcessor:
    pass


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
        self._bad_words_token_ids = None
        self.allowed_token_ids = None
        self.logit_bias = None
        self.structured_outputs = None
        self.thinking_token_budget = None


def _install_stubs(monkeypatch, tp_rank=0):
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
    modules["vllm.distributed.parallel_state"].get_tensor_model_parallel_rank = (
        lambda: tp_rank
    )
    modules["vllm.exceptions"].VLLMValidationError = FakeValidationError
    modules["vllm.sampling_params"].SamplingParams = FakeSamplingParams
    modules["vllm.v1.sample.logits_processor.interface"].BatchUpdate = object
    modules["vllm.v1.sample.logits_processor.interface"].LogitsProcessor = (
        FakeLogitsProcessor
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(PATCH_ROOT))
    for name in list(sys.modules):
        if name == "p8_decode_capture" or name.startswith("p8_decode_capture."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    processor = importlib.import_module("p8_decode_capture.processor")
    v2 = importlib.import_module("p8_decode_capture.v2_hook")
    return processor, v2


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_V2", "1")
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS", "conditional-fit-0003"
    )
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS", "3")
    processor, v2 = _install_stubs(monkeypatch)
    return processor, v2, tmp_path


def _params(processor, forced=(2, 3, 4), start=0):
    prompt = (1,)
    extra = {
        processor.WINDOW_KEY: "conditional-fit-0003",
        processor.FORCED_KEY: list(forced),
        processor.START_KEY: start,
        processor.PROMPT_HASH_KEY: processor.token_ids_sha256(prompt),
        processor.FORCED_HASH_KEY: processor.token_ids_sha256(forced),
    }
    return FakeSamplingParams(extra, len(forced))


def _warmup_params():
    params = FakeSamplingParams(None, 16)
    params.temperature = 0.9
    params.top_p = 0.9
    params.top_k = 50
    params.min_p = 0.1
    params.frequency_penalty = 0.5
    params.presence_penalty = 0.5
    params.repetition_penalty = 1.2
    params.min_tokens = 2
    params.logit_bias = {0: -1.0, 1: 0.5}
    params._bad_words_token_ids = [[0], [1, 2]]
    params.logprobs = 5
    params.prompt_logprobs = 1
    return params


def _input(position, token, *, req_id="cmpl-proof"):
    return SimpleNamespace(
        req_ids=[req_id],
        num_reqs=1,
        num_tokens=1,
        max_query_len=1,
        num_draft_tokens=0,
        has_structured_output_reqs=False,
        idx_mapping_np=np.array([0], dtype=np.intp),
        logits_indices=torch.tensor([0], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
        input_ids=torch.tensor([token], dtype=torch.int32),
        is_padding=torch.tensor([False]),
    )


def _warmup_input(sample, *, prompt_len=2, req_id="_warmup_0_"):
    batch = _input(
        prompt_len - 1 + sample,
        prompt_len - 1 if sample == 0 else 17,
        req_id=req_id,
    )
    batch.max_query_len = prompt_len if sample == 0 else 1
    batch.num_tokens = prompt_len if sample == 0 else 1
    return batch


def _helper(v2, *, req_states=None, max_reqs=1, vocab=154_880, spec=1):
    if req_states is None:
        req_states = SimpleNamespace(
            num_reqs=1,
            req_id_to_index={"cmpl-proof": 0},
            index_to_req_id={0: "cmpl-proof"},
        )
    return v2.V2ForcedDecodeCapture(
        max_num_reqs=max_reqs,
        vocab_size=vocab,
        num_speculative_tokens=spec,
        req_states=req_states,
    )


def test_v2_captures_before_mask_and_validates_consumed_history(loaded):
    processor, v2, output = loaded
    helper = _helper(v2)
    helper.add_request(0, 1, _params(processor))
    inputs = [1, 2, 3]
    originals = []
    for position, (input_token, target) in enumerate(zip(inputs, (2, 3, 4))):
        logits = (
            torch.arange(154_882, dtype=torch.float32).reshape(1, -1) + position
        )
        originals.append(logits[0, :154_880].clone().numpy())
        result = helper.capture_and_force(logits, _input(position, input_token))
        assert result[0, target].isfinite()
        assert int(torch.isfinite(result).sum()) == 1

    raw = np.memmap(
        output / "conditional-fit-0003.logits.f32",
        mode="r",
        dtype="<f4",
        shape=(3, 154_880),
    )
    np.testing.assert_array_equal(raw, np.stack(originals))
    metadata = json.loads((output / "conditional-fit-0003.capture.json").read_text())
    assert metadata["causal_teacher_row_range"] == [0, 2]
    assert metadata["original_logit_dtype"] == "torch.float32"
    assert metadata["raw_sha256"] == processor.hashlib.sha256(
        (output / "conditional-fit-0003.logits.f32").read_bytes()
    ).hexdigest()


def test_v2_converts_untouched_bfloat16_logits_to_stored_float32(
    loaded, monkeypatch
):
    processor, v2, output = loaded
    monkeypatch.setenv(v2.ALLOW_NONZERO_START_ENV, "1")
    helper = _helper(v2)
    helper.add_request(0, 1, _params(processor, forced=(2, 3, 4), start=2))
    for position, input_token in enumerate((1, 2, 3)):
        logits = torch.arange(154_880, dtype=torch.bfloat16).reshape(1, -1)
        expected = logits[0].to(torch.float32).numpy().copy()
        helper.capture_and_force(logits, _input(position, input_token))
    raw = np.memmap(
        output / "conditional-fit-0003.logits.f32",
        mode="r",
        dtype="<f4",
        shape=(1, 154_880),
    )
    np.testing.assert_array_equal(raw[0], expected)
    metadata = json.loads((output / "conditional-fit-0003.capture.json").read_text())
    assert metadata["original_logit_dtype"] == "torch.bfloat16"


def test_v2_nonzero_capture_start_is_rejected_without_explicit_canary_opt_in(
    loaded,
):
    processor, v2, _ = loaded
    helper = _helper(v2)
    with pytest.raises(RuntimeError, match="nonzero capture start"):
        helper.add_request(0, 1, _params(processor, start=2))


def test_only_proven_empty_state_dummy_batch_passes_through(loaded):
    _, v2, _ = loaded
    helper = _helper(v2, req_states=SimpleNamespace(num_reqs=0))
    logits = torch.ones((1, 154_880), dtype=torch.float32)
    dummy = _input(0, 0, req_id="req_0_abcdef01")
    dummy.is_padding.fill_(True)
    assert helper.capture_and_force(logits, dummy) is logits
    for mutate in (
        lambda batch: setattr(batch, "req_ids", ["cmpl-real"]),
        lambda batch: batch.is_padding.fill_(False),
    ):
        changed = _input(0, 0, req_id="req_0_abcdef01")
        changed.is_padding.fill_(True)
        mutate(changed)
        with pytest.raises(RuntimeError, match="without a registered"):
            helper.capture_and_force(logits.clone(), changed)

    real_state_helper = _helper(
        v2,
        req_states=SimpleNamespace(
            num_reqs=1,
            req_id_to_index={"cmpl-proof": 0},
            index_to_req_id={0: "cmpl-proof"},
        ),
    )
    with pytest.raises(RuntimeError, match="without a registered"):
        real_state_helper.capture_and_force(logits.clone(), dummy)


@pytest.mark.parametrize(
    "max_reqs,vocab,spec,match",
    [
        (2, 154_880, 1, "max_num_seqs=1"),
        (1, 154_879, 1, "narrower"),
        (1, 154_880, 2, "speculative"),
    ],
)
def test_v2_server_shape_fails_closed(loaded, max_reqs, vocab, spec, match):
    _, v2, _ = loaded
    with pytest.raises(RuntimeError, match=match):
        _helper(v2, max_reqs=max_reqs, vocab=vocab, spec=spec)


@pytest.mark.parametrize(
    "position,input_token,match",
    [
        (1, 1, "duplicate/skipped"),
        (0, 9, "prompt token hash"),
    ],
)
def test_v2_rejects_position_or_prompt_before_mask(loaded, position, input_token, match):
    processor, v2, _ = loaded
    helper = _helper(v2)
    helper.add_request(0, 1, _params(processor))
    logits = torch.zeros((1, 154_880), dtype=torch.float32)
    before = logits.clone()
    with pytest.raises(RuntimeError, match=match):
        helper.capture_and_force(logits, _input(position, input_token))
    torch.testing.assert_close(logits, before)


def test_v2_rejects_wrong_consumed_forced_token_and_preserves_failure(loaded):
    processor, v2, output = loaded
    helper = _helper(v2)
    helper.add_request(0, 1, _params(processor))
    helper.capture_and_force(
        torch.zeros((1, 154_880), dtype=torch.float32), _input(0, 1)
    )
    logits = torch.ones((1, 154_880), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="expected forced token"):
        helper.capture_and_force(logits, _input(1, 99))
    failure = json.loads((output / "conditional-fit-0003.capture.failed.json").read_text())
    assert failure["rows_completed"] == 1
    assert not (output / "conditional-fit-0003.capture.json").exists()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda batch, states: setattr(batch, "num_tokens", 2),
            "invalid V2 capture batch/logit geometry",
        ),
        (
            lambda batch, states: setattr(batch, "max_query_len", 2),
            "invalid V2 capture batch/logit geometry",
        ),
        (
            lambda batch, states: setattr(
                batch, "has_structured_output_reqs", True
            ),
            "invalid V2 capture batch/logit geometry",
        ),
        (
            lambda batch, states: batch.is_padding.fill_(True),
            "selected a padded logit row",
        ),
        (
            lambda batch, states: states.req_id_to_index.update(
                {"cmpl-proof": 7}
            ),
            "identity maps",
        ),
        (
            lambda batch, states: states.index_to_req_id.update(
                {0: "cmpl-remapped"}
            ),
            "identity maps",
        ),
    ],
)
def test_v2_real_batch_scheduler_seams_fail_closed_before_mask(
    loaded, mutate, match
):
    processor, v2, _ = loaded
    req_states = SimpleNamespace(
        num_reqs=1,
        req_id_to_index={"cmpl-proof": 0},
        index_to_req_id={0: "cmpl-proof"},
    )
    helper = _helper(v2, req_states=req_states)
    helper.add_request(0, 1, _params(processor))
    batch = _input(0, 1)
    mutate(batch, req_states)
    logits = torch.zeros((1, 154_880), dtype=torch.float32)
    before = logits.clone()
    with pytest.raises(RuntimeError, match=match):
        helper.capture_and_force(logits, batch)
    torch.testing.assert_close(logits, before)


def test_v2_nonzero_tp_rank_forces_but_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_V2", "1")
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS", "conditional-fit-0003"
    )
    monkeypatch.setenv("GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS", "3")
    processor, v2 = _install_stubs(monkeypatch, tp_rank=1)
    helper = _helper(v2)
    helper.add_request(0, 1, _params(processor))
    for position, (input_token, target) in enumerate(zip((1, 2, 3), (2, 3, 4))):
        logits = torch.ones((1, 154_880), dtype=torch.float32)
        helper.capture_and_force(logits, _input(position, input_token))
        assert int(torch.isfinite(logits).sum()) == 1
        assert logits[0, target].isfinite()
    assert list(tmp_path.iterdir()) == []


def test_disabled_v2_helper_is_strict_noop(monkeypatch):
    monkeypatch.delenv("GLM53_P8_DECODE_CAPTURE_V2", raising=False)
    _, v2 = _install_stubs(monkeypatch)
    helper = v2.V2ForcedDecodeCapture(
        max_num_reqs=8,
        vocab_size=1,
        num_speculative_tokens=4,
        req_states=SimpleNamespace(num_reqs=7),
    )
    logits = torch.ones((3, 5))
    assert helper.capture_and_force(logits, object()) is logits
    helper.add_request(9, 99, object())

    with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
        pass


def _empty_req_states():
    return SimpleNamespace(
        num_reqs=0,
        req_id_to_index={},
        index_to_req_id={},
    )


def _register_warmup_request(req_states):
    req_states.num_reqs = 1
    req_states.req_id_to_index["_warmup_0_"] = 0
    req_states.index_to_req_id[0] = "_warmup_0_"


def _finish_warmup_request(req_states):
    req_states.num_reqs = 0
    req_states.req_id_to_index.clear()
    req_states.index_to_req_id.clear()


def test_pinned_warmup_scope_allows_exact_lifecycle_once(
    loaded, capsys
):
    _, v2, output = loaded
    req_states = _empty_req_states()
    helper = _helper(v2, req_states=req_states)
    with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
        _register_warmup_request(req_states)
        helper.add_request(0, 2, _warmup_params())
        for sample in range(2):
            logits = torch.arange(154_880, dtype=torch.float32).reshape(1, -1)
            before = logits.clone()
            assert helper.capture_and_force(logits, _warmup_input(sample)) is logits
            torch.testing.assert_close(logits, before)
        _finish_warmup_request(req_states)

    assert helper._pinned_warmup is None
    assert helper._pinned_warmup_terminal == "complete"
    assert list(output.iterdir()) == []
    assert (
        "GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED "
        "tp_rank=0 registrations=1 samples=2"
    ) in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="cannot be replayed after complete"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            pass


def test_pinned_warmup_exception_resets_and_permanently_poison_scope(loaded):
    _, v2, _ = loaded
    helper = _helper(v2, req_states=_empty_req_states())
    with pytest.raises(ValueError, match="synthetic failure"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            raise ValueError("synthetic failure")
    assert helper._pinned_warmup is None
    assert helper._pinned_warmup_terminal == "failure"
    with pytest.raises(RuntimeError, match="cannot be replayed after failure"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            pass
    with pytest.raises(RuntimeError, match="poisoned capture helper"):
        helper.add_request(0, 1, object())
    with pytest.raises(RuntimeError, match="poisoned capture helper"):
        helper.capture_and_force(torch.ones((1, 154_880)), object())


def test_real_capture_guards_and_path_are_unchanged_after_warmup(loaded):
    processor, v2, output = loaded
    req_states = _empty_req_states()
    helper = _helper(v2, req_states=req_states)
    with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
        _register_warmup_request(req_states)
        helper.add_request(0, 2, _warmup_params())
        for sample in range(2):
            helper.capture_and_force(
                torch.ones((1, 154_880)), _warmup_input(sample)
            )
        _finish_warmup_request(req_states)

    req_states.num_reqs = 1
    req_states.req_id_to_index["cmpl-proof"] = 0
    req_states.index_to_req_id[0] = "cmpl-proof"
    with pytest.raises(RuntimeError, match="one prompt token"):
        helper.add_request(0, 2, _params(processor))
    helper.add_request(0, 1, _params(processor))
    for position, input_token in enumerate((1, 2, 3)):
        helper.capture_and_force(
            torch.ones((1, 154_880)), _input(position, input_token)
        )
    assert (output / "conditional-fit-0003.capture.json").is_file()


def test_pinned_warmup_rejects_nested_or_incomplete_lifecycle(loaded):
    _, v2, _ = loaded
    helper = _helper(v2, req_states=_empty_req_states())
    with pytest.raises(RuntimeError, match="nested"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
                pass
    assert helper._pinned_warmup_terminal == "failure"

    helper = _helper(v2, req_states=_empty_req_states())
    with pytest.raises(RuntimeError, match="lifecycle mismatch"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            pass
    assert helper._pinned_warmup is None
    assert helper._pinned_warmup_terminal == "failure"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda req_states, params: req_states.req_id_to_index.clear(),
            "identity mismatch",
        ),
        (
            lambda req_states, params: setattr(params, "extra_args", {"x": 1}),
            "sampling parameters changed",
        ),
        (
            lambda req_states, params: setattr(params, "top_k", 49),
            "sampling parameters changed",
        ),
    ],
)
def test_pinned_warmup_registration_is_exact(loaded, mutate, match):
    _, v2, _ = loaded
    req_states = _empty_req_states()
    helper = _helper(v2, req_states=req_states)
    params = _warmup_params()
    with pytest.raises(RuntimeError, match=match):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            _register_warmup_request(req_states)
            mutate(req_states, params)
            helper.add_request(0, 2, params)
    assert helper._pinned_warmup is None
    assert helper._pinned_warmup_terminal == "failure"


def test_pinned_warmup_rejects_wrong_prompt_geometry(loaded):
    _, v2, _ = loaded
    req_states = _empty_req_states()
    helper = _helper(v2, req_states=req_states)
    with pytest.raises(RuntimeError, match="request geometry mismatch"):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            _register_warmup_request(req_states)
            helper.add_request(0, 3, _warmup_params())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda batch: setattr(batch, "idx_mapping_np", np.array([1])),
            "mapping mismatch",
        ),
        (
            lambda batch: setattr(batch, "max_query_len", 1),
            "sequence mismatch",
        ),
        (
            lambda batch: batch.input_ids.fill_(9),
            "prompt token sequence changed",
        ),
    ],
)
def test_pinned_warmup_prefill_sample_is_exact(loaded, mutate, match):
    _, v2, _ = loaded
    req_states = _empty_req_states()
    helper = _helper(v2, req_states=req_states)
    with pytest.raises(RuntimeError, match=match):
        with helper.pinned_startup_warmup_scope(expected_prompt_len=2):
            _register_warmup_request(req_states)
            helper.add_request(0, 2, _warmup_params())
            batch = _warmup_input(0)
            mutate(batch)
            helper.capture_and_force(torch.ones((1, 154_880)), batch)


def test_warmup_name_cannot_bypass_real_guards_outside_scope(loaded):
    _, v2, _ = loaded
    req_states = _empty_req_states()
    _register_warmup_request(req_states)
    helper = _helper(v2, req_states=req_states)
    with pytest.raises(RuntimeError, match="one prompt token"):
        helper.add_request(0, 2, _warmup_params())
    with pytest.raises(RuntimeError, match="without a registered"):
        helper.capture_and_force(
            torch.ones((1, 154_880)), _warmup_input(0)
        )


def test_patch_is_three_call_seam_against_pinned_source():
    patch = (PATCH_ROOT / "p8_decode_capture/sampler-v2-hook.patch").read_text()
    assert patch.count("+from p8_decode_capture.v2_hook import") == 1
    assert patch.count("+        self.p8_decode_capture =") == 1
    assert patch.count("+        self.p8_decode_capture.add_request") == 1
    assert patch.count("+        logits = self.p8_decode_capture.capture_and_force") == 1
    assert "vllm/v1/worker/gpu/sample/sampler.py" in patch


def test_warmup_patch_is_a_lexical_scope_at_the_pinned_entrypoint():
    patch = (PATCH_ROOT / "p8_decode_capture/warmup-v2-hook.patch").read_text()
    assert "vllm/v1/worker/gpu/warmup.py" in patch
    assert patch.count("+def _warmup_kernels_impl(") == 1
    assert patch.count("+    with capture.pinned_startup_warmup_scope(") == 1
    assert "_warmup_0_" not in patch
    assert "os.environ" not in patch
