import copy
import json

import numpy as np
import pytest

from glm53_nvfp4 import p8_decode_protocol as protocol


def role():
    windows = []
    for index in range(32):
        name = f"conditional-fit-{index:04d}"
        windows.append({"id": name, "domain": sorted(protocol.DOMAINS)[index // 8],
                        "prediction_positions": 2047, "teacher_source_role": "conditional-fit",
                        "teacher_path": f"logits/full-panel/conditional-fit/{name}.safetensors"})
    return {"schema": "glm53-codec.conditional-fit32-development-role.v1",
            "teacher_repo_id": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
            "teacher_revision": protocol.FULL_PANEL_SOURCE_REVISION,
            "roles": {"conditional-fit": windows, "fit": [], "selection": [],
                      "confirmation": [], "final": []}}


def test_full_role_not_a_tail_or_protected_substitute():
    payload = role()
    assert len(protocol.validate_role_payload(payload)) == 32
    for mutate in (
        lambda p: p["roles"]["confirmation"].append({"id": "protected"}),
        lambda p: p["roles"]["conditional-fit"].pop(),
        lambda p: p["roles"]["conditional-fit"][0].update(prediction_positions=32),
        lambda p: p["roles"]["conditional-fit"][0].update(teacher_source_role="confirmation"),
        lambda p: p["roles"]["conditional-fit"][0].update(teacher_path="../confirmation.safetensors"),
        lambda p: p["roles"]["conditional-fit"][0].update(domain="other"),
        lambda p: p["roles"]["conditional-fit"].__setitem__(1, p["roles"]["conditional-fit"][0]),
    ):
        changed = copy.deepcopy(payload)
        mutate(changed)
        with pytest.raises(ValueError):
            protocol.validate_role_payload(changed)


def test_every_teacher_row_has_its_exact_causal_history():
    tokens = np.arange(2048, dtype=np.int64)
    contract = protocol.full_window_contract(tokens)
    prompt = contract["prompt_token_ids"]
    forced = contract["forced_token_ids"]
    assert len(prompt) == 1 and len(forced) == 2047
    assert contract["one_token_prefill_rows"] == 1
    assert contract["true_decode_rows"] == 2046
    for row in range(2047):
        assert prompt + forced[:row] == tokens[:row + 1].tolist()
        assert forced[row] == tokens[row + 1]
    assert contract["teacher_row_start"] == 0
    assert contract["teacher_row_stop"] == 2047
    assert 32 * contract["prediction_rows"] == 65504


def test_token_value_hash_is_not_file_or_platform_dtype_hash():
    tokens = np.arange(2048, dtype=np.int32)
    assert protocol.tokens_sha(tokens) == protocol.tokens_sha(tokens.astype(">i8"))
    for bad in (tokens[:-1], tokens.astype(np.float32), -tokens,
                np.full(2048, protocol.VOCAB_LIMIT), tokens.reshape(2, 1024)):
        with pytest.raises(ValueError):
            protocol.full_window_contract(bad)


def test_exact_closure_includes_signed_zero_and_last_row(monkeypatch):
    monkeypatch.setattr(protocol, "VOCAB_LIMIT", 5)
    a = np.zeros((7, 5), dtype="<f4")
    b = a.copy()
    assert protocol.exact_logits(a, b, chunk_rows=2)["exact"]
    b[-1, -1] = -0.0
    result = protocol.exact_logits(a, b, chunk_rows=2)
    assert not result["exact"] and result["unequal_values"] == 1
    assert result["differing_rows"] == [6] and result["maximum_absolute_difference"] == 0
    b[0, 0] = 3
    assert protocol.exact_logits(a, b)["maximum_absolute_difference"] == 3
    b[2, 2] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        protocol.exact_logits(a, b)
    with pytest.raises(ValueError, match="geometry"):
        protocol.exact_logits(a.astype(np.float64), a.astype(np.float64))


def test_pinned_cpu_metric_has_no_alignment_shift(monkeypatch):
    monkeypatch.setattr(protocol, "VOCAB_LIMIT", 5)
    values = np.arange(15, dtype=np.float32).reshape(3, 5)
    tokens = np.array([0, 2, 3, 4], dtype=np.int64)
    score = protocol.score_aligned(values, values.copy(), tokens)
    assert np.array_equal(score.kld, np.zeros(3))
    assert np.array_equal(score.realized_token, tokens[1:])


def test_request_and_response_prove_actual_forced_ids_not_text():
    tokens = np.arange(2048, dtype=np.int64)
    request = protocol.completion_request(tokens, "conditional-fit-0056", "closure-test")
    assert request["prompt"] == [0] and request["max_tokens"] == 2047
    assert request["return_token_ids"] and request["ignore_eos"]
    assert request["vllm_xargs"]["p8_decode_capture_start_output_len"] == 0
    response = {"model": "closure-test", "choices": [{"token_ids": tokens[1:].tolist(),
                "finish_reason": "length", "text": "irrelevant"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2047}}
    assert protocol.verify_response(response, request)["prediction_positions"] == 2047
    for mutate in (lambda p: p["choices"][0]["token_ids"].__setitem__(900, 42),
                   lambda p: p["choices"][0].update(finish_reason="stop"),
                   lambda p: p["usage"].update(prompt_tokens=2),
                   lambda p: p.update(model="other")):
        changed = copy.deepcopy(response)
        mutate(changed)
        with pytest.raises(ValueError, match="exact forced sequence"):
            protocol.verify_response(changed, request)


def test_capture_receipt_checks_raw_bytes_and_causal_range(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROWS", 3)
    monkeypatch.setattr(protocol, "VOCAB_LIMIT", 5)
    tokens = np.arange(4, dtype=np.int64)
    window_id = "conditional-fit-0056"
    raw = tmp_path / f"{window_id}.logits.f32"
    values = np.arange(15, dtype="<f4").reshape(3, 5)
    raw.write_bytes(values.tobytes())
    xargs = protocol.completion_request(tokens, window_id, "test")["vllm_xargs"]
    metadata = {
        "schema": "glm53-p8.forced-decode-logits.v1", "status": "complete",
        "window_id": window_id, "dtype": "<f4", "shape": [3, 5], "real_vocab_size": 5,
        "original_logit_width": 8, "rows_completed": 3, "capture_start_output_len": 0,
        "captured_output_len_range": [0, 2], "causal_teacher_row_range": [0, 2],
        "one_token_prefill_row_range": [0, 0], "true_decode_row_range": [1, 2],
        "prompt_token_count": 1, "forced_token_count": 3,
        "prompt_token_ids_sha256": protocol.tokens_sha(tokens[:1]),
        "forced_token_ids_sha256": protocol.tokens_sha(tokens[1:]),
        "sequence_token_ids_sha256": protocol.tokens_sha(tokens),
        "raw_file": raw.name, "raw_bytes": 60, "raw_sha256": protocol.sha(raw),
        "pre_mask_raw_fp32": True, "finite_real_vocab": True, "tp_rank": 0,
        "request_extra_arg_keys": sorted(xargs), "started_unix_ns": 1, "completed_unix_ns": 2,
    }
    path = tmp_path / f"{window_id}.capture.json"
    path.write_text(json.dumps(metadata))
    array, _ = protocol.load_capture(tmp_path, window_id, tokens)
    assert np.array_equal(array, values)
    assert not array.flags.writeable
    for key, value in (("causal_teacher_row_range", [1, 3]), ("pre_mask_raw_fp32", False),
                       ("tp_rank", 1), ("raw_sha256", "0" * 64)):
        altered = {**metadata, key: value}
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            protocol.load_capture(tmp_path, window_id, tokens)
