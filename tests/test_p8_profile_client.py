import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from glm53_nvfp4.p8_profile_client import (
    COMPLETION_TOKENS, PROMPT_TOKENS, Config, Response, run,
)


def response(data, status=200):
    return Response(status, json.dumps(data).encode())


class MockServer:
    def __init__(self):
        self.calls = []
        self.fail_stage = None
        self.model = "p8-exact"
        self.tokens = {"tokens": [10, 11, 12], "count": 3, "max_model_len": 131072}

    def __call__(self, method, endpoint, payload):
        self.calls.append((method, endpoint, payload))
        stage = len(self.calls)
        if stage == self.fail_stage:
            return Response(500, b"preserved failure body")
        if endpoint == "/v1/models":
            return response({"data": [{"id": self.model}]})
        if endpoint == "/tokenize":
            return response(self.tokens)
        if endpoint == "/v1/completions":
            return response({"model": self.model,
                             "usage": {"prompt_tokens": PROMPT_TOKENS,
                                       "completion_tokens": COMPLETION_TOKENS},
                             "choices": [{"finish_reason": "length", "text": "synthetic"}]})
        return Response(200, b"")


@pytest.fixture
def config(tmp_path):
    return Config("http://localhost:8000", "p8-exact", tmp_path / "artifacts", 1, 18)


def receipt(config):
    return json.loads((config.output_dir / "receipt.json").read_text())


def test_success_exact_synthetic_prompt_cleanup_and_hashes(config):
    server = MockServer()
    result = run(config, transport=server)
    assert result["status"] == "client-completed-trace-unverified"
    assert result["stop_succeeded"] and not result["trace_verified"]
    assert [call[1] for call in server.calls] == ["/v1/models", "/tokenize", "/v1/completions",
                                                "/v1/models", "/start_profile", "/v1/completions",
                                                "/stop_profile"]
    warm, profiled = server.calls[2][2], server.calls[5][2]
    assert len(profiled["prompt"]) == PROMPT_TOKENS
    assert len(server.calls[1][2]["prompt"]) < 256
    assert profiled["prompt"] == ([10, 11, 12] * 10923)[:PROMPT_TOKENS]
    construction = json.loads((config.output_dir / "prompt-construction.json").read_text())
    assert construction["seed_token_count"] == 3
    assert construction["repetitions"] == 10923
    assert warm["prompt"][0] != profiled["prompt"][0]
    assert warm["prompt"][1:] == profiled["prompt"][1:]
    assert profiled["ignore_eos"] and profiled["max_tokens"] == 34
    assert profiled["temperature"] == 0 and profiled["seed"] == 0
    for name, digest in result["artifact_sha256"].items():
        assert hashlib.sha256((config.output_dir / name).read_bytes()).hexdigest() == digest
    assert (config.output_dir / "receipt.sha256").read_text().strip() == hashlib.sha256(
        (config.output_dir / "receipt.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        run(config, transport=server)
    assert len(server.calls) == 7


def test_model_mismatch_stops_before_tokenize(config):
    server = MockServer()
    server.model = "exl3"
    with pytest.raises(ValueError, match="expected model"):
        run(config, transport=server)
    assert len(server.calls) == 1
    assert receipt(config)["status"] == "failed"
    assert not receipt(config)["stop_attempted"]


@pytest.mark.parametrize("bad", [-1, True, 1.5, "10", 2**63])
def test_invalid_token_id(config, bad):
    server = MockServer()
    server.tokens["tokens"][0] = bad
    with pytest.raises(ValueError, match="token IDs"):
        run(config, transport=server)
    assert len(server.calls) == 2
    assert not receipt(config)["stop_attempted"]


@pytest.mark.parametrize("change", ["count", "short", "capacity"])
def test_tokenizer_counts_and_capacity(config, change):
    server = MockServer()
    if change == "count":
        server.tokens["count"] += 1
    elif change == "short":
        server.tokens["tokens"] = [10]
        server.tokens["count"] = 1
    else:
        server.tokens["max_model_len"] = PROMPT_TOKENS
    with pytest.raises(ValueError):
        run(config, transport=server)
    assert len(server.calls) == 2
    assert receipt(config)["status"] == "failed"


@pytest.mark.parametrize("stage", [3, 5, 6, 7])
def test_http_failure_and_stop_cleanup(config, stage):
    server = MockServer()
    server.fail_stage = stage
    with pytest.raises(RuntimeError, match="HTTP 500"):
        run(config, transport=server)
    saved = receipt(config)
    assert saved["status"] == "failed"
    assert saved["stop_attempted"] == (stage >= 5)
    if stage >= 5:
        assert server.calls[-1][1] == "/stop_profile"
    assert any(path.read_bytes() == b"preserved failure body"
               for path in config.output_dir.glob("*.response.bin"))


def test_network_failure_after_start_still_stops(config):
    server = MockServer()

    def send(method, endpoint, payload):
        if len(server.calls) == 5:
            server.calls.append((method, endpoint, payload))
            raise TimeoutError("mock timeout")
        return server(method, endpoint, payload)

    with pytest.raises(TimeoutError):
        run(config, transport=send)
    assert server.calls[-1][1] == "/stop_profile"
    assert receipt(config)["stop_succeeded"]


def test_primary_error_retained_when_stop_also_fails(config):
    server = MockServer()

    def send(method, endpoint, payload):
        if endpoint == "/stop_profile":
            raise TimeoutError("cleanup failure")
        if len(server.calls) == 5:
            raise RuntimeError("primary failure")
        return server(method, endpoint, payload)

    with pytest.raises(RuntimeError, match="primary failure"):
        run(config, transport=send)
    assert receipt(config)["stop_error"]["message"] == "cleanup failure"


@pytest.mark.parametrize("changes", [{"prefill_iterations": 0}, {"profiler_delay_iterations": 2},
                                     {"profiler_max_iterations": 15}])
def test_invalid_profiler_mapping_before_network(config, changes):
    server = MockServer()
    with pytest.raises(ValueError):
        run(replace(config, **changes), transport=server)
    assert not server.calls


def test_chunked_prefill_config(config):
    server = MockServer()
    run(replace(config, prefill_iterations=16, profiler_delay_iterations=33), transport=server)
    saved = receipt(config)
    assert saved["expected_profiled_decode_iterations"] == 16
    assert saved["expected_recorded_worker_iteration_first"] == 33
    assert saved["expected_recorded_worker_iteration_last"] == 48
    assert saved["expected_automatic_stop_before_worker_iteration"] == 49
    assert saved["required_server_profiler_settings"]["profiler"] == "cuda"


def test_model_changed_after_warm_request_never_starts_profiler(config):
    server = MockServer()

    def send(method, endpoint, payload):
        if len(server.calls) == 3:
            server.model = "replacement-model"
        return server(method, endpoint, payload)

    with pytest.raises(ValueError, match="expected model"):
        run(config, transport=send)
    assert len(server.calls) == 4
    assert not receipt(config)["stop_attempted"]


def test_bad_profile_usage_still_stops(config):
    server = MockServer()

    def send(method, endpoint, payload):
        reply = server(method, endpoint, payload)
        if len(server.calls) == 6:
            data = json.loads(reply.body)
            data["usage"]["completion_tokens"] = 3
            return response(data)
        return reply

    with pytest.raises(ValueError, match="token counts"):
        run(config, transport=send)
    assert receipt(config)["stop_succeeded"]


def test_keyboard_interrupt_still_stops_and_records_failure(config):
    server = MockServer()

    def send(method, endpoint, payload):
        if len(server.calls) == 5:
            server.calls.append((method, endpoint, payload))
            raise KeyboardInterrupt()
        return server(method, endpoint, payload)

    with pytest.raises(KeyboardInterrupt):
        run(config, transport=send)
    assert receipt(config)["stop_succeeded"]
    assert receipt(config)["error"]["type"] == "KeyboardInterrupt"


def test_malformed_profile_json_preserved_and_cleanup_attempted(config):
    server = MockServer()

    def send(method, endpoint, payload):
        reply = server(method, endpoint, payload)
        return Response(200, b"not json") if len(server.calls) == 6 else reply

    with pytest.raises(ValueError):
        run(config, transport=send)
    assert (config.output_dir / "06-profile.response.bin").read_bytes() == b"not json"
    assert receipt(config)["stop_succeeded"]


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_nonfinite_or_nonpositive_timeout_rejected(config, timeout):
    server = MockServer()
    with pytest.raises(ValueError, match="finite timeout"):
        run(replace(config, timeout_seconds=timeout), transport=server)
    assert not server.calls


def test_cli_help_runs_without_endpoint_calls():
    result = subprocess.run(
        [sys.executable, "-m", "glm53_nvfp4.p8_profile_client", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--prefill-iterations" in result.stdout
    assert "profiler='cuda'" in result.stdout
