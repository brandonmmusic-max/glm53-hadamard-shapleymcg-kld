"""Synthetic 32K HTTP profiling client; never configures or launches a server.

Run only on a separately authorized, otherwise idle profiling server. The pinned
worker calls profiler.step() BEFORE model work. With P prefill iterations, use
delay_iterations=P+17, max_iterations=16, warmup_iterations=0. A 34-token
completion supplies 16 warm decode steps, 16 recorded decode steps, and a step
that triggers automatic stopping. Chunking, prefix caching, speculative decode,
or other requests can invalidate this mapping: inspect the worker trace.

For 2048-token chunked prefill, prefix caching disabled, and an otherwise idle
server, expected P=16: configure delay_iterations=33 and max_iterations=16.
The expected recorded worker iterations are 33..48, with stop before 49.
Use profiler='cuda' for the Nsight capture; this client does not configure it.

Example (server configuration is supplied separately):
python3 -m glm53_nvfp4.p8_profile_client --base-url http://127.0.0.1:8000
  --expected-model MODEL --output-dir NEW_DIRECTORY --prefill-iterations 16
  --profiler-delay-iterations 33 --profiler-max-iterations 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROMPT_TOKENS = 32768
WARM_DECODE_ITERATIONS = 16
PROFILE_ITERATIONS = 16
COMPLETION_TOKENS = WARM_DECODE_ITERATIONS + PROFILE_ITERATIONS + 2
PROFILER_SOURCE = {
    "image": "sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8",
    "wrapper_path": "/opt/venv/lib/python3.12/site-packages/vllm/profiler/wrapper.py",
    "wrapper_sha256": "96bf700a1507bf05796f310e64f569c9fd645d0089ca44b76bdb9e12ceddd998",
    "wrapper_lines": "94-124",
    "config_sha256": "9757f4b9680038b804fb5a3564a9a45a8e47a6b8492a7bb5376b5530dcc0a70d",
    "worker_step_location": "vllm/v1/worker/gpu_worker.py:995-1002,1174",
}


@dataclass(frozen=True)
class Config:
    base_url: str
    expected_model: str
    output_dir: Path
    prefill_iterations: int
    profiler_delay_iterations: int
    profiler_max_iterations: int = PROFILE_ITERATIONS
    timeout_seconds: float = 600.0

    def validate(self) -> None:
        url = urlsplit(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.netloc or url.username
                or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("base URL must be an HTTP server origin without credentials")
        if not self.expected_model or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("expected model and positive finite timeout are required")
        if self.prefill_iterations < 1:
            raise ValueError("prefill_iterations must be positive")
        if self.profiler_delay_iterations != self.prefill_iterations + WARM_DECODE_ITERATIONS + 1:
            raise ValueError("profiler delay must equal prefill_iterations + 17")
        if self.profiler_max_iterations != PROFILE_ITERATIONS:
            raise ValueError("profiler max_iterations must be exactly 16")


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes


Transport = Callable[[str, str, dict | None], Response]


def http_transport(config: Config, api_key: str | None = None) -> Transport:
    def send(method: str, endpoint: str, payload: dict | None) -> Response:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(config.base_url.rstrip("/") + endpoint,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers=headers, method=method)
        try:
            with urlopen(request, timeout=config.timeout_seconds) as response:
                return Response(response.status, response.read())
        except HTTPError as error:
            return Response(error.code, error.read())
    return send


class Artifacts:
    """Exclusive writes plus hashes; protects against accidental overwrite, not tampering."""

    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.hashes: dict[str, str] = {}

    def write(self, name: str, value: object, *, raw: bool = False) -> None:
        body = value if raw else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with (self.path / name).open("xb") as file:
            file.write(body)
        (self.path / name).chmod(0o444)
        self.hashes[name] = hashlib.sha256(body).hexdigest()

    def exchange(self, send: Transport, stage: str, method: str,
                 endpoint: str, payload: dict | None = None, *, empty_ok: bool = False):
        self.write(f"{stage}.request.json", {"method": method, "endpoint": endpoint,
                                           "payload": payload})
        started = time.monotonic()
        try:
            response = send(method, endpoint, payload)
            self.write(f"{stage}.response.bin", response.body, raw=True)
            self.write(f"{stage}.http.json", {"status": response.status,
                                             "elapsed_seconds": time.monotonic() - started})
            if response.status != 200:
                raise RuntimeError(f"{stage}: HTTP {response.status}")
            if empty_ok and not response.body.strip():
                return None
            data = json.loads(response.body)
            if not isinstance(data, dict):
                raise ValueError(f"{stage}: expected JSON object")
            return data
        except BaseException as error:
            self.write(f"{stage}.error.json", {"type": type(error).__name__, "message": str(error)})
            raise


def validated_tokens(data: dict, *, minimum: int) -> list[int]:
    tokens = data.get("tokens")
    if (not isinstance(tokens, list) or len(tokens) < minimum
            or any(type(token) is not int or not 0 <= token < 2**63 for token in tokens)):
        raise ValueError("tokenizer returned invalid or insufficient token IDs")
    if type(data.get("count")) is not int or data["count"] != len(tokens):
        raise ValueError("tokenizer count does not match token IDs")
    if (type(data.get("max_model_len")) is not int
            or data["max_model_len"] < PROMPT_TOKENS + COMPLETION_TOKENS):
        raise ValueError("tokenizer reports insufficient model context capacity")
    return tokens


def verify_model(data: dict, expected: str) -> None:
    entries = data.get("data")
    if (not isinstance(entries, list) or not entries
            or any(not isinstance(entry, dict) for entry in entries)
            or [entry.get("id") for entry in entries] != [expected]):
        raise ValueError("server model list does not exactly match expected model")


def verify_completion(data: dict, expected: str) -> None:
    if data.get("model") != expected:
        raise ValueError("completion model differs from expected model")
    usage = data.get("usage", {})
    if (usage.get("prompt_tokens") != PROMPT_TOKENS
            or usage.get("completion_tokens") != COMPLETION_TOKENS):
        raise ValueError("completion usage differs from exact requested token counts")
    choices = data.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1
            or choices[0].get("finish_reason") != "length"):
        raise ValueError("completion did not end at requested token limit")


def run(config: Config, *, transport: Transport | None = None,
        api_key: str | None = None) -> dict:
    config.validate()
    artifacts = Artifacts(config.output_dir)
    send = transport if transport is not None else http_transport(config, api_key)
    config_data = asdict(config)
    config_data["output_dir"] = str(config.output_dir.resolve())
    artifacts.write("config.json", config_data)
    run_id = uuid.uuid4().hex
    receipt = {"schema": "p8-synthetic-profile-client.v1", "run_id": run_id,
               "status": "failed", "role": "synthetic-performance-only",
               "profiler_source": PROFILER_SOURCE,
               "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "server_config_verified": False, "trace_verified": False,
               "assumptions": ["idle dedicated server; no concurrent requests",
                               "no speculative decoding; one output token per decode step",
                               "declared prefill iteration count matches scheduler and cache behavior",
                               "server uses profiler=cuda, declared delay/max, and warmup_iterations=0",
                               "Nsight trace capture is configured separately and trace files verified separately"],
               "expected_profiled_decode_iterations": PROFILE_ITERATIONS,
               "expected_recorded_worker_iteration_first": config.profiler_delay_iterations,
               "expected_recorded_worker_iteration_last": config.profiler_delay_iterations + PROFILE_ITERATIONS - 1,
               "expected_automatic_stop_before_worker_iteration": config.profiler_delay_iterations + PROFILE_ITERATIONS,
               "required_server_profiler_settings": {
                   "profiler": "cuda",
                   "delay_iterations": config.profiler_delay_iterations,
                   "max_iterations": PROFILE_ITERATIONS, "warmup_iterations": 0,
                   "ignore_frontend": True},
               "stop_attempted": False, "stop_succeeded": False}
    primary_error: BaseException | None = None
    start_attempted = False
    try:
        verify_model(artifacts.exchange(send, "01-models", "GET", "/v1/models"), config.expected_model)
        # Tokenize a short seed to stay comfortably below server context limits.
        # Repeating returned valid IDs avoids guessing how text tokenizes at seams.
        synthetic = f"Synthetic profile {run_id}. Amber river stone cloud."
        tokenized = artifacts.exchange(send, "02-tokenize", "POST", "/tokenize",
                                       {"model": config.expected_model, "prompt": synthetic,
                                        "add_special_tokens": False})
        seed_tokens = validated_tokens(tokenized, minimum=2)
        repetitions = (PROMPT_TOKENS + len(seed_tokens) - 1) // len(seed_tokens)
        tokens = (seed_tokens * repetitions)[:PROMPT_TOKENS]
        artifacts.write("prompt-construction.json", {
            "rule": "repeat the complete tokenizer seed ID sequence, then truncate to exactly 32768 IDs",
            "seed_token_count": len(seed_tokens), "repetitions": repetitions,
            "prompt_token_count": len(tokens),
            "warm_rule": "replace only the first profile token with the first distinct token in the sequence"})
        # A different first token prevents this client's warm request from priming
        # the profile prompt's prefix cache. Existing server cache remains an assumption.
        alternative = next((token for token in tokens if token != tokens[0]), None)
        if alternative is None:
            raise ValueError("synthetic prompt has no distinct warmup first token")
        warm_tokens = [alternative, *tokens[1:]]
        artifacts.write("prompt-tokens.json", {"profile": tokens, "warm": warm_tokens})
        common = {"model": config.expected_model, "max_tokens": COMPLETION_TOKENS,
                  "temperature": 0.0, "top_p": 1.0, "seed": 0, "n": 1,
                  "ignore_eos": True, "stream": False, "add_special_tokens": False}
        warm = artifacts.exchange(send, "03-warm", "POST", "/v1/completions",
                                  {**common, "prompt": warm_tokens})
        verify_completion(warm, config.expected_model)
        verify_model(artifacts.exchange(send, "04-models", "GET", "/v1/models"), config.expected_model)
        # Even a failed/ambiguous start response may have activated remote workers.
        start_attempted = True
        artifacts.exchange(send, "05-start", "POST", "/start_profile", empty_ok=True)
        profiled = artifacts.exchange(send, "06-profile", "POST", "/v1/completions",
                                      {**common, "prompt": tokens})
        verify_completion(profiled, config.expected_model)
        receipt["status"] = "client-completed-trace-unverified"
    except BaseException as error:
        primary_error = error
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if start_attempted:
            receipt["stop_attempted"] = True
            try:
                artifacts.exchange(send, "07-stop", "POST", "/stop_profile", empty_ok=True)
                receipt["stop_succeeded"] = True
            except BaseException as error:
                receipt["stop_error"] = {"type": type(error).__name__, "message": str(error)}
                if primary_error is None:
                    primary_error = error
        if primary_error is not None:
            receipt["status"] = "failed"
        receipt["artifact_sha256"] = dict(artifacts.hashes)
        artifacts.write("receipt.json", receipt)
        artifacts.write("receipt.sha256", artifacts.hashes["receipt.json"].encode() + b"\n", raw=True)
    if primary_error is not None:
        raise primary_error
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-iterations", type=int, required=True)
    parser.add_argument("--profiler-delay-iterations", type=int, required=True)
    parser.add_argument("--profiler-max-iterations", type=int, default=PROFILE_ITERATIONS)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key-env", help="optional environment variable name; credentials are never recorded")
    args = vars(parser.parse_args())
    key_env = args.pop("api_key_env")
    api_key = os.environ[key_env] if key_env else None
    result = run(Config(**args), api_key=api_key)
    print(json.dumps({key: result[key] for key in ("status", "run_id", "trace_verified")}))


if __name__ == "__main__":
    main()
