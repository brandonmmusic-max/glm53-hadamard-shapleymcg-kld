from __future__ import annotations

import json
from pathlib import Path

import pytest

from glm53_nvfp4.analyze_p8_vs_exl3_speed import analyze


def _run(path: Path, model: str, prefill: float, decode: float) -> Path:
    payload = {
        "metadata": {
            "model": model,
            "engine": "vllm",
            "decode_mode": "duration",
            "standalone_prefill": True,
            "concurrency_levels": [1],
            "context_lengths": [32768, 65536],
            "duration_per_test": 20.0,
            "decode_warmup_seconds": 3.0,
            "max_tokens": 4096,
        },
        "prefill": {},
        "results": [],
        "hardware_run_summary": {"gpu_count": 4},
    }
    for context in (32768, 65536):
        payload["prefill"][str(context)] = {
            "tok_per_sec": prefill,
            "client_tok_per_sec": prefill,
            "prompt_tokens": context - 100,
            "server_validation": {
                "method": "prometheus",
                "tok_per_sec": prefill,
                "invalid_reason": "",
            },
        }
        payload["results"].append({
            "concurrency": 1,
            "context_tokens": context,
            "aggregate_source": "openai_continuous_usage",
            "aggregate_tps": decode,
            "inter_token_latency_p50": 1.0 / decode,
            "server_gen_throughput": decode,
            "num_errors": 0,
            "underfilled": False,
            "warmup_timed_out": False,
            "effective_concurrency": 1,
        })
    path.write_text(json.dumps(payload))
    return path


def _plan(path: Path) -> Path:
    path.write_text(json.dumps({
        "benchmark": {"cold_process_runs_per_arm": 5},
        "candidate": {"model_name": "p8"},
        "baseline": {"model_name": "exl3"},
        "decision_before_result": {"pass": "both greater", "claim_boundary": "product"},
    }))
    return path


def test_passes_only_when_both_primary_metrics_win(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "plan.json")
    p8 = [_run(tmp_path / f"p8-{i}.json", "p8", 110.0, 105.0) for i in range(5)]
    exl3 = [_run(tmp_path / f"exl3-{i}.json", "exl3", 100.0, 100.0) for i in range(5)]
    result = analyze(plan, p8, exl3)
    assert result["decision"]["passed"] is True
    assert result["primary_effect"]["prefill_percent_gain"] == pytest.approx(10.0)
    assert result["primary_effect"]["decode_percent_gain"] == pytest.approx(5.0)


def test_decode_regression_stops(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "plan.json")
    p8 = [_run(tmp_path / f"p8-{i}.json", "p8", 110.0, 99.0) for i in range(5)]
    exl3 = [_run(tmp_path / f"exl3-{i}.json", "exl3", 100.0, 100.0) for i in range(5)]
    assert analyze(plan, p8, exl3)["decision"] == {
        "passed": False,
        "allocation_game": "stop",
        "rule": "both greater",
    }


def test_rejects_missing_run_or_unmatched_method(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "plan.json")
    p8 = [_run(tmp_path / f"p8-{i}.json", "p8", 110.0, 105.0) for i in range(5)]
    exl3 = [_run(tmp_path / f"exl3-{i}.json", "exl3", 100.0, 100.0) for i in range(5)]
    with pytest.raises(ValueError, match="exactly 5"):
        analyze(plan, p8[:4], exl3)
    broken = json.loads(p8[0].read_text())
    broken["metadata"]["duration_per_test"] = 10.0
    p8[0].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="duration mismatch"):
        analyze(plan, p8, exl3)
