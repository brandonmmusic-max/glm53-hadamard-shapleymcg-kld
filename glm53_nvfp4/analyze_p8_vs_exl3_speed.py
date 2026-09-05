"""Analyze the preregistered five-cold-run P8 versus EXL3 speed gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PRIMARY_CONTEXT = 32768
SECONDARY_CONTEXT = 65536


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return out


def _extract(path: Path, expected_model: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    if metadata.get("model") != expected_model:
        raise ValueError(f"model mismatch in {path}")
    if metadata.get("engine") != "vllm":
        raise ValueError(f"engine mismatch in {path}")
    if metadata.get("decode_mode") != "duration":
        raise ValueError(f"decode mode mismatch in {path}")
    if metadata.get("standalone_prefill") is not True:
        raise ValueError(f"standalone prefill missing in {path}")
    if metadata.get("concurrency_levels") != [1]:
        raise ValueError(f"concurrency mismatch in {path}")
    if metadata.get("context_lengths") != [PRIMARY_CONTEXT, SECONDARY_CONTEXT]:
        raise ValueError(f"context mismatch in {path}")
    if float(metadata.get("duration_per_test", -1)) != 20.0:
        raise ValueError(f"duration mismatch in {path}")
    if float(metadata.get("decode_warmup_seconds", -1)) != 3.0:
        raise ValueError(f"warmup mismatch in {path}")
    if int(metadata.get("max_tokens", -1)) != 4096:
        raise ValueError(f"max-token mismatch in {path}")

    prefill: dict[str, Any] = {}
    for context in (PRIMARY_CONTEXT, SECONDARY_CONTEXT):
        row = payload.get("prefill", {}).get(str(context), {})
        validation = row.get("server_validation", {})
        if validation.get("invalid_reason"):
            raise ValueError(f"invalid prefill validation in {path}: {context}")
        if validation.get("method") != "prometheus":
            raise ValueError(f"missing Prometheus prefill validation in {path}: {context}")
        prefill[str(context)] = {
            "client_tps": _finite_positive(row.get("client_tok_per_sec", row.get("tok_per_sec")), "client prefill"),
            "server_tps": _finite_positive(validation.get("tok_per_sec"), "server prefill"),
            "prompt_tokens": int(row.get("prompt_tokens", 0)),
        }

    result_index = {
        (int(row.get("concurrency", -1)), int(row.get("context_tokens", -1))): row
        for row in payload.get("results", [])
    }
    decode: dict[str, Any] = {}
    for context in (PRIMARY_CONTEXT, SECONDARY_CONTEXT):
        row = result_index.get((1, context))
        if row is None:
            raise ValueError(f"missing decode cell in {path}: {context}")
        if row.get("aggregate_source") != "openai_continuous_usage":
            raise ValueError(f"decode source mismatch in {path}: {context}")
        if int(row.get("num_errors", 0)) != 0:
            raise ValueError(f"decode errors in {path}: {context}")
        if row.get("underfilled") is True or row.get("warmup_timed_out") is True:
            raise ValueError(f"invalid decode readiness in {path}: {context}")
        effective = row.get("effective_concurrency")
        if effective is not None and float(effective) != 1.0:
            raise ValueError(f"effective concurrency mismatch in {path}: {context}")
        decode[str(context)] = {
            "aggregate_tps": _finite_positive(row.get("aggregate_tps"), "decode throughput"),
            "itl_p50": _finite_positive(row.get("inter_token_latency_p50"), "decode ITL"),
            "server_tps": _finite_positive(row.get("server_gen_throughput"), "server decode throughput"),
        }

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "prefill": prefill,
        "decode": decode,
        "hardware": payload.get("hardware_run_summary", {}),
    }


def analyze(plan_path: Path, p8_paths: list[Path], exl3_paths: list[Path]) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    required = int(plan["benchmark"]["cold_process_runs_per_arm"])
    if len(p8_paths) != required or len(exl3_paths) != required:
        raise ValueError(f"exactly {required} runs per arm are required")
    p8 = [_extract(path, plan["candidate"]["model_name"]) for path in p8_paths]
    exl3 = [_extract(path, plan["baseline"]["model_name"]) for path in exl3_paths]

    def summarize(arm: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"runs": arm, "median": {"prefill": {}, "decode": {}}}
        for context in (PRIMARY_CONTEXT, SECONDARY_CONTEXT):
            key = str(context)
            out["median"]["prefill"][key] = {
                "server_tps": statistics.median(row["prefill"][key]["server_tps"] for row in arm),
                "client_tps": statistics.median(row["prefill"][key]["client_tps"] for row in arm),
            }
            out["median"]["decode"][key] = {
                "aggregate_tps": statistics.median(row["decode"][key]["aggregate_tps"] for row in arm),
                "itl_p50": statistics.median(row["decode"][key]["itl_p50"] for row in arm),
                "server_tps": statistics.median(row["decode"][key]["server_tps"] for row in arm),
            }
        return out

    p8_summary = summarize(p8)
    exl3_summary = summarize(exl3)
    p8_prefill = p8_summary["median"]["prefill"][str(PRIMARY_CONTEXT)]["server_tps"]
    exl3_prefill = exl3_summary["median"]["prefill"][str(PRIMARY_CONTEXT)]["server_tps"]
    p8_decode = p8_summary["median"]["decode"][str(PRIMARY_CONTEXT)]["aggregate_tps"]
    exl3_decode = exl3_summary["median"]["decode"][str(PRIMARY_CONTEXT)]["aggregate_tps"]
    prefill_gain = p8_prefill / exl3_prefill - 1.0
    decode_gain = p8_decode / exl3_decode - 1.0
    passed = prefill_gain > 0.0 and decode_gain > 0.0
    return {
        "schema": "glm53-p8-uniform-all42-vs-exl3-speed-analysis.v1",
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "primary_context_tokens": PRIMARY_CONTEXT,
        "p8": p8_summary,
        "exl3": exl3_summary,
        "primary_effect": {
            "prefill_relative_gain": prefill_gain,
            "prefill_percent_gain": 100.0 * prefill_gain,
            "decode_relative_gain": decode_gain,
            "decode_percent_gain": 100.0 * decode_gain,
        },
        "decision": {
            "passed": passed,
            "allocation_game": "advance-to-power-sized-design" if passed else "stop",
            "rule": plan["decision_before_result"]["pass"],
        },
        "claim_boundary": plan["decision_before_result"]["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--p8", type=Path, nargs="+", required=True)
    parser.add_argument("--exl3", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.plan, args.p8, args.exl3)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary_effect"], sort_keys=True))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
