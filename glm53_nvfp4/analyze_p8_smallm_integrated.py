"""Audit the frozen one-run integrated P8 small-M diagnostic."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OLD_P8_C1 = 20.6937
EXL3_C1 = 91.2645


def analyze(result: dict, graph: dict, execution: dict, server_log: str) -> dict:
    if execution.get("exit_code") != 0:
        raise ValueError("execution did not complete")
    if execution.get("protected_roles_opened") or execution.get("ldlq"):
        raise ValueError("experiment boundary differs")
    if graph != {"full_capture_complete": True, "ranks": [0, 1, 2, 3], "status": "pass"}:
        raise ValueError("FULL graph audit failed")
    pairs = {(int(layer), int(rank)) for layer, rank in re.findall(
        r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+).*small_m_scheduler=true",
        server_log,
    )}
    expected = {(layer, rank) for layer in range(3, 45) for rank in range(4)}
    if pairs != expected:
        raise ValueError("small-M layer/rank inventory differs")
    rows = result.get("results", [])
    if len(rows) != 1:
        raise ValueError("decode inventory differs")
    row = rows[0]
    if row.get("context_tokens") != 32768 or row.get("concurrency") != 1:
        raise ValueError("decode regime differs")
    decode = float(row["aggregate_tps"])
    prefill = result.get("prefill", {}).get("32768")
    if not prefill:
        raise ValueError("32K prefill result missing")
    prefill_tps = float(prefill["client_tok_per_sec"])
    max_temp = max(float(row["hardware_summary"]["temp_max_c"]),
                   float(prefill["hardware_summary"]["temp_max_c"]))
    integration_closed = max_temp < 90.0
    product_won = decode > EXL3_C1
    return {
        "schema": "glm53-p8-smallm-integrated-analysis.v1",
        "integration_status": "pass" if integration_closed else "fail",
        "product_gate": "pass" if product_won else "fail",
        "allocation_game": "eligible_to_restart" if product_won else "stopped",
        "decode_c1_tokens_per_second": decode,
        "prefill_32k_client_tokens_per_second": prefill_tps,
        "old_p8_decode_c1_tokens_per_second": OLD_P8_C1,
        "exl3_decode_c1_tokens_per_second": EXL3_C1,
        "decode_improvement_vs_old_p8_percent": 100.0 * (decode / OLD_P8_C1 - 1.0),
        "decode_delta_vs_exl3_percent": 100.0 * (decode / EXL3_C1 - 1.0),
        "old_p8_to_exl3_deficit_closed_percent": 100.0 * (decode - OLD_P8_C1) / (EXL3_C1 - OLD_P8_C1),
        "full_cuda_graph": True,
        "small_m_forward_pairs": len(pairs),
        "max_measured_gpu_temp_c": max_temp,
        "single_run_diagnostic": True,
        "five_cold_run_product_comparison": False,
        "protected_roles_opened": [],
        "ldlq": False,
        "interpretation": (
            "The run proves integrated recovery but cannot replace the frozen "
            "five-cold-run product comparison. The product gate remains failed "
            "unless P8 exceeds the fixed EXL3 C1 reference."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(
        json.loads((args.input_dir / "result.json").read_text()),
        json.loads((args.input_dir / "graph-audit.json").read_text()),
        json.loads((args.input_dir / "execution.json").read_text()),
        (args.input_dir / "server-ready.log").read_text(errors="replace"),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["integration_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
