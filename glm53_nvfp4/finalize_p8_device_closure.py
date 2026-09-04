"""Validate and seal the frozen P8 MCG device-closure repeat."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    plan = _read(args.plan)
    decode = _read(args.decode)
    moe = _read(args.moe)
    thresholds = plan["thresholds"]
    decode_pass = (
        decode.get("decision") == "pass"
        and {row["bits"] for row in decode["results"]} == {3, 4}
        and all(
            row["bit_exact"]
            and row["mismatched_words"]
            == thresholds["decode_mismatched_words"]
            for row in decode["results"]
        )
    )
    moe_pass = (
        moe.get("decision") == "pass"
        and {row["bits"] for row in moe["cells"]} == {3, 4}
        and all(
            row["pass"]
            and row["finite"] == thresholds["finite"]
            and row["cosine"] > thresholds["moe_cosine_min_exclusive"]
            and row["relative_l2"] < thresholds["moe_relative_l2_max_exclusive"]
            for row in moe["cells"]
        )
    )
    source_match = (
        decode["runtime_source"]["sha256"]
        == moe["runtime_dynamic"]["sha256"]
        == plan["sources"]["runtime_dynamic"]
    )
    passed = decode_pass and moe_pass and source_match
    result = {
        "schema": "glm53-p8-mcg-device-closure-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass-device-arithmetic-closure" if passed else "fail",
        "scope": (
            "developmental decoder and small-M native arithmetic only; "
            "not end-to-end KLD, split-prefill, speed, or product qualification"
        ),
        "checks": {
            "decode": decode_pass,
            "moe": moe_pass,
            "runtime_source_identity": source_match,
        },
        "inputs": {
            "plan": sha256_file(args.plan),
            "decode_repeat": sha256_file(args.decode),
            "moe_repeat": sha256_file(args.moe),
        },
        "products": ["P8-K3", "P8-K4"],
        "compute": "mxf8f6f4 m16n8k32 with E4M3 activations and weights",
        "table_bytes": 0,
        "ldlq": False,
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
        "strict_kld_gate_relaxed": False,
        "engineering_blocker_relaxed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
