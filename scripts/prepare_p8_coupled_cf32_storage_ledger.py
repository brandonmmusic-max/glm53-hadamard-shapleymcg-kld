#!/usr/bin/env python3
"""Create the fixed-scope external storage ledger; CPU/read-only inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from glm53_nvfp4 import p8_coupled_cf32_executor as executor


FUTURE_JIT_LOG_ALLOWANCE = 67_108_864
UNALLOCATED_RESERVE = 963_556_368


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--valid-seconds", type=int, default=3600)
    args = parser.parse_args()
    if (args.output != args.output.resolve() or args.capture_output != args.capture_output.resolve()
            or args.output.exists() or args.output.with_suffix(".sha256").exists()
            or args.valid_seconds < 300 or args.capture_output.exists()
            or not args.capture_output.is_relative_to(executor.CAMPAIGN_ROOT)):
        raise ValueError("fresh canonical ledger/capture paths and >=300-second validity required")
    rows = []
    for name, mode in executor.LEDGER_COMPONENTS.items():
        paths = ((args.capture_output,) if name == "capture_output"
                 else executor.FIXED_LEDGER_PATHS[name])
        baseline = executor._measure_component(name, mode, list(paths), args.capture_output)
        projected = (executor.QUALITY_PEAK if name == "quality_root" else
                     FUTURE_JIT_LOG_ALLOWANCE if name == "capture_output" else baseline)
        if projected < baseline:
            raise ValueError(f"fixed projection already exceeded by {name}")
        rows.append({"name": name, "mode": mode, "paths": [str(path) for path in paths],
                     "baseline_bytes": baseline, "projected_bytes": projected})
    projected = sum(row["projected_bytes"] for row in rows) + UNALLOCATED_RESERVE
    if projected > executor.GLOBAL_BUDGET:
        raise ValueError("fixed campaign projection exceeds the user ceiling")
    now = int(time.time())
    value = {"schema": executor.LEDGER_SCHEMA, "status": "pass",
        "measured_unix": now, "measured_utc": datetime.now(timezone.utc).isoformat(),
        "valid_until_unix": now + args.valid_seconds, "cutoff_unix": executor.LEDGER_CUTOFF,
        "ceiling_bytes": executor.GLOBAL_BUDGET,
        "retained_prior_quality_peak_charge_bytes": executor.QUALITY_PEAK,
        "capture_peak_already_charged": True, "pinned_raw_peak_bytes": executor.RAW_BYTES,
        "future_jit_log_allowance_in_baseline_projection": True,
        "future_jit_log_allowance_bytes": FUTURE_JIT_LOG_ALLOWANCE,
        "unallocated_reserve_bytes": UNALLOCATED_RESERVE,
        "baseline_projected_aggregate_bytes": projected,
        "remaining_below_ceiling_bytes": executor.GLOBAL_BUDGET - projected,
        "components": rows,
        "measurement_method": "fixed source-owned paths; apparent lstat trees; original-cutoff file sums; privileged read-only Docker find",
        "authorization": "storage only; execution still requires separate runtime manifest and execution seal"}
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".sha256").write_text(
        executor.runtime.sha(args.output) + "  " + args.output.name + "\n")


if __name__ == "__main__":
    main()
