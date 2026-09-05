#!/usr/bin/env python3
"""Explicit scoring/retirement commands for a sealed Tail-V2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_nvfp4 import tail_v2_product_validation as validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    subs = parser.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit-log")
    audit.add_argument("--arm", choices=validation.ARMS, required=True)
    audit.add_argument("--log", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    score = subs.add_parser("score-retire")
    score.add_argument("--arm", choices=validation.ARMS, required=True)
    score.add_argument("--repeat", type=int, choices=range(1, 6), required=True)
    score.add_argument("--window-id", required=True)
    score.add_argument("--capture-root", type=Path, required=True)
    score.add_argument("--receipt-root", type=Path, required=True)
    score.add_argument("--runtime-audit", type=Path, required=True)
    verify = subs.add_parser("verify-determinism")
    verify.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()
    plan = validation.authenticate_plan(args.plan)
    if args.command == "audit-log":
        result = validation.audit_runtime_log(args.log.read_text(errors="replace"), args.arm)
        result = {**result, "source_log": str(args.log.resolve()),
                  "source_log_sha256": validation.sha(args.log)}
        validation._durable_json(args.output, result)
    elif args.command == "score-retire":
        runtime = json.loads(args.runtime_audit.read_text())
        source_log = Path(runtime["source_log"])
        replay = validation.audit_runtime_log(source_log.read_text(errors="replace"), args.arm)
        recorded = {key: value for key, value in runtime.items()
                    if key not in ("source_log", "source_log_sha256")}
        if (recorded != replay or validation.sha(source_log) != runtime["source_log_sha256"]):
            raise ValueError("runtime audit does not replay from its source log")
        by_id = {window["id"]: window for window in plan["windows"]}
        result = validation.score_retire_window(
            arm=args.arm, repeat=args.repeat, window=by_id[args.window_id],
            teacher_root=Path(plan["teacher_root"]), capture_root=args.capture_root,
            receipt_root=args.receipt_root, runtime_audit_sha256=validation.sha(args.runtime_audit),
        )
    else:
        result = validation.verify_five_run_determinism(args.receipt_root, plan["windows"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
