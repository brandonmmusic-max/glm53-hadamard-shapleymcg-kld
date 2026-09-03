"""Append an integrity-chained stage receipt without changing the sealed plan projection."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    receipts = record["stage_receipts"]
    prior = receipts[-1]["receipt_sha256"] if receipts else "Not tested"
    evidence = []
    for path in args.evidence:
        evidence.append({"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = {
        "stage": args.stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": record["seal"]["plan_sha256"],
        "prior_receipt_sha256": prior,
        "evidence": evidence,
        "notes": args.note,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    receipts.append(payload)
    if record["status"] == "sealed":
        record["status"] = "executing"
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(payload["receipt_sha256"])


if __name__ == "__main__":
    main()
