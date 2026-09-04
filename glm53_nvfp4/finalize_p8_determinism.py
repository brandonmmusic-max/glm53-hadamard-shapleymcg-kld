"""Seal the five-cold-run P8 split determinism decision."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if len(args.runs) != plan["runs"]:
        raise ValueError("run count disagrees with frozen plan")
    rows = [json.loads(path.read_text()) for path in args.runs]
    keys = [(bits, tokens) for bits in (3, 4) for tokens in (33, 64)]
    hashes = {
        f"k{bits}-m{tokens}": [
            next(cell["output_sha256"] for cell in row["cells"] if (cell["bits"], cell["tokens"]) == (bits, tokens))
            for row in rows
        ]
        for bits, tokens in keys
    }
    passed = all(row["decision"] == "pass" for row in rows) and all(
        len(set(values)) == 1 for values in hashes.values()
    )
    payload = {
        "schema": "glm53-p8-mcg-split-determinism-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass-five-run-bitwise-determinism" if passed else "fail",
        "output_sha256_by_run": hashes,
        "inputs": {"plan": sha256_file(args.plan), "runs": [sha256_file(path) for path in args.runs]},
        "scope": "split P8 microbenchmark tensors only; not full-model determinism",
        "ldlq": False,
        "strict_kld_gate_relaxed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
