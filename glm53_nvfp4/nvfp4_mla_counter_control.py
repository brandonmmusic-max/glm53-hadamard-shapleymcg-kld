"""Issue atomic request-boundary commands to the NVFP4 MLA counter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def write_command(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def receipt_path(output_dir: Path, sequence: int, rank: int) -> Path:
    return output_dir / (
        f"nvfp4-mla-scale-counter-seq-{sequence:06d}-rank-{rank:03d}.json"
    )


def wait_for_receipts(
    output_dir: Path, sequence: int, ranks: tuple[int, ...], timeout_seconds: float
) -> list[Path]:
    deadline = time.monotonic() + timeout_seconds
    paths = [receipt_path(output_dir, sequence, rank) for rank in ranks]
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("sequence") != sequence:
                    raise RuntimeError(f"wrong sequence in {path}")
            return paths
        time.sleep(0.1)
    missing = [str(path) for path in paths if not path.is_file()]
    raise TimeoutError(f"counter receipts did not arrive: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--control", type=Path, required=True)
    issue.add_argument("--sequence", type=int, required=True)
    issue.add_argument(
        "--action", choices=("reset", "dump_reset", "dump"), required=True
    )
    issue.add_argument("--request-id")
    issue.add_argument("--dump-request-id")

    wait = subparsers.add_parser("wait")
    wait.add_argument("--output-dir", type=Path, required=True)
    wait.add_argument("--sequence", type=int, required=True)
    wait.add_argument("--ranks", default="0,1,2,3")
    wait.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.subcommand == "issue":
        if args.sequence < 0:
            raise ValueError("sequence must be non-negative")
        payload: dict[str, object] = {
            "schema": "glm53.nvfp4-mla-scale-counter-control.v1",
            "sequence": args.sequence,
            "action": args.action,
        }
        if args.action in {"reset", "dump_reset"}:
            if not args.request_id:
                raise ValueError(f"{args.action} requires --request-id")
            payload["request_id"] = args.request_id
        if args.action in {"dump", "dump_reset"}:
            if not args.dump_request_id:
                raise ValueError(f"{args.action} requires --dump-request-id")
            payload["dump_request_id"] = args.dump_request_id
        write_command(args.control, payload)
        return 0
    ranks = tuple(int(value) for value in args.ranks.split(",") if value)
    for path in wait_for_receipts(
        args.output_dir, args.sequence, ranks, args.timeout_seconds
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

