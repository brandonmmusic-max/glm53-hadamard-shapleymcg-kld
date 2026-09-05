#!/usr/bin/env python3
"""Verify the immutable FP8-NoPE reference patch receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing reference_manifest.json and patch files",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "reference_manifest.json").read_text())
    failures: list[str] = []
    for item in manifest["references"]:
        path = root / item["file"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(
                f"hash mismatch: {path.name}: expected {item['sha256']}, got {actual}"
            )
            continue
        header = path.read_bytes()[:4096].decode("utf-8", errors="replace")
        if item["commit"] not in header:
            failures.append(f"commit header missing: {path.name}: {item['commit']}")
            continue
        print(f"PASS {path.name} {actual}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"PASS record_bytes={manifest['record_contract']['record_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
