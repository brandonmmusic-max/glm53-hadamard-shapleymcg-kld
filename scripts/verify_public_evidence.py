#!/usr/bin/env python3
"""Verify every file recorded in the public evidence checksum manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence/MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checked = 0
    for raw in MANIFEST.read_text().splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {relative}: {actual} != {expected}")
        checked += 1
    print(f"verified {checked} evidence files")


if __name__ == "__main__":
    main()
