#!/usr/bin/env python3
"""Build the checksum manifest for the publishable evidence snapshot."""

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
    paths = sorted(
        path
        for path in (ROOT / "evidence").rglob("*")
        if path.is_file() and path != MANIFEST
    )
    paths.extend(
        [
            ROOT / "results/current-kld-summary.json",
            ROOT / "results/selective-h16-summary.json",
            ROOT / "results/GLM_DECISIONS.md",
            ROOT / "figures/current-layer3-kld.png",
            ROOT / "figures/selective-h16-generalization.png",
        ]
    )
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in paths]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"recorded {len(paths)} evidence files")


if __name__ == "__main__":
    main()
