"""Download only preregistered non-final teacher files from the pinned Hub revision."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
METADATA = [
    "README.md",
    "backend.json",
    "plan.json",
    "source-inventory.json",
    "token-panel-receipt.json",
    "logits/full-panel/full-panel-manifest.json",
    "logits/full-panel/full-panel-hf-verification.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--role", action="append", choices=("conditional-fit", "selection", "confirmation"), required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.roles.read_text())
    files = list(METADATA)
    for role in args.role:
        files.extend(item["teacher_path"] for item in manifest["roles"][role])
    if len(files) != len(set(files)):
        raise RuntimeError("duplicate teacher download path")
    command = [
        "/home/brandonmusic/.local/bin/hf", "download", REPO, *files,
        "--type", "dataset", "--revision", REVISION,
        "--local-dir", str(args.local_dir), "--max-workers", str(args.max_workers), "--format", "agent",
    ]
    print(json.dumps({"repo": REPO, "revision": REVISION, "roles": args.role, "files": len(files)}, sort_keys=True), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
