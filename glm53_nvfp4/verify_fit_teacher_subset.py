"""Verify an explicitly open fit-role teacher-logit subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


TEACHER_REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--subset-label", choices=("train32", "tune32"), default="train32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    rows = roles["roles"]["fit"]
    if len(rows) != 32 or any(row.get("teacher_source_role") != "fit" for row in rows):
        raise RuntimeError("receipt is restricted to the sealed fit/train32 subset")
    manifest_path = args.teacher_root / "logits/full-panel/full-panel-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {row["path"]: row for row in manifest["logit_files"]}
    files = []
    for row in rows:
        relative = row["teacher_path"]
        path = args.teacher_root / relative
        entry = entries[relative]
        metadata = (
            args.teacher_root
            / ".cache/huggingface/download"
            / f"{relative}.metadata"
        ).read_text().splitlines()
        if (
            len(metadata) < 2
            or metadata[0] != TEACHER_REVISION
            or metadata[1] != entry["sha256"]
            or entry["window_id"] != row["id"]
            or entry["role"] != "fit"
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(f"teacher verification failed: {relative}")
        files.append(
            {
                "window_id": row["id"],
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": entry["sha256"],
                "hub_revision": metadata[0],
            }
        )
    result = {
        "schema": "glm53-rotation-v8.fit-teacher-subset.v1",
        "status": "pass",
        "role": "fit",
        "subset": args.subset_label,
        "windows": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "roles_sha256": sha256_file(args.roles),
        "teacher_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "hub_repo": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
        "hub_repo_type": "dataset",
        "hub_revision": TEACHER_REVISION,
        "files": files,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "bytes": result["bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
