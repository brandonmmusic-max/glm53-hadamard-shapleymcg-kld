"""Verify every teacher payload required by one experimental role."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-hub-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    manifest = json.loads(args.teacher_manifest.read_text())
    expected = {item["path"]: item for item in manifest["logit_files"]}
    requested = roles["roles"][args.role]

    def verify(item: dict) -> dict:
        relative = item["teacher_path"]
        entry = expected[relative]
        path = args.teacher_root / relative
        source_role = item.get("teacher_source_role", args.role)
        if (
            entry["window_id"] != item["id"]
            or entry["role"] != source_role
            or entry["domain"] != item["domain"]
            or entry["token_ids_sha256"] != item["input_sha256"]
            or entry["prediction_positions"] != item["prediction_positions"]
        ):
            raise RuntimeError(f"{item['id']}: role/manifest metadata mismatch")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"{item['id']}: missing or wrong-size teacher payload")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"{item['id']}: teacher SHA-256 mismatch")
        if args.expected_hub_revision:
            metadata = (
                args.teacher_root
                / ".cache/huggingface/download"
                / f"{relative}.metadata"
            )
            lines = metadata.read_text().splitlines()
            if (
                len(lines) < 2
                or lines[0] != args.expected_hub_revision
                or lines[1] != entry["sha256"]
            ):
                raise RuntimeError(f"{item['id']}: Hub sidecar mismatch")
        return {
            "id": item["id"],
            "domain": item["domain"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(verify, requested))
    payload = {
        "schema": "glm53-role-teacher-payload-verification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "role": args.role,
        "roles_sha256": sha256_file(args.roles),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "expected_hub_revision": args.expected_hub_revision,
        "files": files,
        "file_count": len(files),
        "payload_bytes": sum(item["bytes"] for item in files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "pass",
        "file_count": payload["file_count"],
        "payload_bytes": payload["payload_bytes"],
        "output_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
