"""Create a deterministic byte-and-SHA-256 manifest for a runtime patch tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name.endswith(".pyc"):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {
        "schema": "glm53-runtime-patch-tree.v1",
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = build_manifest(args.root)
    if args.verify:
        expected = json.loads(args.output.read_text())
        if expected != payload:
            raise SystemExit(
                "runtime patch tree does not match its frozen manifest: "
                f"expected={expected.get('files_sha256')} "
                f"actual={payload.get('files_sha256')}"
            )
        print(json.dumps({"status": "pass", "files_sha256": payload["files_sha256"]}, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes", "files_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
