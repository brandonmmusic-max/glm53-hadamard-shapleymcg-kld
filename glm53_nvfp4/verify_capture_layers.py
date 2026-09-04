"""Verify selected local calibration layers against the frozen capture manifest."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest_path = args.capture_root / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requests = []
    for layer in args.layer:
        for kind, expected in manifest["files"][str(layer)].items():
            requests.append((layer, kind, expected))

    def verify(request: tuple[int, str, dict]) -> dict:
        layer, kind, expected = request
        path = args.capture_root / expected["path"]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"layer {layer} {kind}: missing or wrong size")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"layer {layer} {kind}: SHA-256 mismatch")
        return {"layer": layer, "kind": kind, "path": str(path), "bytes": path.stat().st_size, "sha256": actual}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(verify, requests))
    payload = {
        "schema": "glm53-capture-layer-verification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "capture_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "layers": sorted(args.layer),
        "files": files,
        "payload_bytes": sum(item["bytes"] for item in files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "layers": payload["layers"], "payload_bytes": payload["payload_bytes"], "output_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
