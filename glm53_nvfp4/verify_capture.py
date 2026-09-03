"""Verify one streamed capture layer against the pinned capture manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.capture_root / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["files"][str(args.layer)]
    files = []
    for key in ("hidden_bf16", "topk_ids_u16le", "topk_weights_f32le"):
        expected = entries[key]
        path = args.capture_root / expected["path"]
        observed = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        observed["expected_bytes"] = expected["bytes"]
        observed["expected_sha256"] = expected["sha256"]
        observed["status"] = "pass" if observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"] else "fail"
        files.append(observed)
    payload = {
        "schema": "glm53-nvfp4-v2.capture-integrity.v1",
        "layer": args.layer,
        "model_revision": manifest["model_revision"],
        "capture_manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "status": "pass" if all(item["status"] == "pass" for item in files) else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"layer": args.layer, "status": payload["status"]}, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
