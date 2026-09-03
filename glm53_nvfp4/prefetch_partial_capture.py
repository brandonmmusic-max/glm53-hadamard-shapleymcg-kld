"""Materialize only the sealed fit-window byte ranges of a capture layer.

The published capture payloads contain 640 fixed 2,048-token windows.  A
campaign role uses only a sealed subset.  This downloader creates exact-size
sparse files, fetches the selected ranges from the immutable Hub revision, and
leaves all unselected ranges as filesystem holes.  ``LayerCapture`` only reads
the selected windows, so its observed bytes and sample order are identical to
the full payload without transferring unused calibration roles.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
from pathlib import Path

import requests

from .shard_index import sha256_file


DATASET = "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
WINDOW_ROWS = 2048
PAYLOADS = {
    "hidden_bf16": ("hidden.bf16.bin", 2),
    "topk_ids_u16le": ("topk_ids.u16le.bin", 2),
    "topk_weights_f32le": ("topk_weights.f32le.bin", 4),
}


def merge_windows(indices: list[int]) -> list[tuple[int, int]]:
    """Return inclusive runs of consecutive window indices."""
    if not indices:
        raise ValueError("at least one window is required")
    values = sorted(set(indices))
    result: list[tuple[int, int]] = []
    first = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((first, previous))
        first = previous = value
    result.append((first, previous))
    return result


def _url(layer: int, relative_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
        f"calibration/main-ep4-full/{relative_path}"
    )


_thread_local = threading.local()


def _session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        _thread_local.session = value
    return value


def _fetch_range(url: str, start: int, end: int) -> tuple[int, bytes, str]:
    response = _session().get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(30, 600),
    )
    response.raise_for_status()
    expected = end - start + 1
    content_range = response.headers.get("Content-Range", "")
    if response.status_code != 206 or len(response.content) != expected:
        raise RuntimeError(
            f"range fetch failed status={response.status_code} "
            f"bytes={len(response.content)} expected={expected}"
        )
    if not content_range.startswith(f"bytes {start}-{end}/"):
        raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
    return start, response.content, hashlib.sha256(response.content).hexdigest()


def _materialize(
    *,
    layer: int,
    target: Path,
    relative_path: str,
    full_bytes: int,
    row_bytes: int,
    runs: list[tuple[int, int]],
    workers: int,
    expected_sha256: str,
) -> dict:
    if target.is_file() and target.stat().st_size == full_bytes:
        # A complete file may already have been streamed by an earlier run.
        # Verify it against the publication rather than replacing it.
        actual = sha256_file(target)
        if actual != expected_sha256:
            raise RuntimeError(f"existing capture hash mismatch: {target}")
        return {
            "path": str(target),
            "mode": "complete-file",
            "apparent_bytes": full_bytes,
            "allocated_bytes": target.stat().st_blocks * 512,
            "sha256": actual,
            "ranges": [],
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".fit-partial")
    if temporary.exists():
        temporary.unlink()
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        os.ftruncate(descriptor, full_bytes)
        requests_to_make = []
        for first, last in runs:
            start = first * WINDOW_ROWS * row_bytes
            end = (last + 1) * WINDOW_ROWS * row_bytes - 1
            requests_to_make.append((start, end))
        url = _url(layer, relative_path)
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fetch_range, url, start, end)
                for start, end in requests_to_make
            ]
            for future, (_, end) in zip(futures, requests_to_make):
                start, content, digest = future.result()
                os.pwrite(descriptor, content, start)
                records.append(
                    {
                        "start": start,
                        "end": end,
                        "bytes": len(content),
                        "sha256": digest,
                    }
                )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    stat = target.stat()
    return {
        "path": str(target),
        "mode": "sealed-role-sparse-ranges",
        "apparent_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
        "published_full_sha256": expected_sha256,
        "ranges": sorted(records, key=lambda item: item["start"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--role", default="fit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.layer <= 44 or args.workers <= 0:
        raise ValueError("invalid layer or worker count")

    manifest_path = args.capture_root / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    roles = json.loads(args.roles.read_text())["roles"]
    role_items = roles[args.role]
    windows = {item["window_id"]: item for item in manifest["windows"]}
    indices = sorted(windows[item["id"]]["window_index"] for item in role_items)
    for item in role_items:
        if item["input_sha256"] != windows[item["id"]]["token_ids_sha256"]:
            raise RuntimeError(f"role/capture token mismatch: {item['id']}")
    runs = merge_windows(indices)
    info = manifest["files"][str(args.layer)]
    abi = manifest["file_abi"]
    layer_root = args.capture_root / f"layers/layer-{args.layer:03d}"
    outputs = {}
    for key, (name, element_bytes) in PAYLOADS.items():
        shape = abi[key]["shape"]
        row_bytes = int(shape[1]) * element_bytes
        outputs[key] = _materialize(
            layer=args.layer,
            target=layer_root / name,
            relative_path=info[key]["path"],
            full_bytes=int(info[key]["bytes"]),
            row_bytes=row_bytes,
            runs=runs,
            workers=args.workers,
            expected_sha256=info[key]["sha256"],
        )

    payload = {
        "schema": "glm53-nvfp4-v9.partial-calibration-capture.v1",
        "status": "pass",
        "dataset": DATASET,
        "revision": REVISION,
        "layer": args.layer,
        "role": args.role,
        "windows": len(indices),
        "window_indices": indices,
        "runs": [[first, last] for first, last in runs],
        "manifest_sha256": sha256_file(manifest_path),
        "roles_sha256": sha256_file(args.roles),
        "files": outputs,
        "observed_contract": (
            "Only sealed role ranges are materialized; LayerCapture reads no "
            "unselected range. Apparent sizes retain the published ABI."
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "layer": args.layer,
                "status": "pass",
                "windows": len(indices),
                "downloaded_bytes": sum(
                    sum(row["bytes"] for row in item["ranges"])
                    for item in outputs.values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
