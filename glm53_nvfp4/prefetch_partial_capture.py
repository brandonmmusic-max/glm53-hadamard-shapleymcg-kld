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
import time
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
RANGE_FETCH_ATTEMPTS = 8
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


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
    response = None
    error: Exception | None = None
    for attempt in range(RANGE_FETCH_ATTEMPTS):
        try:
            response = _session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 600),
            )
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            error = exc
        except requests.HTTPError as exc:
            error = exc
            if response is None or response.status_code not in RETRYABLE_HTTP_STATUS:
                raise
        if attempt + 1 == RANGE_FETCH_ATTEMPTS:
            assert error is not None
            raise error
        retry_after = response.headers.get("Retry-After") if response is not None else None
        try:
            delay = float(retry_after) if retry_after is not None else 2**attempt
        except ValueError:
            delay = 2**attempt
        time.sleep(min(max(delay, 0.0), 30.0))
    assert response is not None
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


def _sha256_range(path: Path, start: int, byte_count: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"short sparse range while validating {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _reuse_sparse(target: Path, full_bytes: int, prior: dict | None) -> dict | None:
    """Return a range-verified prior sparse record, or None if unavailable."""
    if not prior or prior.get("mode") != "sealed-role-sparse-ranges":
        return None
    if not target.is_file() or target.stat().st_size != full_bytes:
        return None
    if prior.get("path") != str(target) or prior.get("apparent_bytes") != full_bytes:
        return None
    for record in prior.get("ranges", []):
        start, end = int(record["start"]), int(record["end"])
        byte_count = end - start + 1
        if record.get("bytes") != byte_count:
            return None
        if _sha256_range(target, start, byte_count) != record.get("sha256"):
            return None
    if not prior.get("ranges"):
        return None
    value = dict(prior)
    value["allocated_bytes"] = target.stat().st_blocks * 512
    value["resume_validation"] = "all-materialized-ranges-sha256"
    value["reused_without_transfer"] = True
    return value


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
    prior: dict | None = None,
) -> dict:
    reused = _reuse_sparse(target, full_bytes, prior)
    if reused is not None:
        return reused
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
    prior_payload = None
    if args.receipt.is_file():
        candidate = json.loads(args.receipt.read_text())
        identity = (
            candidate.get("schema") == "glm53-nvfp4-v9.partial-calibration-capture.v1"
            and candidate.get("status") == "pass"
            and candidate.get("dataset") == DATASET
            and candidate.get("revision") == REVISION
            and candidate.get("layer") == args.layer
            and candidate.get("role") == args.role
            and candidate.get("window_indices") == indices
            and candidate.get("manifest_sha256") == sha256_file(manifest_path)
            and candidate.get("roles_sha256") == sha256_file(args.roles)
        )
        if identity:
            prior_payload = candidate
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
            prior=(prior_payload or {}).get("files", {}).get(key),
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
                    if not item.get("reused_without_transfer")
                ),
                "reused_bytes": sum(
                    sum(row["bytes"] for row in item["ranges"])
                    for item in outputs.values()
                    if item.get("reused_without_transfer")
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
