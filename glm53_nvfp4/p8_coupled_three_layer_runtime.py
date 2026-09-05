"""Fail-closed runtime manifest helpers for the fixed three-layer CF32 pilot.

This module performs no work on import.  It prepares launch records only; it
does not start containers, touch GPUs, score KLD, or restore production.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct


LAYERS = (3, 20, 22)
RANKS = (0, 1, 2, 3)
V9_IMAGE = "sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3"
V9_MANIFEST_SHA256 = "9a57438b3cefd022772bc471980fb0ece8c19087d08f02c875a73da2b6e392d1"
TAIL_V2_SHA256 = "494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251"
IDENTITY_DESIGN_SHA256 = "74637836d2680d3040ef000167a052c68519f89cdbc5fdeba03231c180a4d781"
COUPLED_DESIGN_SHA256 = "4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695"
TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
ROLE_SHA256 = "b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14"
CAPTURE_SOURCES = {
    "__init__.py": "267c4f5a565687a18b224d04c1f5c702149ea8ac866986dc5922c084077fcb41",
    "processor.py": "4886091710d11c00641af8074332cdd19116fa41a811be1134e7dfb713b6f287",
    "v2_hook.py": "62f35d78931a7251cd3be594d3228b0f13868f39b3e25adb5c0ad8a96c87c683",
}
SAMPLER_PATCHED_SHA256 = "717bdd2203c8977205e16ca63385027f7e7ec8acd54afd962ada3e611cf7b958"
WARMUP_PATCHED_SHA256 = "de321498f305e2f61d5cfe4701d19a066ebfec83147f527b2ec82bfb653ea8c7"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safetensors_metadata(path: Path) -> dict[str, str]:
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 2 or length > 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    value = header.get("__metadata__")
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"missing string safetensors metadata: {path}")
    return value


def _check_metadata(path: Path, expected: dict[str, str]) -> None:
    metadata = safetensors_metadata(path)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"sidecar metadata/fallback mismatch {path}: {key}")


def validate_identity_sidecars(root: Path, manifest_path: Path, design_path: Path) -> list[dict]:
    root, manifest_path, design_path = map(Path, (root, manifest_path, design_path))
    if sha(design_path) != IDENTITY_DESIGN_SHA256:
        raise ValueError("identity design identity differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("world_size") != 4 or manifest.get("design", {}).get("sha256") != IDENTITY_DESIGN_SHA256:
        raise ValueError("identity all-layer manifest design/TP differs")
    by_layer = {row.get("layer"): row for row in manifest.get("layers", [])}
    result = []
    for layer in LAYERS:
        rows = by_layer.get(layer, {}).get("ranks", [])
        if sorted(row.get("rank") for row in rows) != list(RANKS):
            raise ValueError(f"identity layer {layer} is missing or duplicates a rank")
        for row in sorted(rows, key=lambda item: item["rank"]):
            rank = row["rank"]
            path = root / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            if Path(row.get("path", "")).name != path.name or not path.is_file():
                raise ValueError(f"identity sidecar path missing for layer {layer} rank {rank}")
            if path.stat().st_size != row.get("bytes") or sha(path) != row.get("sha256"):
                raise ValueError(f"identity sidecar bytes/hash differ for layer {layer} rank {rank}")
            _check_metadata(path, {"schema": "glm53-p8-mcg-tp4-rank.v2", "boundary": "identity",
                                   "layer": str(layer), "rank": str(rank), "world_size": "4",
                                   "source_design_sha256": IDENTITY_DESIGN_SHA256})
            result.append({"layer": layer, "rank": rank, "path": str(path),
                           "bytes": row["bytes"], "sha256": row["sha256"]})
    return result


def validate_coupled_sidecars(root: Path, design_path: Path, transform_path: Path) -> list[dict]:
    root, design_path, transform_path = map(Path, (root, design_path, transform_path))
    if sha(design_path) != COUPLED_DESIGN_SHA256 or sha(transform_path) != TRANSFORM_SHA256:
        raise ValueError("coupled design/transform identity differs")
    result = []
    for layer in LAYERS:
        receipt_path = root / "receipts" / f"layer-{layer:03d}-postwrite-abi-v2.json"
        if not receipt_path.is_file():
            raise ValueError(f"missing coupled postwrite closure for layer {layer}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("layer") != layer
                or receipt.get("evidence_level") != "postwrite-source-exact-structural-closure"
                or receipt.get("chunk_retirement_gate_closed") != "postwrite-all-tensor-source-closure"
                or receipt.get("exl3_scale_source_sha256") is None):
            raise ValueError(f"coupled postwrite closure differs for layer {layer}")
        rows = receipt.get("ranks", [])
        if sorted(row.get("rank") for row in rows) != list(RANKS):
            raise ValueError(f"coupled layer {layer} is missing or duplicates a rank")
        for row in sorted(rows, key=lambda item: item["rank"]):
            rank = row["rank"]
            path = root / "sidecars" / f"layer-{layer:03d}" / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            if Path(row.get("path", "")).name != path.name or not row.get("source_exact") or not path.is_file():
                raise ValueError(f"coupled sidecar/fallback missing for layer {layer} rank {rank}")
            if path.stat().st_size != row.get("bytes") or sha(path) != row.get("sha256"):
                raise ValueError(f"coupled sidecar bytes/hash differ for layer {layer} rank {rank}")
            _check_metadata(path, {"schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
                                   "boundary": "coupled-h512-h128-suh-svh-v1", "full_coupled": "true",
                                   "layer": str(layer), "rank": str(rank), "world_size": "4",
                                   "source_design_sha256": COUPLED_DESIGN_SHA256,
                                   "encoder_transform_sha256": TRANSFORM_SHA256})
            result.append({"layer": layer, "rank": rank, "path": str(path),
                           "bytes": row["bytes"], "sha256": row["sha256"]})
    return result


def validate_capture_image_receipt(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    installed = value.get("installed_sha256", {})
    if (value.get("schema") != "glm53.p8-coupled-cf32-capture-image.v1"
            or value.get("status") != "complete" or value.get("parent_image_id") != V9_IMAGE
            or value.get("runtime_manifest_sha256") != V9_MANIFEST_SHA256
            or value.get("tail_v2_sha256") != TAIL_V2_SHA256
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", value.get("image_id", ""))
            or value.get("gpu_used") is not False or value.get("speed_measurement_valid") is not False):
        raise ValueError("coupled-v9 capture image receipt differs")
    for name, expected in CAPTURE_SOURCES.items():
        if installed.get(f"p8_decode_capture/{name}") != expected:
            raise ValueError("capture package identity differs")
    if (installed.get("sampler.py") != SAMPLER_PATCHED_SHA256
            or installed.get("warmup.py") != WARMUP_PATCHED_SHA256):
        raise ValueError("capture sampler/warmup installation differs")
    return value


def arm_environment(arm: str, window_ids: list[str]) -> dict[str, str]:
    if arm not in {"stock", "identity_p8", "coupled_p8"}:
        raise ValueError("undeclared arm")
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "GLM53_P8_DECODE_CAPTURE_V2": "1",
        "GLM53_P8_DECODE_CAPTURE_ROOT": "/p8-captures",
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS": ",".join(window_ids),
        "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS": "2047",
        "GLM53_P8_NATIVE": "", "GLM53_P8_SMALL_M": "", "GLM53_P8_FC1_TILE_N": "128",
        "GLM53_P8_FUSED_SCRATCH": "", "GLM53_P8_NATIVE_LAYERS": "",
        "GLM53_P8_NATIVE_SIDECAR_DIR": "", "GLM53_P8_NATIVE_DESIGN": "",
        "GLM53_P8_NATIVE_TRANSFORM": "",
    }
    if arm != "stock":
        env.update({"GLM53_P8_NATIVE": "1", "GLM53_P8_SMALL_M": "1",
                    "GLM53_P8_NATIVE_LAYERS": "3,20,22", "GLM53_P8_NATIVE_SIDECAR_DIR": "/p8-sidecars",
                    "GLM53_P8_NATIVE_DESIGN": "/p8-design/design.json"})
    if arm == "coupled_p8":
        env["GLM53_P8_NATIVE_TRANSFORM"] = "/p8-design/transform.json"
    return env


def verify_runtime_log(text: str, arm: str) -> dict:
    if arm == "stock":
        if "GLM53_P8_NATIVE_" in text:
            raise ValueError("stock control unexpectedly activated P8")
        return {"arm": arm, "selected_layer_rank_pairs": []}
    expected_boundary = "identity" if arm == "identity_p8" else "coupled-h512-h128-suh-svh-v1"
    expected_full = "false" if arm == "identity_p8" else "true"
    ready = re.findall(r"GLM53_P8_NATIVE_WEIGHTS_READY layer=(\d+) rank=(\d+)[^\n]*?boundary=(\S+) full_coupled=(true|false)", text)
    forwards = re.findall(r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\n]*?boundary=(\S+)", text)
    expected = {(str(layer), str(rank), expected_boundary) for layer in LAYERS for rank in RANKS}
    if {(a, b, c) for a, b, c, full in ready if full == expected_full} != expected or len(ready) != 12:
        raise ValueError("weights-ready inventory indicates missing rank/layer or fallback")
    if set(forwards) != expected or len(forwards) != 12:
        raise ValueError("native forward inventory indicates missing rank/layer or fallback")
    return {"arm": arm, "selected_layer_rank_pairs": [[layer, rank] for layer in LAYERS for rank in RANKS],
            "boundary": expected_boundary, "full_coupled": expected_full == "true"}
