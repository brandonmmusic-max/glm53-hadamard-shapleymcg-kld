"""Fail-closed runtime manifest helpers for the full-model (layers 3..44) CF32 arms.

Two arms share one immutable v10 image and one serving recipe:

``coupled_full``
    all 42 routed layers native P8 with the coupled H512/H128 + suh/svh boundary;
    sidecars come from the full coupled build manifest and may carry two encoder
    design hashes (the 42-layer preparation and the pilot V3 design for the reused
    layers 3, 20 and 22).
``identity_full``
    all 42 routed layers native P8 with the identity boundary from the pinned
    uniform checkpoint, re-measured on the same image so the pair is matched.

This module performs no work on import and never touches a GPU, a container,
or production.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .full_coupled_build_support import MANIFEST_SCHEMA as FULL_COUPLED_MANIFEST_SCHEMA
from .full_coupled_build_support import safetensors_header

LAYERS = tuple(range(3, 45))
RANKS = (0, 1, 2, 3)
ARMS = ("coupled_full", "identity_full")
PAIRS = len(LAYERS) * len(RANKS)
IDENTITY_MANIFEST_SCHEMA = "glm53-p8-uniform-all42-tp4-checkpoint.v1"
IDENTITY_DESIGN_SHA256 = "74637836d2680d3040ef000167a052c68519f89cdbc5fdeba03231c180a4d781"
TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
ROLE_SHA256 = "b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14"
V10_PYTHONPATH = "/opt/p8-coupled-runtime:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x"
BOUNDARIES = {"coupled_full": "coupled-h512-h128-suh-svh-v1", "identity_full": "identity"}
FULL_COUPLED_FLAG = {"coupled_full": "true", "identity_full": "false"}
LAYER_SPEC = ",".join(str(layer) for layer in LAYERS)
SERVED_NAME = "glm53-p8-full-cf32-{arm}"


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm_environment(arm: str, window_ids: list[str], *, design_count: int) -> dict[str, str]:
    """Container environment for one arm; every P8 flag is set explicitly."""
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    if design_count < 1:
        raise ValueError("at least one design file is required")
    designs = ":".join(f"/p8-design/design-{index}.json" for index in range(design_count))
    env = {
        "PYTHONPATH": V10_PYTHONPATH,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "GLM53_P8_DECODE_CAPTURE_V2": "1",
        "GLM53_P8_DECODE_CAPTURE_ROOT": "/p8-captures",
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS": ",".join(window_ids),
        "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS": "2047",
        "GLM53_P8_NATIVE": "1",
        "GLM53_P8_SMALL_M": "1",
        "GLM53_P8_FC1_TILE_N": "128",
        "GLM53_P8_FUSED_SCRATCH": "",
        "GLM53_P8_NATIVE_LAYERS": LAYER_SPEC,
        "GLM53_P8_NATIVE_SIDECAR_DIR": "/p8-sidecars",
        "GLM53_P8_NATIVE_DESIGN": designs,
        "GLM53_P8_NATIVE_TRANSFORM": "/p8-design/transform.json" if arm == "coupled_full" else "",
    }
    return env


def verify_runtime_log(text: str, arm: str, *, design_by_layer: dict[int, str]) -> dict:
    """Require every layer/rank pair to report the intended boundary and design."""
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    if set(design_by_layer) != set(LAYERS):
        raise ValueError("design_by_layer must cover exactly layers 3..44")
    expected_boundary = BOUNDARIES[arm]
    expected_full = FULL_COUPLED_FLAG[arm]
    ready = re.findall(
        r"GLM53_P8_NATIVE_WEIGHTS_READY layer=(\d+) rank=(\d+) sidecar=\S+ "
        r"design_sha256=([0-9a-f]{64})[^\n]*?boundary=(\S+) full_coupled=(true|false)",
        text,
    )
    forwards = re.findall(
        r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\n]*?boundary=(\S+)", text
    )
    if len(ready) != PAIRS:
        raise ValueError(f"weights-ready inventory has {len(ready)} lines, expected {PAIRS}")
    seen = set()
    for layer_text, rank_text, design, boundary, full in ready:
        layer, rank = int(layer_text), int(rank_text)
        if layer not in design_by_layer or rank not in RANKS:
            raise ValueError(f"unexpected weights-ready pair layer={layer} rank={rank}")
        if design != design_by_layer[layer]:
            raise ValueError(f"layer {layer} rank {rank} loaded design {design}, expected {design_by_layer[layer]}")
        if boundary != expected_boundary or full != expected_full:
            raise ValueError(f"layer {layer} rank {rank} boundary/full_coupled differs from {arm}")
        seen.add((layer, rank))
    if seen != {(layer, rank) for layer in LAYERS for rank in RANKS}:
        raise ValueError("weights-ready inventory does not cover every layer/rank exactly once")
    if len(forwards) != PAIRS or {(int(a), int(b), c) for a, b, c in forwards} != {
        (layer, rank, expected_boundary) for layer in LAYERS for rank in RANKS
    }:
        raise ValueError("native forward inventory indicates missing rank/layer or fallback")
    if "GLM53_P8_NATIVE_PATCH_ACTIVE layers=" + LAYER_SPEC + " " not in text:
        raise ValueError("native patch activation line for layers 3..44 is missing")
    return {
        "arm": arm,
        "pairs": PAIRS,
        "boundary": expected_boundary,
        "full_coupled": expected_full == "true",
        "designs": sorted(set(design_by_layer.values())),
    }


def validate_full_coupled_manifest(path: Path, *, sidecar_dir: Path, transform: Path) -> dict:
    """Authenticate the full coupled checkpoint manifest and return its design map."""
    path, sidecar_dir, transform = Path(path), Path(sidecar_dir), Path(transform)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != FULL_COUPLED_MANIFEST_SCHEMA or manifest.get("layer_range") != [3, 44]:
        raise ValueError("full coupled manifest schema/layer range differs")
    if sha(transform) != TRANSFORM_SHA256 or manifest.get("transform", {}).get("sha256") != TRANSFORM_SHA256:
        raise ValueError("coupled transform identity differs")
    payload = manifest.get("payload", {})
    if payload.get("weight_payload_bpw") != 4.25:
        raise ValueError("full coupled manifest is not exactly 4.25 weight-payload bpw")
    layers = manifest.get("layers", [])
    if [row.get("layer") for row in layers] != list(LAYERS):
        raise ValueError("full coupled manifest does not list layers 3..44 in order")
    design_by_layer: dict[int, str] = {}
    files = []
    for row in layers:
        if row.get("status") != "pass" or [r.get("rank") for r in row.get("ranks", [])] != list(RANKS):
            raise ValueError(f"layer {row.get('layer')} lacks a passing four-rank receipt")
        design_by_layer[int(row["layer"])] = row["source_design_sha256"]
        for rank in row["ranks"]:
            expected = sidecar_dir / f"p8-layer-{row['layer']:03d}-tp4-rank-{rank['rank']}.safetensors"
            if Path(rank["path"]) != expected.resolve() or not expected.is_file():
                raise ValueError(f"sidecar path differs for layer {row['layer']} rank {rank['rank']}")
            if expected.stat().st_size != rank["bytes"]:
                raise ValueError(f"sidecar bytes differ for layer {row['layer']} rank {rank['rank']}")
            metadata, _, _ = safetensors_header(expected)
            if (metadata.get("source_design_sha256") != rank["source_design_sha256"]
                    or metadata.get("boundary") != BOUNDARIES["coupled_full"]
                    or metadata.get("full_coupled") != "true"
                    or metadata.get("encoder_transform_sha256") != TRANSFORM_SHA256):
                raise ValueError(f"sidecar metadata differs for layer {row['layer']} rank {rank['rank']}")
            files.append({"layer": row["layer"], "rank": rank["rank"], "path": str(expected),
                          "bytes": rank["bytes"], "sha256": rank["sha256"]})
    return {"manifest_sha256": sha(path), "design_by_layer": design_by_layer,
            "designs": manifest.get("designs"), "payload": payload, "files": files}


def validate_identity_manifest(path: Path, *, sidecar_dir: Path, design: Path) -> dict:
    """Authenticate the pinned uniform identity checkpoint for the matched control."""
    path, sidecar_dir, design = Path(path), Path(sidecar_dir), Path(design)
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != IDENTITY_MANIFEST_SCHEMA or manifest.get("layer_range") != [3, 44]
            or manifest.get("codec", {}).get("bits") != 4 or manifest.get("codec", {}).get("ldlq") is not False
            or manifest.get("payload", {}).get("bpw") != 4.25):
        raise ValueError("identity manifest schema/codec/rate differs")
    design_sha = sha(design)
    if design_sha != IDENTITY_DESIGN_SHA256 or manifest.get("design", {}).get("sha256") != design_sha:
        raise ValueError("identity design identity differs")
    layers = manifest.get("layers", [])
    if [row.get("layer") for row in layers] != list(LAYERS):
        raise ValueError("identity manifest does not list layers 3..44 in order")
    files = []
    for row in layers:
        if [r.get("rank") for r in row.get("ranks", [])] != list(RANKS):
            raise ValueError(f"identity layer {row.get('layer')} lacks four ranks")
        for rank in row["ranks"]:
            expected = sidecar_dir / f"p8-layer-{row['layer']:03d}-tp4-rank-{rank['rank']}.safetensors"
            if Path(rank["path"]) != expected.resolve() or not expected.is_file():
                raise ValueError(f"identity sidecar path differs for layer {row['layer']} rank {rank['rank']}")
            if expected.stat().st_size != rank["bytes"] or rank.get("payload_bpw") != 4.25:
                raise ValueError(f"identity sidecar bytes/rate differ for layer {row['layer']} rank {rank['rank']}")
            metadata, _, _ = safetensors_header(expected)
            if metadata.get("source_design_sha256") != design_sha or metadata.get("boundary") != "identity":
                raise ValueError(f"identity sidecar metadata differs for layer {row['layer']} rank {rank['rank']}")
            files.append({"layer": row["layer"], "rank": rank["rank"], "path": str(expected),
                          "bytes": rank["bytes"], "sha256": rank["sha256"]})
    return {"manifest_sha256": sha(path), "design_sha256": design_sha,
            "design_by_layer": {layer: design_sha for layer in LAYERS}, "files": files}
