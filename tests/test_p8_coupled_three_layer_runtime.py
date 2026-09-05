import hashlib
import json
from pathlib import Path
import struct

import pytest

from glm53_nvfp4 import p8_coupled_three_layer_runtime as runtime


def _safe(path: Path, metadata: dict[str, str], payload: bytes = b"x") -> tuple[int, str]:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode()
    pad = (-len(header)) % 8
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header) + pad) + header + b" " * pad + payload)
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def _coupled_tree(tmp_path: Path):
    design, transform = tmp_path / "design.json", tmp_path / "transform.json"
    design.write_bytes(b"design")
    transform.write_bytes(b"transform")
    old_d, old_t = runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256
    runtime.COUPLED_DESIGN_SHA256 = hashlib.sha256(b"design").hexdigest()
    runtime.TRANSFORM_SHA256 = hashlib.sha256(b"transform").hexdigest()
    for layer in runtime.LAYERS:
        rows = []
        for rank in runtime.RANKS:
            path = tmp_path / "sidecars" / f"layer-{layer:03d}" / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            size, digest = _safe(path, {"schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
                "boundary": "coupled-h512-h128-suh-svh-v1", "full_coupled": "true",
                "layer": str(layer), "rank": str(rank), "world_size": "4",
                "source_design_sha256": runtime.COUPLED_DESIGN_SHA256,
                "encoder_transform_sha256": runtime.TRANSFORM_SHA256})
            rows.append({"rank": rank, "path": str(path), "bytes": size, "sha256": digest, "source_exact": True})
        receipt = {"layer": layer, "evidence_level": "postwrite-source-exact-structural-closure",
            "chunk_retirement_gate_closed": "postwrite-all-tensor-source-closure", "exl3_scale_source_sha256": "x", "ranks": rows}
        p = tmp_path / "receipts" / f"layer-{layer:03d}-postwrite-abi-v2.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(receipt))
    return design, transform, old_d, old_t


def test_coupled_sidecars_all_layers_and_ranks(tmp_path):
    design, transform, old_d, old_t = _coupled_tree(tmp_path)
    try:
        rows = runtime.validate_coupled_sidecars(tmp_path, design, transform)
        assert {(r["layer"], r["rank"]) for r in rows} == set(__import__("itertools").product(runtime.LAYERS, runtime.RANKS))
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256 = old_d, old_t


def test_coupled_missing_rank_fails(tmp_path):
    design, transform, old_d, old_t = _coupled_tree(tmp_path)
    try:
        receipt = tmp_path / "receipts/layer-020-postwrite-abi-v2.json"
        value = json.loads(receipt.read_text()); value["ranks"].pop(); receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="missing or duplicates"):
            runtime.validate_coupled_sidecars(tmp_path, design, transform)
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256 = old_d, old_t


def test_coupled_identity_fallback_fails(tmp_path):
    design, transform, old_d, old_t = _coupled_tree(tmp_path)
    try:
        target = tmp_path / "sidecars/layer-022/p8-layer-022-tp4-rank-2.safetensors"
        size, digest = _safe(target, {"schema": "glm53-p8-mcg-tp4-rank.v2", "boundary": "identity"})
        receipt = tmp_path / "receipts/layer-022-postwrite-abi-v2.json"
        value = json.loads(receipt.read_text()); value["ranks"][2].update(bytes=size, sha256=digest); receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="metadata/fallback"):
            runtime.validate_coupled_sidecars(tmp_path, design, transform)
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256 = old_d, old_t


def _log(arm: str) -> str:
    boundary = "identity" if arm == "identity_p8" else "coupled-h512-h128-suh-svh-v1"
    full = "false" if arm == "identity_p8" else "true"
    lines = []
    for layer in runtime.LAYERS:
        for rank in runtime.RANKS:
            lines += [f"GLM53_P8_NATIVE_WEIGHTS_READY layer={layer} rank={rank} x boundary={boundary} full_coupled={full}",
                      f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} x boundary={boundary}"]
    return "\n".join(lines)


def test_log_gate_rejects_missing_pair_and_wrong_boundary():
    text = _log("coupled_p8")
    assert runtime.verify_runtime_log(text, "coupled_p8")["full_coupled"] is True
    with pytest.raises(ValueError, match="missing rank/layer"):
        runtime.verify_runtime_log(text.replace("GLM53_P8_NATIVE_WEIGHTS_READY layer=22 rank=3", "REMOVED", 1), "coupled_p8")
    with pytest.raises(ValueError, match="fallback"):
        runtime.verify_runtime_log(text.replace("boundary=coupled-h512-h128-suh-svh-v1", "boundary=identity", 1), "coupled_p8")


def test_arm_environment_stock_is_explicitly_off_and_coupled_exact():
    stock = runtime.arm_environment("stock", ["conditional-fit-0001"])
    assert stock["GLM53_P8_NATIVE"] == "" and stock["GLM53_P8_NATIVE_TRANSFORM"] == ""
    coupled = runtime.arm_environment("coupled_p8", ["conditional-fit-0001"])
    assert coupled["GLM53_P8_NATIVE_LAYERS"] == "3,20,22"
    assert coupled["GLM53_P8_FC1_TILE_N"] == "128"
    assert coupled["GLM53_P8_FUSED_SCRATCH"] == ""
    assert coupled["GLM53_P8_NATIVE_TRANSFORM"] == "/p8-design/transform.json"


def test_capture_receipt_rejects_non_v9_parent(tmp_path):
    receipt = {"schema": "glm53.p8-coupled-cf32-capture-image.v1", "status": "complete",
        "parent_image_id": "sha256:" + "0" * 64, "image_id": "sha256:" + "1" * 64,
        "runtime_manifest_sha256": runtime.V9_MANIFEST_SHA256, "tail_v2_sha256": runtime.TAIL_V2_SHA256,
        "gpu_used": False, "speed_measurement_valid": False, "installed_sha256": {}}
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt differs"):
        runtime.validate_capture_image_receipt(path)


def test_capture_dockerfile_composes_on_v9_and_is_numerical_only():
    source = (Path(__file__).parents[1] / "runtime_patch/p8_decode_capture/Dockerfile.coupled-v9-control").read_text()
    assert f"FROM {runtime.V9_IMAGE}" in source
    assert runtime.V9_MANIFEST_SHA256 in source
    assert runtime.SAMPLER_PATCHED_SHA256 in source
    assert runtime.WARMUP_PATCHED_SHA256 in source
    assert 'qualification="numerical-capture-only-not-speed"' in source
