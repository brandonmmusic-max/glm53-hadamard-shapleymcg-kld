import hashlib
import json
import struct
from pathlib import Path

import pytest

from glm53_nvfp4 import p8_full_coupled_runtime as runtime
from glm53_nvfp4.full_coupled_build_support import MANIFEST_SCHEMA


DESIGN_A = "a" * 64
DESIGN_B = "b" * 64


def _log(arm: str, design_by_layer: dict[int, str], *, drop=None, wrong_boundary=None) -> str:
    boundary = runtime.BOUNDARIES[arm]
    full = runtime.FULL_COUPLED_FLAG[arm]
    lines = [f"GLM53_P8_NATIVE_PATCH_ACTIVE layers={runtime.LAYER_SPEC} tp=4 design_sha256_allowlist=x K4"]
    for layer in runtime.LAYERS:
        for rank in runtime.RANKS:
            if drop == (layer, rank):
                continue
            b = "identity" if wrong_boundary == (layer, rank) else boundary
            lines.append(
                f"GLM53_P8_NATIVE_WEIGHTS_READY layer={layer} rank={rank} sidecar=/p8-sidecars/x.safetensors "
                f"design_sha256={design_by_layer[layer]} released_carrier_bytes=1 design_allowlist_size=2 "
                f"transform_sha256=t stream=K4 law=mcg alphabet=E4M3 scale=UE8M0_K32 boundary={b} "
                f"full_coupled={full} ldlq=false small_m_scheduler=true"
            )
            lines.append(
                f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} stream=K4 mma=mxf8f6f4 alphabet=E4M3 "
                f"scale=UE8M0_K32 law=procedural_mcg boundary={b} deterministic=route_topk_sum physical_bpw=4.25 ldlq=false"
            )
    return "\n".join(lines)


def _designs(mixed: bool) -> dict[int, str]:
    return {layer: (DESIGN_B if mixed and layer in (3, 20, 22) else DESIGN_A) for layer in runtime.LAYERS}


def test_runtime_log_gate_accepts_full_inventory_with_mixed_designs():
    designs = _designs(mixed=True)
    result = runtime.verify_runtime_log(_log("coupled_full", designs), "coupled_full", design_by_layer=designs)
    assert result["pairs"] == 168 and result["full_coupled"] is True and result["designs"] == [DESIGN_A, DESIGN_B]
    identity = {layer: DESIGN_A for layer in runtime.LAYERS}
    result = runtime.verify_runtime_log(_log("identity_full", identity), "identity_full", design_by_layer=identity)
    assert result["boundary"] == "identity" and result["full_coupled"] is False


def test_runtime_log_gate_fails_closed_on_missing_pair_wrong_design_or_boundary():
    designs = _designs(mixed=True)
    with pytest.raises(ValueError, match="expected 168"):
        runtime.verify_runtime_log(_log("coupled_full", designs, drop=(17, 2)), "coupled_full", design_by_layer=designs)
    swapped = dict(designs); swapped[30] = DESIGN_B
    with pytest.raises(ValueError, match="loaded design"):
        runtime.verify_runtime_log(_log("coupled_full", designs), "coupled_full", design_by_layer=swapped)
    with pytest.raises(ValueError, match="boundary/full_coupled differs"):
        runtime.verify_runtime_log(_log("coupled_full", designs, wrong_boundary=(9, 0)), "coupled_full", design_by_layer=designs)
    # A coupled log offered as the identity arm must fail on every pair.
    with pytest.raises(ValueError, match="boundary/full_coupled differs"):
        runtime.verify_runtime_log(_log("coupled_full", designs), "identity_full", design_by_layer=designs)
    with pytest.raises(ValueError, match="exactly layers 3..44"):
        runtime.verify_runtime_log(_log("coupled_full", designs), "coupled_full", design_by_layer={3: DESIGN_A})


def test_arm_environment_sets_every_p8_flag_explicitly():
    env = runtime.arm_environment("coupled_full", ["w1", "w2"], design_count=2)
    assert env["GLM53_P8_NATIVE_LAYERS"] == ",".join(str(l) for l in range(3, 45))
    assert env["GLM53_P8_NATIVE_DESIGN"] == "/p8-design/design-0.json:/p8-design/design-1.json"
    assert env["GLM53_P8_NATIVE_TRANSFORM"] == "/p8-design/transform.json"
    assert env["GLM53_P8_SMALL_M"] == "1" and env["GLM53_P8_FC1_TILE_N"] == "128" and env["GLM53_P8_FUSED_SCRATCH"] == ""
    identity = runtime.arm_environment("identity_full", ["w1"], design_count=1)
    assert identity["GLM53_P8_NATIVE_DESIGN"] == "/p8-design/design-0.json"
    assert identity["GLM53_P8_NATIVE_TRANSFORM"] == ""
    assert identity["GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS"] == "w1"
    with pytest.raises(ValueError):
        runtime.arm_environment("stock", ["w1"], design_count=1)


def _sidecar(path: Path, metadata: dict[str, str]) -> tuple[int, str]:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"payload")
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_coupled_manifest_validation_binds_paths_designs_and_transform(tmp_path: Path, monkeypatch):
    transform = tmp_path / "transform.json"; transform.write_bytes(b"transform")
    monkeypatch.setattr(runtime, "TRANSFORM_SHA256", hashlib.sha256(b"transform").hexdigest())
    sidecars = tmp_path / "sidecars"
    layers = []
    for layer in runtime.LAYERS:
        design = DESIGN_B if layer in (3, 20, 22) else DESIGN_A
        ranks = []
        for rank in runtime.RANKS:
            path = sidecars / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            size, digest = _sidecar(path, {"source_design_sha256": design, "boundary": runtime.BOUNDARIES["coupled_full"],
                                           "full_coupled": "true", "bits": "4", "encoder_transform_sha256": runtime.TRANSFORM_SHA256})
            ranks.append({"rank": rank, "path": str(path.resolve()), "bytes": size, "sha256": digest,
                          "source_design_sha256": design})
        layers.append({"layer": layer, "status": "pass", "ranks": ranks, "source_design_sha256": design})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "layer_range": [3, 44],
                                    "transform": {"sha256": runtime.TRANSFORM_SHA256},
                                    "payload": {"weight_payload_bpw": 4.25}, "layers": layers,
                                    "designs": {DESIGN_B: [3, 20, 22]}}))
    result = runtime.validate_full_coupled_manifest(manifest, sidecar_dir=sidecars, transform=transform)
    assert result["design_by_layer"][20] == DESIGN_B and result["design_by_layer"][21] == DESIGN_A
    assert len(result["files"]) == 168
    # Tamper with one sidecar's metadata design -> fail closed.
    victim = sidecars / "p8-layer-010-tp4-rank-3.safetensors"
    _sidecar(victim, {"source_design_sha256": DESIGN_B, "boundary": runtime.BOUNDARIES["coupled_full"],
                      "full_coupled": "true", "bits": "4", "encoder_transform_sha256": runtime.TRANSFORM_SHA256})
    with pytest.raises(ValueError, match="differ"):
        runtime.validate_full_coupled_manifest(manifest, sidecar_dir=sidecars, transform=transform)


def test_identity_manifest_validation_requires_pinned_design(tmp_path: Path, monkeypatch):
    design = tmp_path / "native6.json"; design.write_bytes(b"native6")
    design_sha = hashlib.sha256(b"native6").hexdigest()
    monkeypatch.setattr(runtime, "IDENTITY_DESIGN_SHA256", design_sha)
    sidecars = tmp_path / "identity"
    layers = []
    for layer in runtime.LAYERS:
        ranks = []
        for rank in runtime.RANKS:
            path = sidecars / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            size, digest = _sidecar(path, {"source_design_sha256": design_sha, "boundary": "identity"})
            ranks.append({"rank": rank, "path": str(path.resolve()), "bytes": size, "sha256": digest, "payload_bpw": 4.25})
        layers.append({"layer": layer, "ranks": ranks})
    manifest = tmp_path / "identity-manifest.json"
    body = {"schema": runtime.IDENTITY_MANIFEST_SCHEMA, "layer_range": [3, 44],
            "codec": {"bits": 4, "ldlq": False}, "payload": {"bpw": 4.25},
            "design": {"sha256": design_sha}, "layers": layers}
    manifest.write_text(json.dumps(body))
    result = runtime.validate_identity_manifest(manifest, sidecar_dir=sidecars, design=design)
    assert result["design_sha256"] == design_sha and len(result["files"]) == 168
    body["design"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identity design identity differs"):
        runtime.validate_identity_manifest(manifest, sidecar_dir=sidecars, design=design)
