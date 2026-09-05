from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_p8_coupled_m1_synthetic_fixture.py"


def _module():
    spec = importlib.util.spec_from_file_location("synthetic_p8_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_is_plan_only_with_exact_full_size() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    plan = json.loads(result.stdout)
    assert plan["synthetic"] is True
    assert plan["quality_claim_allowed"] is False
    assert plan["payload_bytes"] == 963_494_176
    assert plan["sidecar_bytes"] > plan["payload_bytes"]
    assert plan["sidecar_bytes"] < 965_000_000
    assert plan["max_aggregate_bytes"] == 30_000_000_000


def test_splitmix_stream_is_chunk_boundary_invariant_and_nonconstant() -> None:
    module = _module()
    spec = module.TensorSpec("w2_trellis", "I16", (33,), 2, "test")
    whole = b"".join(module.iter_tensor_bytes(spec, chunk_bytes=1024))
    split = b"".join(module.iter_tensor_bytes(spec, chunk_bytes=10))
    assert whole == split
    values = np.frombuffer(whole, dtype="<u2")
    assert len(set(values.tolist())) > 24


def test_small_counterpart_writes_valid_streamed_safetensors(tmp_path: Path) -> None:
    module = _module()
    geometry = module.SMALL_GEOMETRY
    design_bytes = module.canonical_json_bytes(module.design_record(geometry))
    design_sha = module.hashlib.sha256(design_bytes).hexdigest()
    output = tmp_path / "small.safetensors"
    hashes, expected = module._write_sidecar(
        output, geometry, design_sha256=design_sha
    )
    assert output.stat().st_size == expected
    result = module._validate_written(
        output, geometry, design_sha256=design_sha, tensor_hashes=hashes
    )
    assert result["runtime_cpu_schema_validation"] == "pass"
    assert result["post_write_tensor_hash_validation"] == "pass"
    assert set(result["expert0_k4_mcg_decode"]) == {"gate", "up", "down"}
    assert all(
        item["finite"] for item in result["expert0_k4_mcg_decode"].values()
    )
    with safe_open(output, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        assert metadata["synthetic_fixture"] == "true"
        assert metadata["quality_claim_allowed"] == "false"
        assert metadata["encoder_transform_sha256"] == module.TRANSFORM_SHA256
        assert source.get_slice("w13_trellis").get_shape() == [2, 2, 32, 8, 64]
        assert source.get_slice("w2_trellis").get_shape() == [2, 8, 32, 64]


def test_header_patch_keeps_offsets_and_tensor_hashes_exact(tmp_path: Path) -> None:
    module = _module()
    geometry = module.SMALL_GEOMETRY
    design_sha = "a" * 64
    output = tmp_path / "small.safetensors"
    hashes, _ = module._write_sidecar(output, geometry, design_sha256=design_sha)
    with safe_open(output, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name, digest in hashes.items():
            assert metadata[f"sha256_{name}"] == digest
        assert metadata["source_design_sha256"] == design_sha


def test_full_generation_requires_explicit_fresh_budgeted_destination(tmp_path: Path) -> None:
    module = _module()
    plan = module.forecast()
    existing = tmp_path / "existing"
    existing.mkdir()
    try:
        module._validate_budget(existing, tmp_path, plan)
    except ValueError as error:
        assert "fresh absolute" in str(error)
    else:
        raise AssertionError("existing output must be rejected")
