from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_p8_coupled_m1_synthetic_fixture.py"


def test_external_charge_prevents_empty_root_budget_reset(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="aggregate budget"):
        module._validate_budget(tmp_path / "new", tmp_path, module.forecast(),
                                external_bytes=29_500_000_000)
    receipt = module._validate_budget(tmp_path / "new", tmp_path, module.forecast(),
                                      external_bytes=4_000_000_000)
    assert receipt["projected_aggregate_bytes"] == (
        4_000_000_000 + receipt["existing_budget_root_bytes"]
        + receipt["forecast_new_bytes_with_receipt_allowance"])


def test_external_budget_receipt_binding_and_expiry(tmp_path: Path) -> None:
    module = _module()
    evidence = tmp_path / "audit.txt"
    evidence.write_text("audited upper bound")
    receipt = tmp_path / "budget.json"
    record = {"schema": "glm53.p8-coupled.external-budget.v1",
              "budget_root": str(tmp_path), "max_aggregate_bytes": 30_000_000_000,
              "external_bytes_upper_bound": 4_000_000_000,
              "measured_unix": module.time.time()-10,
              "valid_until_unix": module.time.time()+60,
              "evidence": [{"path": str(evidence), "sha256": module.sha256_file(evidence)}]}
    receipt.write_text(json.dumps(record))
    digest = module.sha256_file(receipt)
    assert module._external_budget(receipt, digest, tmp_path)["external_bytes_upper_bound"] == 4_000_000_000
    with pytest.raises(ValueError, match="identity"):
        module._external_budget(receipt, "0"*64, tmp_path)
    with pytest.raises(ValueError, match="another budget root"):
        module._external_budget(receipt, digest, tmp_path / "other")
    record["valid_until_unix"] = module.time.time()-1
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="stale"):
        module._external_budget(receipt, module.sha256_file(receipt), tmp_path)
    record["valid_until_unix"] = module.time.time()+60
    receipt.write_text(json.dumps(record))
    evidence.write_text("changed audit")
    with pytest.raises(ValueError, match="evidence identity"):
        module._external_budget(receipt, module.sha256_file(receipt), tmp_path)


def test_full_generate_requires_external_receipt_before_writes(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "fixture"
    with pytest.raises(ValueError, match="campaign-wide"):
        module.main(["--generate", "--budget-root", str(tmp_path),
                     "--output-dir", str(output)])
    assert not output.exists()


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


def test_absolute_script_execution_exposes_repository_packages() -> None:
    code = (
        "import importlib,runpy;"
        f"runpy.run_path({str(SCRIPT)!r});"
        "importlib.import_module('glm53_nvfp4.trellis_mxf');"
        "importlib.import_module('runtime_patch.p8_coupled_scales')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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


def _preserved_small_fixture(module, tmp_path: Path):
    output = tmp_path / "rank0-v1"
    output.mkdir()
    geometry = module.SMALL_GEOMETRY
    design = output / "synthetic-design.json"
    design.write_bytes(module.canonical_json_bytes(module.design_record(geometry)))
    sidecar = output / "p8-synthetic-layer-003-tp4-rank-0.safetensors"
    module._write_sidecar(
        sidecar, geometry, design_sha256=module.sha256_file(design)
    )

    evidence = tmp_path / "historical-audit.json"
    evidence.write_text("{}")
    external = tmp_path / "external-budget.json"
    external.write_text(json.dumps({
        "schema": "glm53.p8-coupled.external-budget.v1",
        "budget_root": str(tmp_path.resolve()),
        "max_aggregate_bytes": module.MAX_AGGREGATE_BYTES,
        "external_bytes_upper_bound": 4_000_000_000,
        "measured_unix": time.time() - 60,
        "valid_until_unix": time.time() + 60,
        "evidence": [{"path": str(evidence.resolve()), "sha256": "a" * 64}],
    }))
    external_sha = module.sha256_file(external)
    failure = tmp_path / "generation-failure.json"
    failure.write_text(json.dumps({
        "schema": "glm53.p8-coupled.synthetic-generation-failure.v1",
        "status": "failed_post_write_validation",
        "source_generation_commit": "test-source",
        "budget_receipt_sha256": external_sha,
        "sidecar": {
            "path": str(sidecar.resolve()), "bytes": sidecar.stat().st_size,
            "sha256": module.sha256_file(sidecar),
        },
        "design": {
            "path": str(design.resolve()), "bytes": design.stat().st_size,
            "sha256": module.sha256_file(design),
        },
        "success_receipt_created": False,
        "gpu_used": False,
        "quality_claim_allowed": False,
    }))
    return {
        "output": output,
        "sidecar": sidecar,
        "design": design,
        "external": external,
        "external_sha": external_sha,
        "failure": failure,
        "failure_sha": module.sha256_file(failure),
        "geometry": geometry,
    }


def test_resume_validates_existing_bytes_without_rewriting_payload(tmp_path: Path) -> None:
    module = _module()
    fixture = _preserved_small_fixture(module, tmp_path)
    before = {
        key: (fixture[key].stat().st_mtime_ns, module.sha256_file(fixture[key]))
        for key in ("sidecar", "design")
    }
    receipt = module.resume_validate_existing(
        fixture["output"],
        tmp_path,
        external_budget_receipt=fixture["external"],
        external_budget_sha256=fixture["external_sha"],
        failure_receipt=fixture["failure"],
        failure_receipt_sha256=fixture["failure_sha"],
        geometry=fixture["geometry"],
    )
    after = {
        key: (fixture[key].stat().st_mtime_ns, module.sha256_file(fixture[key]))
        for key in ("sidecar", "design")
    }
    assert before == after
    assert receipt["status"] == "validated_existing_payload"
    assert receipt["regenerated_or_overwritten"] is False
    assert receipt["validation"]["seeded_tensor_hash_validation"] == "pass"
    assert receipt["identities"]["sidecar"]["sha256"] == before["sidecar"][1]
    assert (fixture["output"] / "receipt.json").is_file()
    with pytest.raises(FileExistsError, match="existing receipt"):
        module.resume_validate_existing(
            fixture["output"], tmp_path,
            external_budget_receipt=fixture["external"],
            external_budget_sha256=fixture["external_sha"],
            failure_receipt=fixture["failure"],
            failure_receipt_sha256=fixture["failure_sha"],
            geometry=fixture["geometry"],
        )


def test_resume_rejects_seed_mismatch_without_receipt(tmp_path: Path) -> None:
    module = _module()
    fixture = _preserved_small_fixture(module, tmp_path)
    with fixture["sidecar"].open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    # Bind the simulated failure receipt to the corrupted artifact so the
    # deterministic recipe check, not merely the archival hash, rejects it.
    record = json.loads(fixture["failure"].read_text())
    record["sidecar"]["sha256"] = module.sha256_file(fixture["sidecar"])
    fixture["failure"].write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="seeded tensor hash mismatch"):
        module.resume_validate_existing(
            fixture["output"], tmp_path,
            external_budget_receipt=fixture["external"],
            external_budget_sha256=fixture["external_sha"],
            failure_receipt=fixture["failure"],
            failure_receipt_sha256=module.sha256_file(fixture["failure"]),
            geometry=fixture["geometry"],
        )
    assert not (fixture["output"] / "receipt.json").exists()
