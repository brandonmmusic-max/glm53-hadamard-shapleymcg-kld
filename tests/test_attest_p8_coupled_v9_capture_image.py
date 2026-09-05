"""CPU-only tests for the no-build coupled-v9 capture attestation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capture_attestation",
    ROOT / "scripts/attest_p8_coupled_v9_capture_image.py",
)
attest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attest)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    context = tmp_path / "capture"
    context.mkdir()
    source_bytes = {}
    for name in attest.CAPTURE_SOURCES:
        payload = f"capture source {name}\n".encode()
        (context / name).write_bytes(payload)
        source_bytes[name] = attest.sha(context / name)
    monkeypatch.setattr(attest, "CONTEXT", context)
    monkeypatch.setattr(attest, "CAPTURE_SOURCES", source_bytes)

    runtime_sources = {"runtime_patch/example.py": "a" * 64}
    manifest = {
        "schema": "glm53.p8-coupled-image-sources.v9",
        "tail_v2": {"path": attest.TAIL_V2, "sha256": attest.TAIL_V2_SHA256},
        "source_sha256": runtime_sources,
        "install": {"runtime_patch/example.py": ["/opt/runtime/example.py"]},
    }
    manifest_raw = json.dumps(manifest, separators=(",", ":")) + "\n"
    monkeypatch.setattr(
        attest, "RUNTIME_MANIFEST_SHA256",
        __import__("hashlib").sha256(manifest_raw.encode()).hexdigest(),
    )
    state = {"calls": [], "bad_path": None, "inspect_id": attest.IMAGE,
             "bad_label": False}

    def run(argv):
        state["calls"].append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            labels = dict(attest.EXPECTED_LABELS)
            if state["bad_label"]:
                labels["org.klc.source.tree"] = "0" * 40
            return json.dumps([{
                "Id": state["inspect_id"], "Size": 123,
                "RootFS": {"Layers": ["sha256:x"]},
                "Config": {"Labels": labels},
            }])
        if "cat" in argv:
            return manifest_raw
        assert "sha256sum" in argv
        paths = argv[argv.index(attest.IMAGE) + 1:]
        expected = {
            "/opt/runtime/example.py": "a" * 64,
            attest.RUNTIME_MANIFEST: attest.RUNTIME_MANIFEST_SHA256,
            attest.TAIL_V2: attest.TAIL_V2_SHA256,
            **{f"{attest.CAPTURE_PACKAGE}/{name}": digest
               for name, digest in attest.CAPTURE_SOURCES.items()},
            **{path: attest.SAMPLER_SHA256 for path in attest.SAMPLERS},
            **{path: attest.WARMUP_SHA256 for path in attest.WARMUPS},
        }
        if state["bad_path"]:
            expected[state["bad_path"]] = "f" * 64
        return "".join(f"{expected[path]}  {path}\n" for path in paths)

    monkeypatch.setattr(attest, "_run", run)
    return tmp_path / "receipt", state


def test_success_reuses_exact_image_without_build_or_gpu(prepared):
    output, state = prepared
    receipt = attest.attest(output)
    assert receipt["status"] == "complete"
    assert receipt["schema"] == "glm53.p8-coupled-cf32-capture-image.v1"
    assert receipt["image_id"] == receipt["parent_image_id"] == attest.IMAGE
    assert receipt["reuse_existing_image"] is True
    assert receipt["image_built"] is False
    assert receipt["added_image_bytes"] == 0
    assert receipt["gpu_used"] is False
    assert receipt["speed_measurement_valid"] is False
    assert (output / "installed-sha256.txt").is_file()
    assert (output / "image-manifest.json").is_file()
    assert (output / "image-inspect.json").is_file()
    commands = [item for command in state["calls"] for item in command]
    assert "build" not in commands and "--gpus" not in commands
    for argv in state["calls"][1:]:
        assert "--runtime=runc" in argv
        assert "--network=none" in argv
        assert "--read-only" in argv
        assert "NVIDIA_VISIBLE_DEVICES=void" in argv


@pytest.mark.parametrize("failure", ["image", "label", "hash"])
def test_failure_is_preserved_and_never_claims_complete(prepared, failure):
    output, state = prepared
    if failure == "image":
        state["inspect_id"] = "sha256:" + "f" * 64
    elif failure == "label":
        state["bad_label"] = True
    else:
        state["bad_path"] = attest.TAIL_V2
    with pytest.raises(ValueError):
        attest.attest(output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"]
    assert receipt["gpu_used"] is False
    assert receipt["image_built"] is False


def test_source_drift_fails_before_docker_or_output(prepared):
    output, state = prepared
    (attest.CONTEXT / "processor.py").write_text("drift\n")
    with pytest.raises(ValueError, match="source identity"):
        attest.attest(output)
    assert not output.exists()
    assert state["calls"] == []


def test_existing_output_is_untouched(prepared):
    output, state = prepared
    output.mkdir()
    marker = output / "receipt.json"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        attest.attest(output)
    assert marker.read_text() == "preserve"
    assert state["calls"] == []


def test_runtime_expected_receipt_fields_are_present(prepared):
    output, _ = prepared
    receipt = attest.attest(output)
    required = {
        "schema", "status", "parent_image_id", "image_id",
        "runtime_manifest_sha256", "tail_v2_sha256", "gpu_used",
        "speed_measurement_valid", "installed_sha256",
    }
    assert required <= receipt.keys()
    assert set(receipt["installed_sha256"]) == {
        "p8_decode_capture/__init__.py",
        "p8_decode_capture/processor.py",
        "p8_decode_capture/v2_hook.py",
        "sampler.py",
        "warmup.py",
    }
    assert (output / "receipt.json").stat().st_size <= attest.MAX_RECEIPT_BYTES
