"""CPU-only builder checks: all Docker calls are mocked, never executed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capture_image_builder", REPO / "scripts/build_p8_decode_capture_image.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    context = tmp_path / "context"
    context.mkdir()
    for name in builder.SOURCE_NAMES:
        (context / name).write_text(f"test source {name}\n")
    output = tmp_path / "receipt"
    monkeypatch.setattr(builder, "CONTEXT", context)
    monkeypatch.setattr(sys, "argv", ["builder", "--output", str(output)])
    state = {"calls": [], "existing": False, "parent_reads": 0,
             "parent_mismatch": False, "parent_changed": False,
             "build_failure": False, "source_changed": False,
             "bad_hash": None, "missing_hash": None}

    def run(argv, **kwargs):
        state["calls"].append((argv, kwargs))
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0 if state["existing"] else 1)
        assert argv[:2] == ["docker", "build"], "unexpected subprocess"
        assert kwargs["env"]["DOCKER_BUILDKIT"] == "0"
        kwargs["stdout"].write("mock build output\n")
        if state["build_failure"]:
            raise subprocess.CalledProcessError(1, argv)
        if state["source_changed"]:
            (context / "v2_hook.py").write_text("changed during build\n")
        return subprocess.CompletedProcess(argv, 0)

    def command(argv):
        state["calls"].append((argv, {}))
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[3] == builder.PARENT_TAG:
                state["parent_reads"] += 1
                if state["parent_mismatch"] or (state["parent_changed"] and state["parent_reads"] > 1):
                    return "sha256:" + "f" * 64
                return builder.PARENT_ID
            return "sha256:" + "b" * 64
        assert argv[:2] == ["docker", "run"], "unexpected command"
        hashes = {name: builder.SAMPLER_PATCHED_SHA for name in builder.SAMPLERS}
        hashes.update({name: builder.WARMUP_PATCHED_SHA for name in builder.WARMUPS})
        hashes.update({builder.PACKAGE + name: builder.sha(context / name)
                       for name in builder.SOURCE_NAMES if name.endswith(".py")})
        hashes[builder.INHERITED_FC2] = builder.INHERITED_FC2_SHA
        if state["bad_hash"]:
            name, value = state["bad_hash"]
            hashes[name] = value
        if state["missing_hash"]:
            hashes.pop(state["missing_hash"])
        return "\n".join(f"{value}  {name}" for name, value in hashes.items())

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setattr(builder, "command", command)
    return output, state


def test_success_authenticates_sources_without_serving_or_gpu(attempt):
    output, state = attempt
    builder.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["parent_image_id"] == builder.PARENT_ID
    assert receipt["gpu_used"] is False
    assert receipt["speed_measurement_valid"] is False
    assert receipt["build_log_sha256"] == builder.sha(output / "build.log")
    assert set(receipt["source_sha256"]) == set(builder.SOURCE_NAMES)
    builds = [argv for argv, _ in state["calls"] if argv[:2] == ["docker", "build"]]
    assert len(builds) == 1
    assert "--network=none" in builds[0] and "--pull=false" in builds[0]
    runs = [argv for argv, _ in state["calls"] if argv[:2] == ["docker", "run"]]
    assert len(runs) == 1
    assert "--network=none" in runs[0] and "--runtime=runc" in runs[0]
    assert "NVIDIA_VISIBLE_DEVICES=void" in runs[0]
    assert runs[0][runs[0].index("--entrypoint") + 1] == "sha256sum"
    assert "--gpus" not in runs[0]
    assert set(builder.SAMPLERS).issubset(runs[0])
    assert set(builder.WARMUPS).issubset(runs[0])


@pytest.mark.parametrize("flag", ["existing", "parent_mismatch"])
def test_identity_failure_before_build_does_not_create_output(attempt, flag):
    output, state = attempt
    state[flag] = True
    with pytest.raises(ValueError):
        builder.main()
    assert not output.exists()
    assert not any(argv[:2] == ["docker", "build"] for argv, _ in state["calls"])


@pytest.mark.parametrize("flag", ["build_failure", "source_changed", "parent_changed"])
def test_failed_build_preserves_failure_and_never_verifies_runtime(attempt, flag):
    output, state = attempt
    state[flag] = True
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        builder.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error"]
    assert "image_id" not in receipt
    assert (output / "plan.json").is_file() and (output / "build.log").is_file()
    assert not any(argv[:2] == ["docker", "run"] for argv, _ in state["calls"])


@pytest.mark.parametrize("kind", ["different_sampler", "unpatched_sampler", "different_warmup", "unpatched_warmup", "helper", "fc2", "missing"])
def test_image_hash_failure_never_labels_complete(attempt, kind):
    output, state = attempt
    if kind == "unpatched_sampler":
        # Both original samplers fail even if they match each other.
        original = "0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711"
        previous = builder.command
        def originals(argv):
            value = previous(argv)
            return value.replace(builder.SAMPLER_PATCHED_SHA, original) if argv[:2] == ["docker", "run"] else value
        # Fixture restores command after this test.
        builder.command = originals
    elif kind == "unpatched_warmup":
        previous = builder.command
        def originals(argv):
            value = previous(argv)
            return value.replace(builder.WARMUP_PATCHED_SHA, builder.WARMUP_ORIGINAL_SHA) if argv[:2] == ["docker", "run"] else value
        builder.command = originals
    elif kind == "missing":
        state["missing_hash"] = builder.SAMPLERS[0]
    else:
        path = {"different_sampler": builder.SAMPLERS[1],
                "different_warmup": builder.WARMUPS[1],
                "helper": builder.PACKAGE + "v2_hook.py", "fc2": builder.INHERITED_FC2}[kind]
        state["bad_hash"] = (path, "f" * 64)
    with pytest.raises((ValueError, KeyError)):
        builder.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert "image_id" not in receipt
    assert (output / "image-source-sha256.txt").exists()


def test_existing_output_is_untouched(attempt):
    output, state = attempt
    output.mkdir()
    marker = output / "receipt.json"
    marker.write_text("preserve me")
    with pytest.raises(ValueError, match="fresh"):
        builder.main()
    assert marker.read_text() == "preserve me"
    assert state["calls"] == []


def test_production_tag_rejected(attempt, monkeypatch):
    output, state = attempt
    monkeypatch.setattr(sys, "argv", ["builder", "--output", str(output), "--tag", builder.PARENT_TAG])
    with pytest.raises(ValueError, match="separate"):
        builder.main()
    assert state["calls"] == []


def test_dockerfile_uses_exact_parent_and_pins_both_samplers():
    dockerfile = (REPO / "runtime_patch/p8_decode_capture/Dockerfile").read_text()
    assert [line for line in dockerfile.splitlines() if line.startswith("FROM ")] == [f"FROM {builder.PARENT_ID}"]
    for sampler in builder.SAMPLERS:
        assert f"sha256sum {sampler}" in dockerfile
    for warmup in builder.WARMUPS:
        assert f"sha256sum {warmup}" in dockerfile
    assert dockerfile.count(builder.WARMUP_ORIGINAL_SHA) == 2
    assert dockerfile.count("0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711") == 2
    assert "git apply --check" in dockerfile
    assert builder.INHERITED_FC2_SHA in dockerfile
