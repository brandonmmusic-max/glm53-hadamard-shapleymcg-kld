import copy
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from glm53_nvfp4 import p8_fc1_integration as integration


SOURCE = Path(__file__).resolve().parents[1] / "evidence/opened/codec-v2/p8-smallm-integrated-v1/container.json"
IMAGE = "sha256:" + "a" * 64
OWNER = "b" * 64


def test_clone_changes_only_candidate_selectors_and_endpoint(tmp_path):
    source = json.loads(SOURCE.read_text())[0]
    argv = integration.clone_argv(source, tmp_path, IMAGE, 64, True, OWNER)
    original = shlex.split(source["Config"]["Cmd"][1])
    actual = shlex.split(argv[-1])
    for option in ("--port", "--served-model-name"):
        actual[actual.index(option) + 1] = original[original.index(option) + 1]
    assert actual == original
    env = [argv[i + 1] for i, item in enumerate(argv) if item == "--env"]
    assert env[:-2] == source["Config"]["Env"]
    assert env[-2:] == ["GLM53_P8_FC1_TILE_N=64", "GLM53_P8_FUSED_SCRATCH=1"]
    assert [argv[i + 1] for i, item in enumerate(argv) if item == "--volume"] == source["HostConfig"]["Binds"]
    assert argv[-3] == IMAGE and argv[-2] == "-lc"
    assert argv[argv.index("--entrypoint") + 1] == "/bin/bash"
    assert argv[argv.index("--workdir") + 1] == source["Config"]["WorkingDir"] == "/"
    assert "nsys" not in argv
    assert source == json.loads(SOURCE.read_text())[0]


def test_clone_rejects_source_drift(tmp_path):
    source = json.loads(SOURCE.read_text())[0]
    for alter in (lambda x: x.update(Image="wrong"),
                  lambda x: x["Config"]["Cmd"].__setitem__(1, x["Config"]["Cmd"][1] + " --enforce-eager"),
                  lambda x: x["Config"].update(WorkingDir="/tmp"),
                  lambda x: x["Config"]["Env"].append(x["Config"]["Env"][0])):
        changed = copy.deepcopy(source)
        alter(changed)
        with pytest.raises(ValueError):
            integration.clone_argv(changed, tmp_path, IMAGE, 64, True, OWNER)


def test_private_evidence_is_exclusive_and_mode600(tmp_path):
    path = tmp_path / "private.json"
    integration.private_save(path, {"private": True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        integration.private_save(path, "replacement")


def runtime_log():
    return "\n".join([
        "enforce_eager=False", "Breakable CUDA graph enabled",
        "Capturing CUDA graphs (FULL): 100%",
        *[f"(Worker_TP{rank} pid=1) Graph capturing finished" for rank in range(4)],
        *[f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} small_m_scheduler=true"
          for layer in range(3, 45) for rank in range(4)],
        *[f"GLM53_P8_M1_DISPATCH layer={layer} rank={rank} fc1_tile_n=64 fused_scratch_zero=true"
          for layer in range(3, 45) for rank in range(4)],
    ])


def test_runtime_requires_exact_forward_m1_and_full_capture():
    log = runtime_log()
    assert integration.verify_runtime(log, 64, True)["m1_pairs"] == 168
    for invalid in (log.replace("100%", "0%"),
                    log.replace("fc1_tile_n=64", "fc1_tile_n=32", 1),
                    log.replace("fused_scratch_zero=true", "fused_scratch_zero=false", 1),
                    log + "\nGLM53_P8_M1_DISPATCH layer=3 rank=0 fc1_tile_n=64 fused_scratch_zero=true",
                    log.replace("layer=3 rank=0 small", "layer=4 rank=0 small", 1)):
        with pytest.raises(ValueError):
            integration.verify_runtime(invalid, 64, True)


def benchmark_result():
    result = json.loads((SOURCE.parent / "result.json").read_text())
    result["metadata"].update(model=integration.NAME, server=f"127.0.0.1:{integration.PORT}")
    return result


def test_exact_benchmark_args_and_diagnostic_only(tmp_path):
    argv = integration.benchmark_argv(tmp_path)
    assert argv[2:] == [
        "--host", "127.0.0.1", "--port", "8021", "--model", integration.NAME,
        "--concurrency", "1", "--contexts", "32k", "--duration", "20",
        "--max-tokens", "4096", "--decode-warmup-seconds", "3",
        "--standalone-prefill", "--prefill-contexts", "32k", "--prefill-duration", "20",
        "--prefill-metric", "auto", "--token-targeting", "estimate", "--display-mode", "plain",
        "--no-resume", "--output", str(tmp_path / "result.json")]
    result = benchmark_result()
    summary = integration.summarize(result)
    assert summary["single_run_diagnostic"]
    assert not summary["allocation_restart"] and not summary["five_cold_run_product_comparison"]
    result["results"][0]["aggregate_tps"] = float("nan")
    with pytest.raises(ValueError):
        integration.summarize(result)


@pytest.mark.parametrize("group,key,value", [
    ("metadata", "duration_per_test", 1), ("metadata", "concurrency_levels", [2]),
    ("metadata", "context_lengths", [8192]), ("metadata", "max_tokens", 512),
    ("metadata", "decode_warmup_seconds", 0), ("metadata", "prefill_mode", "scout"),
    ("metadata", "primary_decode_layer", "burst"), ("metadata", "model", "wrong"),
    ("metadata", "server", "127.0.0.1:8019"),
    ("row", "aggregate_source", "client_chunks"), ("row", "benchmark_mode", "requests"),
    ("row", "num_errors", 1), ("row", "underfilled", True),
    ("row", "effective_concurrency", 0), ("row", "warmup_timed_out", True),
    ("row", "capacity_limited", True), ("row", "measurement_seconds", .2),
    ("row", "input_seq_len_avg", 8192),
    ("prefill", "method", "prometheus"), ("prefill", "samples", 0),
    ("prefill", "samples", 1.5), ("prefill", "prompt_tokens", 8192),
    ("hardware", "temp_max_c", 90), ("hardware", "gpu_count", 3),
])
def test_protocol_integrity_rejects_incomparable_results(group, key, value):
    result = benchmark_result()
    target = {"metadata": result["metadata"], "row": result["results"][0],
              "prefill": result["prefill"]["32768"], "hardware": result["hardware_run_summary"]}[group]
    target[key] = value
    with pytest.raises(ValueError):
        integration.summarize(result)


def test_services_restore_independently_after_backend_error():
    calls = []

    def command(argv):
        calls.append(argv)
        if "klc-backend.service" in argv:
            raise RuntimeError("backend start failed")

    states = integration.restore_services({"backend": True, "timer": True}, command,
                                          lambda unit, user: unit.endswith("timer"))
    assert len(calls) == 2 and "klc-model-stack.timer" in calls[1]
    assert states["backend"] is False and states["timer"] is True
    assert len(states["errors"]) == 1


def test_cleanup_kills_after_failed_stop_and_preserves_privacy(tmp_path):
    container = {"Id": "c" * 64, "Image": IMAGE, "Name": "/" + integration.NAME,
                 "Config": {"Labels": {integration.OWNER_LABEL: OWNER}}, "State": {"Running": True}}
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "stop":
            raise subprocess.TimeoutExpired(argv, 45)
        if argv[1] == "kill":
            container["State"]["Running"] = False
        output = json.dumps([container]) if argv[1] == "inspect" else ""
        return subprocess.CompletedProcess(argv, 0, output, "")

    ok, errors = integration.cleanup_owned(tmp_path, IMAGE, OWNER, command)
    assert not ok and errors  # Incomplete stop receipt fails closed, despite cleanup.
    assert any(argv[1] == "kill" for argv in calls) and calls[-1][1] == "rm"
    assert (tmp_path / "container-final.private.json").stat().st_mode & 0o777 == 0o600


def test_cleanup_removes_stopped_container_even_if_evidence_write_fails(tmp_path, monkeypatch):
    container = {"Id": "c" * 64, "Image": IMAGE, "Name": "/" + integration.NAME,
                 "Config": {"Labels": {integration.OWNER_LABEL: OWNER}}, "State": {"Running": False}}
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps([container]) if argv[1] == "inspect" else "", "")

    def fail(*args):
        raise OSError("write failure")

    monkeypatch.setattr(integration, "private_save", fail)
    ok, errors = integration.cleanup_owned(tmp_path, IMAGE, OWNER, command)
    assert not ok and len(errors) == 2 and calls[-1][1] == "rm"
    calls.clear()
    container["Config"]["Labels"][integration.OWNER_LABEL] = "foreign"
    ok, errors = integration.cleanup_owned(tmp_path, IMAGE, OWNER, command)
    assert not ok and len(calls) == 1


@pytest.fixture
def sealed_plan(tmp_path, monkeypatch):
    repo, source, device = [tmp_path / name for name in ("repo", "source", "device")]
    for path in (repo, source, device):
        path.mkdir()
    monkeypatch.setattr(integration, "REPO", repo)
    monkeypatch.setattr(integration, "SOURCE", source)
    benchmark = tmp_path / "benchmark.py"
    benchmark.write_text("# frozen benchmark\n")
    monkeypatch.setattr(integration, "BENCH", benchmark)
    for relative in integration.REQUIRED_SOURCES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
    (source / "container.json").write_text("[]")
    source_hash = integration.sha(source / "container.json")
    monkeypatch.setattr(integration, "SOURCE_SHA", source_hash)
    (source / "execution.json").write_text(json.dumps({"exit_code": 0, "files": {"container.json": {"sha256": source_hash}}}))
    (source / "analysis.json").write_text(json.dumps({"integration_status": "pass"}))
    hashes = {key: integration.sha(repo / key) for key in integration.REQUIRED_SOURCES}
    device_plan = tmp_path / "device-plan.json"
    device_plan.write_text(json.dumps({"physical_gpu_order": [0, 1, 2, 3], "minimum_reduction": .35,
                                      "candidate_image": IMAGE, "fused_scratch_candidates": True,
                                      "source_sha256": hashes}))
    analysis = {"decision": "advance_to_integrated_tp4", "eligible_tiles": ["64"],
                "devices": [{"physical_gpu": gpu} for gpu in range(4)]}
    monkeypatch.setattr(integration, "analyze_device", lambda root, plan: analysis)
    (device / "analysis.json").write_text(json.dumps(analysis))
    (device / "execution.json").write_text("{}")
    results = {}
    for gpu in range(4):
        path = device / f"gpu{gpu}" / "result.json"
        path.parent.mkdir()
        path.write_text("{}")
        results[str(gpu)] = integration.sha(path)
    plan = {"schema": "glm53-p8-fc1-integration-plan.v1", "runs": 1, "port": 8021,
            "thermal_start_max_c": 75, "thermal_abort_c": 90, "source_container_sha256": source_hash,
            "five_cold_run_product_comparison": False, "allocation_restart": False, "tile_n": 64,
            "fused_scratch_zero": True, "candidate_image": IMAGE, "source_sha256": hashes,
            "source_execution_sha256": integration.sha(source / "execution.json"),
            "source_analysis_sha256": integration.sha(source / "analysis.json"),
            "benchmark_sha256": integration.sha(benchmark), "device_root": str(device),
            "device_plan": str(device_plan), "device_plan_sha256": integration.sha(device_plan),
            "device_execution_sha256": integration.sha(device / "execution.json"),
            "device_analysis_sha256": integration.sha(device / "analysis.json"), "device_result_sha256": results}
    path = tmp_path / "plan.json"

    def seal():
        path.write_text(json.dumps(plan))
        path.with_suffix(".sha256").write_text(integration.sha(path))

    seal()
    return path, plan, seal


def test_prerequisite_replay_and_unqualified_tile_fail_closed(sealed_plan):
    path, plan, seal = sealed_plan
    assert integration.authenticate(path)[0]["tile_n"] == 64
    plan["tile_n"] = 32
    seal()
    with pytest.raises(ValueError, match="all-four-device gate"):
        integration.authenticate(path)
    plan["tile_n"] = 64
    plan["device_result_sha256"].pop("3")
    seal()
    with pytest.raises(ValueError, match="four device results"):
        integration.authenticate(path)


def test_bad_seal_and_runtime_drift_fail_before_execution(sealed_plan):
    path, plan, seal = sealed_plan
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="seal"):
        integration.authenticate(path)
    seal()
    plan["source_sha256"]["runtime_patch/p8_native_kernel.py"] = "0" * 64
    seal()
    with pytest.raises(ValueError, match="source identity"):
        integration.authenticate(path)
