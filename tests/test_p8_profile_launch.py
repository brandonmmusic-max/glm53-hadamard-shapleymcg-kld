import copy
import json
import shlex
import sys

import pytest

from glm53_nvfp4 import p8_profile_launch as mod


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "speed-v2a3"
    root.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"candidate": {"image_id": mod.IMAGE,
                                            "topology": {"tp": 4, "ep": False, "dcp": 1}},
                                "benchmark": {"cold_process_runs_per_arm": 5}}))
    (root / "analysis.json").write_text(json.dumps({
        "schema": "glm53-p8-uniform-all42-vs-exl3-speed-analysis.v1", "plan": {"sha256": mod.sha(plan)}}))
    slot = root / "round-01-p8"
    slot.mkdir()
    cmd = ["exec", *mod.PREFIX]
    for key, value in mod.REQUIRED.items():
        cmd.extend([key, value])
    cmd.extend(sorted(mod.BOOLEAN_FLAGS))
    cmd.extend(["--gpu-memory-utilization", "0.94"])
    source = {"Image": mod.IMAGE, "Config": {"Entrypoint": ["/bin/bash"], "Cmd": ["-lc", shlex.join(cmd)],
              "Env": ["MOCK_SECRET=do-not-print", "CUDA_VISIBLE_DEVICES=0,1,2,3"], "User": "", "WorkingDir": "/"},
              "HostConfig": {"Binds": ["/fake/model:/model:ro", "/fake/cache:/cache:rw"],
                             "DeviceRequests": [copy.deepcopy(mod.DEVICE_REQUEST)],
                             "NetworkMode": "host", "IpcMode": "host", "ShmSize": 68719476736,
                             "Runtime": "runc", "Privileged": False, "CapAdd": None, "CapDrop": None,
                             "SecurityOpt": ["label=disable"], "PortBindings": {}, "ReadonlyRootfs": False,
                             "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}}}
    calls = []

    def rewrite():
        (slot / "container.json").write_text(json.dumps([source]))
        path = slot / "container.json"
        (slot / "receipt.json").write_text(json.dumps({"files": {"container.json": {
            "sha256": mod.sha(path), "bytes": path.stat().st_size, "path": str(path)}}}))

    rewrite()

    def audit(root_arg, plan_arg, allow_incomplete):
        calls.append((root_arg, plan_arg, allow_incomplete))
        return {"status": "pass", "missing": [], "runs": [{}] * 10}

    monkeypatch.setattr(mod, "audit", audit)
    monkeypatch.setattr(mod, "audit_slot", lambda *args: {
        "image_id": mod.IMAGE, "config_sha256": mod.config_fingerprint(source)})
    return root, plan, tmp_path / "diagnostic", source, rewrite, calls


def test_clone_changes_only_declared_serving_arguments(prepared):
    root, plan, out, source, _, calls = prepared
    argv = mod.build_argv(root, plan, out)
    assert calls == [(root, plan, False)]
    assert not out.exists()
    assert argv[:2] == ["docker", "run"]
    assert argv[argv.index("--entrypoint") + 1] == mod.NSYS
    assert argv[argv.index("--cidfile") + 1] == str(out / 'container.cid')
    assert [argv[i + 1] for i, value in enumerate(argv) if value == "--env"] == source["Config"]["Env"]
    assert [argv[i + 1] for i, value in enumerate(argv) if value == "--volume"] == [
        *source["HostConfig"]["Binds"], f"{out}:/profile:rw"]
    changed = argv[argv.index(mod.PREFIX[0]):]
    expected = shlex.split(source["Config"]["Cmd"][1])[1:]
    expected[expected.index("--port") + 1] = "8018"
    expected[expected.index("--served-model-name") + 1] = mod.MODEL
    assert changed[:-2] == expected
    assert json.loads(changed[-1]) == mod.PROFILER
    assert "--capture-range-end=stop" in argv and "--cuda-graph-trace=node" in argv
    assert "--privileged" not in argv and "--cap-add" not in argv


@pytest.mark.parametrize("tail", ["; touch /tmp/forbidden", " $(whoami)", " | cat", " `id`"])
def test_shell_syntax_rejected(prepared, tail):
    root, plan, out, source, rewrite, _ = prepared
    source["Config"]["Cmd"][1] += tail
    rewrite()
    with pytest.raises(ValueError, match="shell"):
        mod.build_argv(root, plan, out)


@pytest.mark.parametrize("cmd", [["-c", "exec x"], ["-lc", "echo hello"], ["-lc", "exec x", "extra"]])
def test_wrong_shell_structure(prepared, cmd):
    root, plan, out, source, rewrite, _ = prepared
    source["Config"]["Cmd"] = cmd
    rewrite()
    with pytest.raises(ValueError, match="shell"):
        mod.build_argv(root, plan, out)


@pytest.mark.parametrize("tail", [" --enable-expert-parallel", " --speculative-config '{}'",
                                  " --enable-prefix-caching", " --tensor-parallel-size 2",
                                  " --profiler-config '{}'", " --port=9999"])
def test_unsupported_or_override_options_rejected(prepared, tail):
    root, plan, out, source, rewrite, _ = prepared
    source["Config"]["Cmd"][1] += tail
    rewrite()
    with pytest.raises(ValueError, match="serving option"):
        mod.build_argv(root, plan, out)


def test_mismatched_required_option(prepared):
    root, plan, out, source, rewrite, _ = prepared
    source["Config"]["Cmd"][1] = source["Config"]["Cmd"][1].replace("2048", "4096")
    rewrite()
    with pytest.raises(ValueError, match="options differ"):
        mod.build_argv(root, plan, out)


def test_wrong_image(prepared):
    root, plan, out, source, rewrite, _ = prepared
    source["Image"] = "untrusted-image"
    rewrite()
    with pytest.raises(ValueError, match="image"):
        mod.build_argv(root, plan, out)


@pytest.mark.parametrize("change", ["existing", "symlink", "relative", "colon", "descendant"])
def test_bad_destination(prepared, change):
    root, plan, out, *_ = prepared
    if change == "existing":
        out.mkdir()
    elif change == "symlink":
        out.symlink_to(root, target_is_directory=True)
    elif change == "relative":
        out = type(out)("relative")
    elif change == "descendant":
        out = root / "new-diagnostic"
    else:
        out = out.with_name("bad:destination")
    with pytest.raises(ValueError, match="destination"):
        mod.build_argv(root, plan, out)


def test_profile_mount_collision(prepared):
    root, plan, out, source, rewrite, _ = prepared
    source["HostConfig"]["Binds"].append("/old:/profile:rw")
    rewrite()
    with pytest.raises(ValueError, match="mount destination"):
        mod.build_argv(root, plan, out)


def test_no_security_relaxation(prepared):
    root, plan, out, source, rewrite, _ = prepared
    source["HostConfig"]["SecurityOpt"].append("seccomp=unconfined")
    rewrite()
    with pytest.raises(ValueError, match="security"):
        mod.build_argv(root, plan, out)


def test_incomplete_series_rejected(prepared, monkeypatch):
    root, plan, out, *_ = prepared
    monkeypatch.setattr(mod, "audit", lambda *args, **kwargs: {"status": "incomplete", "missing": ["slot"]})
    with pytest.raises(ValueError, match="complete receipt audit"):
        mod.build_argv(root, plan, out)


def test_missing_analysis_rejected(prepared):
    root, plan, out, *_ = prepared
    (root / "analysis.json").unlink()
    with pytest.raises(ValueError, match="analysis is missing"):
        mod.build_argv(root, plan, out)


def test_source_hash_guard(prepared):
    root, plan, out, *_ = prepared
    with (root / "round-01-p8" / "container.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed after receipt"):
        mod.build_argv(root, plan, out)


def test_cli_writes_exclusive_private_spec_without_env_console(prepared, monkeypatch, capsys):
    root, plan, out, *_ = prepared
    monkeypatch.setattr(sys, "argv", ["p8_profile_launch", "--root", str(root), "--plan", str(plan),
                                     "--output-dir", str(out)])
    mod.main()
    output = capsys.readouterr().out
    assert "MOCK_SECRET" not in output and "do-not-print" not in output
    path = out / "launch-spec.json"
    assert path.stat().st_mode & 0o777 == 0o600
    spec = json.loads(path.read_text())
    assert spec["status"] == "prepared-not-launched" and not spec["trace_verified"]
    with pytest.raises(ValueError, match="destination"):
        mod.main()
