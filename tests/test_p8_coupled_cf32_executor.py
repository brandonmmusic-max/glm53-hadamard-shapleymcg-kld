import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from glm53_nvfp4 import p8_coupled_cf32_executor as executor
from glm53_nvfp4 import p8_coupled_three_layer_runtime as runtime


def _common_log():
    rows = [
        "Using V2 Model Runner",
        "tensor_parallel_size=4",
        "decode_context_parallel_size=1",
        "speculative_config=None",
        "kv_cache_dtype=nvfp4_ds_mla",
        "quantization=modelopt_mixed",
    ]
    rows += [
        f"GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank={rank} max_num_reqs=1 real_vocab=154880 expected_outputs=2047"
        for rank in range(4)
    ]
    rows += [
        f"GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank={rank} registrations=1 samples=2"
        for rank in range(4)
    ]
    return rows


def _p8_log(arm):
    boundary = "identity" if arm == "identity_p8" else "coupled-h512-h128-suh-svh-v1"
    full = "false" if arm == "identity_p8" else "true"
    rows = _common_log()
    for layer in runtime.LAYERS:
        for rank in runtime.RANKS:
            rows += [
                f"GLM53_P8_NATIVE_WEIGHTS_READY layer={layer} rank={rank} x boundary={boundary} full_coupled={full}",
                f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} x boundary={boundary}",
            ]
    return "\n".join(rows)


def test_runtime_audit_requires_exact_selected_layer_rank_pairs(monkeypatch):
    monkeypatch.setattr(executor, "verify_graphs", lambda _text: {"status": "pass"})
    text = _p8_log("coupled_p8")
    assert executor.audit_runtime_log(text, "coupled_p8")["conditions"]["bpw"] > 4.25
    broken = text.replace("GLM53_P8_NATIVE_FORWARD layer=20 rank=2", "REMOVED", 1)
    with pytest.raises(ValueError, match="missing rank/layer"):
        executor.audit_runtime_log(broken, "coupled_p8")


def test_runtime_audit_records_stock_backend_from_log(monkeypatch):
    monkeypatch.setattr(executor, "verify_graphs", lambda _text: {"status": "pass"})
    text = "\n".join(_common_log() + [
        "Using 'HUMMING' NvFp4 MoE backend",
        "Detected ModelOpt NVFP4 checkpoint",
    ])
    result = executor.audit_runtime_log(text, "stock")
    assert result["conditions"]["moe_backend"] == "HUMMING"
    assert result["conditions"]["activation_precision"] == "runtime-emitted ModelOpt markers: NVFP4"


def test_runtime_audit_requires_ordered_2047_row_completions(monkeypatch):
    monkeypatch.setattr(executor, "verify_graphs", lambda _text: {"status": "pass"})
    windows = ["conditional-fit-0001", "conditional-fit-0002"]
    text = _p8_log("identity_p8") + "\n" + "\n".join(
        f"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window={wid} rows=2047 tp_rank=0"
        for wid in windows)
    assert executor.audit_runtime_log(text, "identity_p8", completed=windows)["capture_rows"] == 2047
    with pytest.raises(ValueError, match="ordered"):
        executor.audit_runtime_log(text, "identity_p8", completed=list(reversed(windows)))


def test_owned_argv_adds_exact_cidfile_and_owner(tmp_path):
    template = ["docker", "create", "--name", "candidate", "--network", "host", "image"]
    argv = executor._owned_argv(template, tmp_path, "sealed-owner")
    assert argv[:2] == ["docker", "create"]
    assert argv.count("--cidfile") == argv.count("--label") == 1
    assert str(tmp_path / "container.cid") in argv
    assert f"{executor.cold.LABEL}=sealed-owner" in argv


def test_global_inventory_excludes_current_campaign_and_rejects_alias(tmp_path):
    old = tmp_path / "old"; old.mkdir()
    current = tmp_path / "current"; current.mkdir()
    (old / "conditional-fit-0001.logits.f32").write_bytes(b"old")
    (current / "conditional-fit-0002.logits.f32").write_bytes(b"new")
    value = executor.global_capture_inventory(tmp_path, current)
    assert value["bytes"] == 3 and len(value["files"]) == 1
    (old / "alias.logits.f32").symlink_to(old / "conditional-fit-0001.logits.f32")
    with pytest.raises(ValueError, match="aliases"):
        executor.global_capture_inventory(tmp_path, current)


def test_outside_coupled_inventory_is_privileged_read_only_and_fixed_exclusion(monkeypatch):
    seen = []
    def run(argv, **kwargs):
        seen.append((argv, kwargs)); return SimpleNamespace(stdout="7\n11\n")
    monkeypatch.setattr(executor.subprocess, "run", run)
    assert executor._coupled_outside_bytes() == 18
    argv, kwargs = seen[0]
    assert argv[:6] == ["sudo", "-n", "find", str(executor.WORKSPACE), "-xdev", "("]
    assert ["-ipath", "*coupled*"] == argv[argv.index("-ipath"):argv.index("-ipath") + 2]
    assert all(str(path) in argv for path in (*executor.WORKTREES, executor.WORKSPACE / "bmxfp4-glm53/.git"))
    assert not any(token in argv for token in ("-delete", "-exec", "-execdir"))
    assert kwargs == {"check": True, "text": True, "capture_output": True, "timeout": 180}


def _ledger(tmp_path, output):
    components, fixed = [], {}
    for name, mode in executor.LEDGER_COMPONENTS.items():
        path = output if name == "capture_output" else tmp_path / name
        if name != "capture_output":
            path.mkdir()
            (path / "evidence.bin").write_bytes(b"x")
            fixed[name] = (path,)
        cutoff = executor.LEDGER_CUTOFF * 1_000_000_000 if mode == "files-modified-since-cutoff" else None
        baseline = executor._apparent_bytes(path, cutoff_ns=cutoff, missing_ok=name == "capture_output")
        projected = (executor.QUALITY_PEAK if name == "quality_root" else
                     65536 if name == "capture_output" else baseline)
        components.append({"name": name, "mode": mode, "paths": [str(path)],
                           "baseline_bytes": baseline, "projected_bytes": projected})
    baseline_sum = sum(row["projected_bytes"] for row in components)
    value = {"schema": executor.LEDGER_SCHEMA, "status": "pass",
        "cutoff_unix": executor.LEDGER_CUTOFF, "ceiling_bytes": executor.GLOBAL_BUDGET,
        "retained_prior_quality_peak_charge_bytes": executor.QUALITY_PEAK,
        "capture_peak_already_charged": True, "pinned_raw_peak_bytes": executor.RAW_BYTES,
        "future_jit_log_allowance_in_baseline_projection": True,
        "future_jit_log_allowance_bytes": 65536, "unallocated_reserve_bytes": 0,
        "measurement_method": "fixed source-owned paths; apparent lstat trees; original-cutoff file sums; privileged read-only Docker find",
        "measured_unix": int(time.time()), "valid_until_unix": int(time.time()) + 3600,
        "baseline_projected_aggregate_bytes": baseline_sum,
        "components": components}
    path = tmp_path / "ledger.json"; path.write_text(json.dumps(value))
    path.with_suffix(".sha256").write_text(executor.runtime.sha(path) + "  ledger.json\n")
    return path, value, fixed


def test_external_ledger_counts_monotonic_growth_and_credits_only_active_raw(tmp_path, monkeypatch):
    output = tmp_path / "capture_output"
    ledger_path, ledger, fixed = _ledger(tmp_path, output)
    monkeypatch.setattr(executor, "FIXED_LEDGER_PATHS", fixed)
    monkeypatch.setattr(executor, "_privileged_modified_bytes",
                        lambda paths: sum(executor._apparent_bytes(path) for path in paths))
    outside = next(row for row in ledger["components"] if row["name"] == "coupled_outside_since_cutoff")
    monkeypatch.setattr(executor, "_coupled_outside_bytes", lambda: outside["baseline_bytes"])
    value, baseline = executor.validate_storage_ledger(ledger_path, output)
    assert value == ledger and baseline["positive_growth_bytes"] == 0
    output.mkdir(); (output / "receipt.json").write_bytes(b"receipt")
    monkeypatch.setattr(executor.time, "time", lambda: ledger["valid_until_unix"] + 600)
    _value, grown = executor.validate_storage_ledger(ledger_path, output, require_fresh=False)
    capture_row = next(row for row in grown["components"] if row["name"] == "capture_output")
    assert capture_row["current_bytes"] >= len(b"receipt")
    assert capture_row["positive_growth_bytes"] == 0
    raw = output / "conditional-fit-0001.logits.f32"
    with raw.open("wb") as stream:
        stream.truncate(executor.RAW_BYTES)
    global_root = tmp_path / "global"; global_root.mkdir()
    seal = {"global_capture_root": str(global_root),
            "global_retained_capture_inventory": executor.global_capture_inventory(global_root, output),
            "storage_ledger": {"path": str(ledger_path)}}
    result = executor._capture_budget(seal, output, reserve_next_raw=False)
    assert result["capture_peak_credit_bytes"] > 0
    assert result["capture_peak_credit_bytes"] <= executor.RAW_BYTES
    assert result["aggregate_charged_bytes"] <= executor.GLOBAL_BUDGET
    assert result["prospective_next_raw_aggregate_bytes"] <= executor.GLOBAL_BUDGET


def test_ledger_rejects_next_raw_when_quality_peak_is_already_live(tmp_path, monkeypatch):
    output = tmp_path / "capture_output"
    ledger_path, ledger, fixed = _ledger(tmp_path, output)
    ledger["baseline_projected_aggregate_bytes"] = executor.GLOBAL_BUDGET
    ledger_path.write_text(json.dumps(ledger))
    ledger_path.with_suffix(".sha256").write_text(executor.runtime.sha(ledger_path) + "  ledger.json\n")
    monkeypatch.setattr(executor, "FIXED_LEDGER_PATHS", fixed)
    monkeypatch.setattr(executor, "_privileged_modified_bytes",
                        lambda paths: sum(executor._apparent_bytes(path) for path in paths))
    outside = next(row for row in ledger["components"] if row["name"] == "coupled_outside_since_cutoff")
    monkeypatch.setattr(executor, "_coupled_outside_bytes", lambda: outside["baseline_bytes"])
    quality = fixed["quality_root"][0] / "live-quality.raw"
    with quality.open("wb") as stream:
        stream.truncate(executor.QUALITY_PEAK - 4096)
    global_root = tmp_path / "global"; global_root.mkdir()
    seal = {"global_capture_root": str(global_root),
            "global_retained_capture_inventory": executor.global_capture_inventory(global_root, output),
            "storage_ledger": {"path": str(ledger_path)}}
    with pytest.raises(ValueError, match="aggregate campaign ceiling"):
        executor._capture_budget(seal, output)


def _tiny_window(tmp_path):
    tokens = tmp_path / "tokens.npy"; np.save(tokens, np.array([1, 2], dtype=np.int64))
    teacher = tmp_path / "teacher.npz"; np.savez(teacher, logits=np.array([1.0]))
    return {"id": "conditional-fit-0001", "domain": "code", "token_path": str(tokens),
            "input_sha256": executor.runtime.sha(tokens),
            "token_values_sha256": executor.protocol.tokens_sha(np.load(tokens, allow_pickle=False)),
            "teacher_path": teacher.name, "teacher_sha256": executor.validation.sha(teacher)}


def _fake_inventory(root):
    raws = list(Path(root).glob("*.logits.f32"))
    ids = [path.name.split(".logits", 1)[0] for path in raws]
    return {"active_window_ids": ids, "raw_files": len(raws)}


def test_score_receipt_precedes_exact_raw_retirement(tmp_path, monkeypatch):
    window = _tiny_window(tmp_path)
    capture = tmp_path / "captures"; capture.mkdir()
    raw = capture / "conditional-fit-0001.logits.f32"; raw.write_bytes(b"data")
    metadata = capture / "conditional-fit-0001.capture.json"
    metadata.write_text(json.dumps({"raw_sha256": executor.validation.sha(raw)}))
    monkeypatch.setattr(executor, "RAW_BYTES", 4)
    monkeypatch.setattr(executor.validation, "storage_inventory", _fake_inventory)
    monkeypatch.setattr(executor.protocol, "load_capture", lambda *_args: (
        np.zeros((2047, 1)), {"raw_sha256": executor.validation.sha(raw)}))
    monkeypatch.setattr(executor.validation.metric, "_load_teacher", lambda *_args: np.zeros((2047, 1)))
    score = SimpleNamespace(kld=np.linspace(0.0, 1.0, 2047))
    monkeypatch.setattr(executor.protocol, "score_aligned", lambda *_args: score)
    monkeypatch.setattr(executor.validation, "_save_scores", lambda path, _scores: path.write_bytes(b"scores"))
    result = executor.score_retire_window(
        arm="stock", window=window, teacher_root=tmp_path, capture_root=capture,
        receipt_root=tmp_path / "receipts", runtime_audit_sha256="a" * 64)
    assert result["prediction_rows"] == 2047 and result["true_decode_rows"] == 2046
    assert not raw.exists()
    assert (tmp_path / "receipts/stock/conditional-fit-0001.score.json").is_file()
    assert (tmp_path / "receipts/stock/conditional-fit-0001.retirement.json").is_file()


def test_raw_survives_failure_to_write_durable_score(tmp_path, monkeypatch):
    window = _tiny_window(tmp_path)
    capture = tmp_path / "captures"; capture.mkdir()
    raw = capture / "conditional-fit-0001.logits.f32"; raw.write_bytes(b"data")
    (capture / "conditional-fit-0001.capture.json").write_text("{}")
    monkeypatch.setattr(executor, "RAW_BYTES", 4)
    monkeypatch.setattr(executor.validation, "storage_inventory", _fake_inventory)
    monkeypatch.setattr(executor.protocol, "load_capture", lambda *_args: (
        np.zeros((2047, 1)), {"raw_sha256": executor.validation.sha(raw)}))
    monkeypatch.setattr(executor.validation.metric, "_load_teacher", lambda *_args: np.zeros((2047, 1)))
    monkeypatch.setattr(executor.protocol, "score_aligned", lambda *_args: SimpleNamespace(kld=np.ones(2047)))
    monkeypatch.setattr(executor.validation, "_save_scores", lambda path, _scores: path.write_bytes(b"scores"))
    monkeypatch.setattr(executor.validation, "_durable_json", lambda *_args: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        executor.score_retire_window(
            arm="stock", window=window, teacher_root=tmp_path, capture_root=capture,
            receipt_root=tmp_path / "receipts", runtime_audit_sha256="a" * 64)
    assert raw.is_file()


def test_execute_never_restores_and_stops_before_next_arm(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    manifest = {"arms": {arm: {"capture_root": str(output / arm)} for arm in executor.ARMS}}
    seal = {"output": str(output)}
    windows = [{"id": f"conditional-fit-{i:04d}"} for i in range(32)]
    seal_path = tmp_path / "seal.json"; seal_path.write_text("seal")
    monkeypatch.setattr(executor, "LOCK", tmp_path / "root.lock")
    monkeypatch.setattr(executor, "authenticate_execution_seal", lambda _path: (seal, manifest, windows))
    observations = []
    import scripts.prepare_p8_coupled_three_layer_cf32_runtime as preparer
    monkeypatch.setattr(preparer, "production_off", lambda: observations.append("off") or {"off": True})
    monkeypatch.setattr(executor.cold.pilot, "command", lambda *_args, **_kwargs: SimpleNamespace(stdout=runtime.V9_IMAGE))
    monkeypatch.setattr(executor.cold, "inventory", lambda: ["gpu"] * 4)
    monkeypatch.setattr(executor.cold, "restore_if_safe", lambda *_args, **_kwargs: pytest.fail("restoration called"), raising=False)
    seen = []
    def arm_runner(_seal, _manifest, _windows, arm, _owner):
        seen.append(arm)
        if arm == "identity_p8":
            raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="production remains off"):
        executor.execute(seal_path, arm_runner=arm_runner)
    assert seen == ["stock", "identity_p8"]
    assert observations == ["off", "off"]
    record = json.loads((output / "execution.json").read_text())
    assert record["completed_arms"] == ["stock"]
    assert record["restoration_attempted"] is False


def _analysis_fixture(tmp_path):
    windows = [{"id": "conditional-fit-0001", "domain": "code",
                "input_sha256": "b" * 64, "token_values_sha256": "c" * 64}]
    output = tmp_path / "campaign"
    manifest = {"arms": {arm: {"capture_root": str(output / arm)} for arm in executor.ARMS}}
    for arm in executor.ARMS:
        root = output / arm; score_root = root / "scores" / arm; score_root.mkdir(parents=True)
        (root / "captures").mkdir()
        audit = {"arm": arm, "conditions": {"attention": "a", "kv_dtype": "k", "moe_backend": arm,
                                              "activation_precision": "p", "bpw": 4.25}}
        final = {**audit, "completed_window_ids": ["conditional-fit-0001"],
                 "true_decode_mask": {"exclude_rows": [0], "include": [1, 2047]}}
        (root / "runtime-audit.json").write_text(json.dumps(audit))
        (root / "runtime-final-audit.json").write_text(json.dumps(final))
        wid = "conditional-fit-0001"
        npz = score_root / f"{wid}.scores.npz"; np.savez(npz, kld=np.ones(2047))
        metadata = root / "captures" / f"{wid}.capture.json"; metadata.write_text("{}")
        score = {"schema": "glm53.p8-coupled-cf32-window-score.v1", "status": "complete",
            "arm": arm, "window_id": wid, "domain": "code", "prediction_rows": 2047,
            "one_token_prefill_rows": 1, "true_decode_rows": 2046, "raw_retired": False,
            "input_sha256": "b" * 64, "token_values_sha256": "c" * 64,
            "runtime_audit_sha256": executor.runtime.sha(root / "runtime-audit.json"),
            "scores_sha256": executor.runtime.sha(npz),
            "capture_metadata_sha256": executor.runtime.sha(metadata),
            "raw_sha256": "a" * 64, "mean_kld": 1.0, "one_token_prefill_kld": 1.0,
            "true_decode_mean_kld": 1.0}
        score_path = score_root / f"{wid}.score.json"; score_path.write_text(json.dumps(score))
        retirement = {"schema": "glm53.p8-coupled-cf32-window-retirement.v1", "status": "complete",
            "score_receipt_sha256": executor.runtime.sha(score_path), "retired_raw_sha256": "a" * 64,
            "retired_raw_bytes": executor.RAW_BYTES, "capture_metadata_retained": str(metadata)}
        (score_root / f"{wid}.retirement.json").write_text(json.dumps(retirement))
        execution = {"schema": "glm53.p8-coupled-cf32-arm-execution.v1", "arm": arm, "exit_code": 0,
            "cleanup": {"ok": True, "errors": []}, "restoration_attempted": False,
            "completed_windows": 1, "windows": [{"window_id": wid,
                "score_sha256": executor.runtime.sha(score_path), "raw_sha256": "a" * 64,
                "raw_retired": True, "input_sha256": "b" * 64,
                "token_values_sha256": "c" * 64}]}
        (root / "execution.json").write_text(json.dumps(execution))
    return manifest, windows


def test_analysis_requires_execution_score_retirement_and_replayed_means(tmp_path, monkeypatch):
    manifest, windows = _analysis_fixture(tmp_path)
    monkeypatch.setattr(executor.analysis, "analyze", lambda _windows, arms: arms)
    assert set(executor._analyze(manifest, windows)) == set(executor.ARMS)
    target = tmp_path / "campaign/identity_p8/scores/identity_p8/conditional-fit-0001.retirement.json"
    value = json.loads(target.read_text()); value["retired_raw_sha256"] = "b" * 64
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="score/retirement/hash"):
        executor._analyze(manifest, windows)
