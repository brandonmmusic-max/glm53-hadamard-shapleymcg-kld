import hashlib
import json
from pathlib import Path
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


def _tiny_window(tmp_path):
    tokens = tmp_path / "tokens.npy"; np.save(tokens, np.array([1, 2], dtype=np.int64))
    teacher = tmp_path / "teacher.npz"; np.savez(teacher, logits=np.array([1.0]))
    return {"id": "conditional-fit-0001", "domain": "code", "token_path": str(tokens),
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
