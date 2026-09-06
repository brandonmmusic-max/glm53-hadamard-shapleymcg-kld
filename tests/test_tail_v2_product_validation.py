from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from glm53_nvfp4 import tail_v2_product_validation as subject
from glm53_nvfp4 import tail_v2_product_runner as runner


def test_cf32_balance_rejects_domain_skew():
    windows = []
    domains = sorted(subject.DOMAINS)
    for index in range(32):
        wid = f"conditional-fit-{index:04d}"
        windows.append({"id": wid, "domain": domains[index // 8],
                        "prediction_positions": subject.ROWS,
                        "teacher_source_role": "conditional-fit",
                        "teacher_path": f"logits/full-panel/conditional-fit/{wid}.safetensors"})
    payload = {"schema": "glm53-codec.conditional-fit32-development-role.v1",
               "teacher_repo_id": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
               "teacher_revision": subject.protocol.FULL_PANEL_SOURCE_REVISION,
               "roles": {"fit": [], "conditional-fit": windows, "selection": [], "confirmation": [], "final": []}}
    assert len(subject.validate_cf32_payload(payload)) == 32
    payload["roles"]["conditional-fit"][0]["domain"] = domains[1]
    with pytest.raises(ValueError, match="eight windows per domain"):
        subject.validate_cf32_payload(payload)


def test_storage_inventory_enforces_one_window_and_limit(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "conditional-fit-0001.logits.f32").write_bytes(b"a" * 16)
    assert subject.storage_inventory(root)["raw_files"] == 1
    (root / "conditional-fit-0002.capture.inprogress.json").write_text("{}")
    with pytest.raises(ValueError, match="one-window"):
        subject.storage_inventory(root)
    (root / "conditional-fit-0002.capture.inprogress.json").unlink()
    monkeypatch.setattr(subject, "MAX_NEW_BYTES", 16)
    with pytest.raises(ValueError, match="one-window"):
        subject.storage_inventory(root)


@dataclass
class FakeScores:
    kld: np.ndarray


def test_score_receipt_is_durable_before_scoped_raw_retirement(tmp_path, monkeypatch):
    capture = (tmp_path / "capture").resolve()
    receipts = (tmp_path / "receipts").resolve()
    capture.mkdir()
    raw = capture / "conditional-fit-0001.logits.f32"
    raw.write_bytes(b"0123456789abcdef")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    metadata = capture / "conditional-fit-0001.capture.json"
    metadata.write_text(json.dumps({"raw_sha256": raw_sha}))
    tokens = np.arange(subject.ROWS + 1, dtype=np.int64)
    token_path = tmp_path / "tokens.npy"
    np.save(token_path, tokens)
    window = {"id": "conditional-fit-0001", "domain": sorted(subject.DOMAINS)[0],
              "token_path": str(token_path.resolve()), "input_sha256": subject.sha(token_path),
              "teacher_path": "unused", "teacher_sha256": "0" * 64}
    monkeypatch.setattr(subject, "RAW_BYTES", 16)
    monkeypatch.setattr(subject.protocol, "load_capture",
                        lambda root, wid, values: (np.zeros((1, 1), dtype="<f4"), {"raw_sha256": raw_sha}))
    result = subject.score_retire_window(
        arm="p8", repeat=1, window=window, teacher_root=tmp_path,
        capture_root=capture, receipt_root=receipts,
        runtime_audit_sha256="a" * 64,
        score_loader=lambda *_: FakeScores(np.linspace(0.0, 1.0, subject.ROWS)),
    )
    assert result["raw_retired"] is True
    assert not raw.exists()
    score = receipts / "repeat-01/p8/conditional-fit-0001.score.json"
    retirement = receipts / "repeat-01/p8/conditional-fit-0001.retirement.json"
    assert json.loads(retirement.read_text())["score_receipt_sha256"] == subject.sha(score)
    assert metadata.exists()


def _runtime_log(arm: str) -> str:
    dcp = subject.TOPOLOGY[arm]["dcp"]
    rows = [
        "Using V2 Model Runner",
        "attention_backend': 'B12X_MLA_SPARSE'",
        f"tensor_parallel_size=4 decode_context_parallel_size={dcp} speculative_config=None "
        "kv_cache_dtype=nvfp4_ds_mla dtype=torch.bfloat16 enforce_eager=False",
    ]
    if subject.TOPOLOGY[arm]["ep"]:
        rows.append("'enable_expert_parallel': True")
    if arm == "exl3":
        rows += ["quantization=exl3", "moe_backend='b12x'", "EXL3 full-expert EP runtime planned"]
    else:
        for layer in range(3, 45):
            for rank in range(4):
                rows.append(f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} stream=K4 "
                            "mma=mxf8f6f4 alphabet=E4M3 scale=UE8M0_K32 law=procedural_mcg "
                            "physical_bpw=4.25 small_m_scheduler=true")
    return "\n".join(rows)


@pytest.mark.parametrize("arm", subject.ARMS)
def test_runtime_audit_fails_closed_on_every_product_contract(arm, monkeypatch):
    monkeypatch.setattr(subject, "verify_graphs", lambda log: {"full": True})
    result = subject.audit_runtime_log(_runtime_log(arm), arm)
    assert result["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert result["payload_bpw"] == subject.PRODUCT[arm]["payload_bpw"]
    with pytest.raises(ValueError):
        subject.audit_runtime_log(_runtime_log(arm).replace("nvfp4_ds_mla", "fp8_ds_mla"), arm)


def test_determinism_failure_is_visible_not_averaged(tmp_path):
    windows = []
    domains = sorted(subject.DOMAINS)
    for index in range(32):
        wid = f"conditional-fit-{index:04d}"
        windows.append({"id": wid, "domain": domains[index // 8],
                        "prediction_positions": subject.ROWS,
                        "teacher_source_role": "conditional-fit",
                        "teacher_path": f"logits/full-panel/conditional-fit/{wid}.safetensors"})
        for repeat in range(1, 6):
            base = tmp_path / f"repeat-{repeat:02d}" / "p8"
            base.mkdir(parents=True, exist_ok=True)
            raw_sha = ("b" if index == 0 and repeat == 5 else "a") * 64
            score = base / f"{wid}.score.json"
            score.write_text(json.dumps({"raw_sha256": raw_sha}))
            (base / f"{wid}.retirement.json").write_text(json.dumps({
                "score_receipt_sha256": subject.sha(score), "retired_raw_sha256": raw_sha}))
        for repeat in range(1, 6):
            base = tmp_path / f"repeat-{repeat:02d}" / "exl3"
            base.mkdir(parents=True, exist_ok=True)
            score = base / f"{wid}.score.json"
            score.write_text(json.dumps({"raw_sha256": "c" * 64}))
            (base / f"{wid}.retirement.json").write_text(json.dumps({
                "score_receipt_sha256": subject.sha(score), "retired_raw_sha256": "c" * 64}))
    result = subject.verify_five_run_determinism(tmp_path.resolve(), windows)
    assert result["status"] == "fail"
    assert sum(not row["bitwise_deterministic"] for row in result["rows"]) == 1


def test_quality_port_probe_permits_time_wait_but_rejects_listener():
    from glm53_nvfp4 import tail_v2_product_runner as runner

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with pytest.raises(OSError):
            runner._check_port_available(port)
    runner._check_port_available(port)


def test_quality_lifecycle_scores_and_retires_before_next_request(tmp_path, monkeypatch):
    from types import SimpleNamespace

    tokens = np.arange(subject.ROWS + 1, dtype=np.int64)
    windows = []
    for index in range(2):
        path = tmp_path / f"tokens-{index}.npy"
        np.save(path, tokens)
        windows.append({"id": f"conditional-fit-{index:04d}",
                        "domain": sorted(subject.DOMAINS)[index],
                        "token_path": str(path), "input_sha256": subject.sha(path),
                        "teacher_path": "unused", "teacher_sha256": "0" * 64})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    plan = {"plan_path": str(plan_path), "images": {"p8": "sha256:" + "a" * 64},
            "windows": windows, "teacher_root": str(tmp_path)}
    entry = {"repeat": 1, "arm": "p8"}
    recipe = {"unused": True}
    events = []
    cid = "b" * 64

    monkeypatch.setattr(subject, "authenticate_plan", lambda path: plan)
    monkeypatch.setattr(subject, "storage_inventory", lambda path: {"raw_files": 0})
    monkeypatch.setattr(subject, "audit_runtime_log", lambda log, arm, require_dispatch=True: {"arm": arm})
    monkeypatch.setattr(runner, "_cool_and_inventory", lambda out, hardware: None)
    monkeypatch.setattr(runner, "clone_argv", lambda *args, **kwargs: ["docker", "create"])
    monkeypatch.setattr(runner.capture, "normalize_capture_ownership", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner.cold, "inventory", lambda: ["gpu"])

    class Reply:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b""

    monkeypatch.setattr(runner.json, "load", lambda stream: {
        "data": [{"id": runner.model_name(runner.QUALITY_PREFIX, "p8")}]
    } if isinstance(stream, Reply) else json.loads(stream.read()))
    monkeypatch.setattr(runner, "urlopen", lambda *args, **kwargs: Reply())

    def request(req, port, tick, timeout):
        wid = req["vllm_xargs"]["p8_decode_capture_window_id"]
        if events:
            assert events[-1].startswith("score:")
        events.append("request:" + wid)
        forced = req["vllm_xargs"]["p8_decode_capture_forced_token_ids"]
        return {"model": req["model"], "choices": [{"token_ids": forced, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": len(forced)}}

    monkeypatch.setattr(runner, "_request", request)

    def score(**kwargs):
        events.append("score:" + kwargs["window"]["id"])
        return {"raw_sha256": "c" * 64}

    monkeypatch.setattr(subject, "score_retire_window", score)

    completed = "\n".join(
        f"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window={w['id']} rows=2047 tp_rank=0"
        for w in windows)

    def command(argv, **kwargs):
        if argv[:3] == ["docker", "ps", "-a"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if argv[:2] == ["nvidia-smi", "--query-compute-apps=pid"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if argv[:3] == ["nvidia-smi", "-q", "-x"]:
            return SimpleNamespace(stdout="<xml/>", stderr="", returncode=0)
        if argv == ["docker", "create"]:
            return SimpleNamespace(stdout=cid, stderr="", returncode=0)
        if argv[:2] == ["docker", "start"]:
            return SimpleNamespace(stdout=cid, stderr="", returncode=0)
        if argv[:2] == ["docker", "inspect"]:
            if "-f" in argv:
                return SimpleNamespace(stdout=json.dumps({"Running": True}), stderr="", returncode=0)
            return SimpleNamespace(stdout="[]", stderr="", returncode=0)
        if argv[:2] == ["docker", "logs"]:
            return SimpleNamespace(stdout="ready", stderr="", returncode=0)
        raise AssertionError(argv)

    monkeypatch.setattr(runner.cold.pilot, "command", command)

    def cleanup(out, name, image, owner):
        (out / "server-final.private.log").write_text(completed)
        return True, []

    monkeypatch.setattr(runner.cold, "cleanup_owned", cleanup)
    quality = tmp_path / "quality"
    quality.mkdir()
    result = runner.run_quality_slot(plan, recipe, entry, quality,
                                     "d" * 64, ["gpu"])
    assert result["exit_code"] == 0
    assert events == ["request:conditional-fit-0000", "score:conditional-fit-0000",
                      "request:conditional-fit-0001", "score:conditional-fit-0001"]
