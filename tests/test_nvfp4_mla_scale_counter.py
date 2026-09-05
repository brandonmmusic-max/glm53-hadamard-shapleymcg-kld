from __future__ import annotations

import json
import weakref
from pathlib import Path

import pytest
import torch

from glm53_nvfp4.nvfp4_mla_counter_control import (
    receipt_path,
    wait_for_receipts,
    write_command,
)
from runtime_patch.nvfp4_mla_scale_counter import (
    NOPE_RECORD_BYTES,
    WRITER_AMAX_LIMIT,
    WriterScaleCounter,
    _RuntimeController,
)


RANK_LABELS = {
    "global_rank": 2,
    "local_rank": 2,
    "tp_rank": 2,
    "dcp_rank": 0,
    "tp_world_size": 4,
    "dcp_world_size": 1,
}


def _cache(record_bytes: int = NOPE_RECORD_BYTES) -> torch.Tensor:
    return torch.full((1, 64, record_bytes), 0xA5, dtype=torch.uint8)


def test_exact_writer_domain_counts_and_no_input_or_cache_mutation() -> None:
    counter = WriterScaleCounter()
    counter.reset("conditional-fit-0032")
    writer_input = torch.zeros((2, 512), dtype=torch.float32)
    writer_input[0, 0] = 3000.0
    writer_input[0, 16] = float("nan")
    writer_input[0, 32] = 0.003
    # This non-finite row has an invalid graph slot and must be excluded.
    writer_input[1].fill_(float("inf"))
    cache = _cache()
    slots = torch.tensor([7, -1], dtype=torch.int64)
    input_before = writer_input.clone()
    cache_before = cache.clone()

    counter.observe(
        layer_name="model.language_model.layers.3.self_attn.mla_attn",
        writer_input=writer_input,
        kv_cache=cache,
        slot_mapping=slots,
        outer_scale=2.0,
        rank_labels=RANK_LABELS,
    )
    result = counter.snapshot()["layers"][0]

    assert result["layer_index"] == 3
    assert result["global_rank"] == 2
    assert result["writer_calls"] == 1
    assert result["valid_tokens"] == 1
    assert result["valid_groups"] == 32
    assert result["valid_values"] == 512
    assert result["saturated_groups"] == 1
    assert result["saturated_tokens"] == 1
    assert result["all_zero_groups"] == 29
    assert result["all_zero_tokens"] == 0
    assert result["zero_values"] == 509
    assert result["scale_rounds_zero_groups"] == 1
    assert result["nonfinite_values"] == 1
    assert result["nonfinite_groups"] == 1
    assert result["nonfinite_tokens"] == 1
    assert result["max_writer_amax"] == pytest.approx(3000.0)
    assert result["max_writer_amax_ratio"] == pytest.approx(
        3000.0 / WRITER_AMAX_LIMIT
    )
    assert result["max_requested_e4m3_scale"] == pytest.approx(500.0)
    assert result["max_original_domain_amax"] == pytest.approx(6000.0)
    assert result["min_outer_scale"] == pytest.approx(2.0)
    assert result["max_outer_scale"] == pytest.approx(2.0)
    torch.testing.assert_close(writer_input, input_before, equal_nan=True)
    torch.testing.assert_close(cache, cache_before)


def test_counter_disarmed_and_abi_fail_closed() -> None:
    counter = WriterScaleCounter()
    values = torch.zeros((1, 512), dtype=torch.bfloat16)
    slots = torch.tensor([0], dtype=torch.int64)
    counter.observe(
        layer_name="layers.3.attn",
        writer_input=values,
        kv_cache=_cache(),
        slot_mapping=slots,
        outer_scale=1.0,
        rank_labels=RANK_LABELS,
    )
    warmup = counter.snapshot()["layers"][0]
    assert warmup["writer_calls"] == 0
    assert warmup["valid_tokens"] == 0

    counter.reset("window")
    with pytest.raises(RuntimeError, match="static 288-byte NoPE ABI"):
        counter.observe(
            layer_name="layers.3.attn",
            writer_input=values,
            kv_cache=_cache(304),
            slot_mapping=slots,
            outer_scale=1.0,
            rank_labels=RANK_LABELS,
        )


class _Owner:
    layer_name = "model.layers.20.self_attn.mla_attn"
    _nvfp4_mla_outer_scale = 4.0


class _Impl:
    dcp_rank = 0
    dcp_world_size = 1
    tp_world_size = 4

    def __init__(self) -> None:
        self._nvfp4_scale_counter_owner = weakref.ref(_Owner())


def test_atomic_control_request_boundaries_and_rank_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this CPU/static test from touching a visible GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    control = tmp_path / "control.json"
    output = tmp_path / "receipts"
    runtime = _RuntimeController(control, output)
    owner = _Owner()
    impl = _Impl()
    impl._nvfp4_scale_counter_owner = weakref.ref(owner)
    values = torch.zeros((1, 512), dtype=torch.float32)
    values[0, 0] = 3000
    cache = _cache()
    slots = torch.tensor([0], dtype=torch.int64)

    write_command(
        control,
        {
            "schema": "glm53.nvfp4-mla-scale-counter-control.v1",
            "sequence": 1,
            "action": "reset",
            "request_id": "conditional-fit-0032",
        },
    )
    # First observation models graph capture: it registers persistent tensors
    # while their device gate is zero. The outer runner then processes reset
    # before the first measured replay.
    runtime.observe(impl, torch.zeros_like(values), cache, slots)
    runtime.before_execute_model()
    runtime.observe(impl, values, cache, slots)
    reset_receipt = receipt_path(output, 1, 2)
    assert reset_receipt.is_file()
    assert json.loads(reset_receipt.read_text())["dump"] is None

    write_command(
        control,
        {
            "schema": "glm53.nvfp4-mla-scale-counter-control.v1",
            "sequence": 2,
            "action": "dump_reset",
            "dump_request_id": "conditional-fit-0032",
            "request_id": "conditional-fit-0021",
        },
    )
    runtime.before_execute_model()
    runtime.observe(impl, torch.zeros_like(values), cache, slots)
    dumped = json.loads(receipt_path(output, 2, 2).read_text())
    assert dumped["timing_admissible"] is False
    assert dumped["dump"]["request_id"] == "conditional-fit-0032"
    assert dumped["dump"]["layers"][0]["saturated_groups"] == 1
    assert runtime.counter.active_request_id == "conditional-fit-0021"
    assert runtime.counter.snapshot()["layers"][0]["saturated_groups"] == 0


def test_wait_for_receipts_requires_every_declared_rank(tmp_path: Path) -> None:
    for rank in (0, 1):
        path = receipt_path(tmp_path, 9, rank)
        path.write_text(json.dumps({"sequence": 9}), encoding="utf-8")
    paths = wait_for_receipts(tmp_path, 9, (0, 1), 0.1)
    assert paths == [receipt_path(tmp_path, 9, 0), receipt_path(tmp_path, 9, 1)]
    with pytest.raises(TimeoutError, match="rank-002"):
        wait_for_receipts(tmp_path, 9, (0, 1, 2), 0.01)


def test_sitecustomize_counter_is_opt_in_and_fail_closed() -> None:
    source = Path("runtime_patch/sitecustomize.py").read_text(encoding="utf-8")
    assert 'os.environ.get("VLLM_NVFP4_MLA_SCALE_COUNTER", "")' in source
    assert "_install_mla_counter()" in source
    assert "os._exit(78)" in source


def test_diagnostic_image_and_prereg_pin_non_mutating_tail_v2_path() -> None:
    dockerfile = Path(
        "runtime_patch/nvfp4_mla_scale_counter/Dockerfile"
    ).read_text(encoding="utf-8")
    assert (
        "FROM sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745"
        in dockerfile
    )
    assert 'org.klc.writer-mutation="none"' in dockerfile
    assert 'org.klc.timing-admissible="false"' in dockerfile
    assert "a070481091d425eddc311e25c0cb770e7ed79d1ee3d196157ad28d789fb5c60c" in dockerfile
    assert "0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572" in dockerfile
    assert "7958acb694a8d7600dd5392b8899a42f4e01ca76a937dc82015cf77ad96feef9" in dockerfile

    plan = json.loads(
        Path("experiments/nvfp4-mla-scale-counter-cf32-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["status"] == "prepared-not-launched"
    assert plan["panel"]["windows"] == 32
    assert plan["mandatory_integrity_gate"]["pass"].startswith("Shapes")
    assert "No instrumented timing" in plan["performance_boundary"]
