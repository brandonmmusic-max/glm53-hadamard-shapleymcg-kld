import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from runtime_patch import p8_index_order_receipt as receipt


class FakeLoader:
    calls = 0

    def __init__(self, name=receipt.MODULE):
        self.name = name

    def exec_module(self, module):
        type(self).calls += 1
        module.run_fused_paged_indexer = lambda *args, **kwargs: ("i", "v")
        FakeOrder.TRANSFORMATIONS[self.name] = exact_source_receipt()
        return "delegated"


class FakeOrder:
    _Loader = FakeLoader
    TRANSFORMATIONS = {}


def exact_source_receipt():
    return {
        "module": receipt.MODULE,
        "mode": receipt.MODE,
        "original_sha256": receipt.ORIGINAL_SHA256,
        "emitted_sha256": receipt.EMITTED_SHA256,
        "inspect_source_sha256": receipt.EMITTED_SHA256,
        "cache_suffix": receipt.CACHE_SUFFIX,
    }


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    original = FakeLoader.exec_module
    FakeLoader.calls = 0
    FakeOrder.TRANSFORMATIONS = {}
    monkeypatch.setattr(receipt, "_INSTALLED", False)
    monkeypatch.setenv("GLM53_P8_INDEX_ORDER", receipt.MODE)
    monkeypatch.setenv("GLM53_P8_INDEX_ORDER_RECEIPT", "1")
    monkeypatch.delitem(sys.modules, receipt.MODULE, raising=False)
    yield
    FakeLoader.exec_module = original
    FakeOrder.TRANSFORMATIONS = {}


def fake_rank_modules(monkeypatch, *, initialized=True):
    distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: initialized,
        get_rank=lambda: 7,
    )
    torch = SimpleNamespace(distributed=distributed)
    parallel = SimpleNamespace(
        model_parallel_is_initialized=lambda: initialized,
        get_tensor_model_parallel_rank=lambda: 3,
        get_tensor_model_parallel_world_size=lambda: 4,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)


def load_target():
    module = ModuleType(receipt.MODULE)
    loader = FakeOrder._Loader()
    assert loader.exec_module(module) == "delegated"
    return module


def markers(capsys):
    lines = capsys.readouterr().out.splitlines()
    return [
        json.loads(line.removeprefix(receipt.PREFIX))
        for line in lines
        if line.startswith(receipt.PREFIX)
    ]


def test_loader_delegates_once_then_wraps_exact_entrypoint(monkeypatch, capsys):
    fake_rank_modules(monkeypatch)
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "7")
    receipt.install(FakeOrder)
    module = load_target()
    assert FakeLoader.calls == 1
    assert module.run_fused_paged_indexer(num_heads=32, topk=512) == ("i", "v")
    assert module.run_fused_paged_indexer(num_heads=32, topk=512) == ("i", "v")
    rows = markers(capsys)
    assert len(rows) == 1
    assert rows[0] == {
        "schema": receipt.SCHEMA,
        "module": receipt.MODULE,
        "mode": receipt.MODE,
        "original_sha256": receipt.ORIGINAL_SHA256,
        "emitted_sha256": receipt.EMITTED_SHA256,
        "inspect_source_sha256": receipt.EMITTED_SHA256,
        "cache_suffix": receipt.CACHE_SUFFIX,
        "num_heads": 32,
        "topk": 512,
        "output_physical_slots": False,
        "pid": rows[0]["pid"],
        "tp_rank": 3,
        "tp_world_size": 4,
        "global_rank": 7,
        "local_rank_env": "3",
        "rank_env": "7",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 16, "topk": 512},
        {"num_heads": 32, "topk": 2048},
        {"num_heads": 32, "topk": 512, "output_physical_slots": True},
    ],
)
def test_noneligible_calls_delegate_without_marker(monkeypatch, capsys, kwargs):
    fake_rank_modules(monkeypatch)
    receipt.install(FakeOrder)
    module = load_target()
    assert module.run_fused_paged_indexer(**kwargs) == ("i", "v")
    assert markers(capsys) == []


def test_failed_eligible_call_never_emits(monkeypatch, capsys):
    fake_rank_modules(monkeypatch)
    receipt.install(FakeOrder)
    module = load_target()

    def failure(**kwargs):
        raise ValueError("launch failed")

    module.run_fused_paged_indexer.__wrapped__ = failure
    # Replace with a newly wrapped target so the closure delegates to failure.
    bare = ModuleType(receipt.MODULE)
    bare.run_fused_paged_indexer = failure
    receipt._wrap_entrypoint(bare, FakeOrder)
    with pytest.raises(ValueError, match="launch failed"):
        bare.run_fused_paged_indexer(num_heads=32, topk=512)
    assert markers(capsys) == []


def test_eligible_call_requires_initialized_rank_without_delegating(monkeypatch):
    fake_rank_modules(monkeypatch, initialized=False)
    receipt.install(FakeOrder)
    module = load_target()
    with pytest.raises(RuntimeError, match="distributed init"):
        module.run_fused_paged_indexer(num_heads=32, topk=512)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_sha256", "0" * 64),
        ("emitted_sha256", "1" * 64),
        ("inspect_source_sha256", "2" * 64),
        ("cache_suffix", "_wrong"),
        ("mode", "wrong"),
    ],
)
def test_source_receipt_drift_fails_import_after_single_delegate(
    monkeypatch, field, value
):
    original = FakeLoader.exec_module

    def drifted(self, module):
        result = original(self, module)
        FakeOrder.TRANSFORMATIONS[self.name][field] = value
        return result

    FakeLoader.exec_module = drifted
    receipt.install(FakeOrder)
    with pytest.raises(RuntimeError, match="source receipt differs"):
        load_target()
    assert FakeLoader.calls == 1


def test_missing_source_receipt_fails_after_single_delegate(monkeypatch):
    original = FakeLoader.exec_module

    def missing(self, module):
        result = original(self, module)
        FakeOrder.TRANSFORMATIONS.clear()
        return result

    FakeLoader.exec_module = missing
    receipt.install(FakeOrder)
    with pytest.raises(RuntimeError, match="without its source receipt"):
        load_target()
    assert FakeLoader.calls == 1


def test_loader_failure_is_unchanged_and_never_wraps(monkeypatch):
    def failure(self, module):
        type(self).calls += 1
        raise ImportError("exact loader failed")

    FakeLoader.exec_module = failure
    receipt.install(FakeOrder)
    module = ModuleType(receipt.MODULE)
    with pytest.raises(ImportError, match="exact loader failed"):
        FakeOrder._Loader().exec_module(module)
    assert FakeLoader.calls == 1
    assert not hasattr(module, "run_fused_paged_indexer")


@pytest.mark.parametrize("name", ["GLM53_P8_INDEX_ORDER", "GLM53_P8_INDEX_ORDER_RECEIPT"])
def test_install_requires_exact_opt_in(monkeypatch, name):
    monkeypatch.setenv(name, "")
    with pytest.raises(RuntimeError, match="requires"):
        receipt.install(FakeOrder)


def test_install_order_and_duplicates_fail_closed(monkeypatch):
    receipt.install(FakeOrder)
    with pytest.raises(RuntimeError, match="duplicate"):
        receipt.install(FakeOrder)

    monkeypatch.setattr(receipt, "_INSTALLED", False)
    monkeypatch.setitem(sys.modules, receipt.MODULE, ModuleType(receipt.MODULE))
    with pytest.raises(RuntimeError, match="before fused-indexer import"):
        receipt.install(FakeOrder)


def test_marker_is_compact_parseable_json(monkeypatch, capsys):
    fake_rank_modules(monkeypatch)
    receipt.install(FakeOrder)
    load_target().run_fused_paged_indexer(num_heads=32, topk=512)
    line = capsys.readouterr().out.strip()
    assert line.startswith(receipt.PREFIX)
    assert " " not in line[len(receipt.PREFIX) :]
    assert json.loads(line[len(receipt.PREFIX) :])["tp_rank"] == 3
