from __future__ import annotations

import hashlib
import inspect
import linecache
from pathlib import Path
import sys
import types

import pytest

from runtime_patch import p8_index_order as loader
from runtime_patch.p8_index_order.patches import (
    CACHE_SUFFIX,
    MODE,
    MODULE,
    ORIGINAL_SHA,
    direct_ranges,
    transform,
)


REPO = Path(__file__).resolve().parents[1]
EXACT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/"
    "b12x/b12x/attention/nsa_indexer/fused_indexer.py"
)


def exact_raw() -> bytes:
    if not EXACT.is_file():
        pytest.skip("exact pinned fused-indexer source is unavailable on this host")
    raw = EXACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    return raw


def transformed():
    return transform(MODULE, exact_raw())


def test_exact_source_transform_is_deterministic_and_receipted():
    source1, receipt1 = transformed()
    source2, receipt2 = transformed()
    assert source1 == source2 and receipt1 == receipt2
    assert receipt1["original_sha256"] == ORIGINAL_SHA
    assert receipt1["emitted_sha256"] == hashlib.sha256(source1.encode()).hexdigest()
    assert receipt1["mode"] == MODE
    assert receipt1["cache_suffix"] == CACHE_SUFFIX
    assert receipt1["contract"] == {
        "kv_layout": "paged",
        "paged_output": False,
        "num_heads_static": 32,
        "topk_pools": 512,
        "cross_cta_merge": True,
        "runtime_pool_seq_len_max": 512,
        "negative_or_missing_page": "legacy-relay",
        "positive_page_oob": "unchanged caller-validity precondition; scorer reads first",
        "tail_expansion_changed": False,
    }


def test_wrong_module_or_source_fails_closed():
    raw = exact_raw()
    with pytest.raises(ValueError, match="unsupported"):
        transform("b12x.attention.other", raw)
    with pytest.raises(ValueError, match="source differs"):
        transform(MODULE, raw + b"\n")


def test_transform_compiles_and_preserves_one_legacy_relay():
    source, _ = transformed()
    compile(source, "<test-index-order>", "exec", dont_inherit=True)
    assert source.count("atomic_add_global_i32(woff_ptr, carry_count)") == 1
    assert source.count("if total > topk_static:") == 1
    assert source.count("p8_direct_short != Int32(0)") == 1
    assert source.count("variant += \"_p8logicalshortv1\"") == 1
    assert "self.num_heads_static == 32" in source
    assert "and self.topk == 512" in source
    assert "and not self.paged_output" in source
    assert "if seq_len > topk_static:" in source


@pytest.mark.parametrize("seq_len", [0, 1, 63, 64, 65, 255, 256, 257, 511, 512])
@pytest.mark.parametrize("ctas", [1, 2, 4, 188])
def test_direct_geometry_exact_ownership(seq_len, ctas):
    ranges = direct_ranges(seq_len, ctas)
    ownership = [0] * 512
    for begin, end in ranges:
        assert 0 <= begin <= end <= seq_len
        for index in range(begin, end):
            ownership[index] += 1
    for index in range(seq_len, 512):
        ownership[index] += 1  # CTA 0's suffix loop.
    assert ownership == [1] * 512
    flattened = [index for begin, end in ranges for index in range(begin, end)]
    assert flattened == list(range(seq_len))


@pytest.mark.parametrize(
    "seq_len,ctas,topk",
    [(-1, 2, 512), (513, 2, 512), (65, 0, 512), (65, 2, 1024)],
)
def test_direct_geometry_rejects_out_of_contract(seq_len, ctas, topk):
    with pytest.raises(ValueError):
        direct_ranges(seq_len, ctas, topk=topk)


def test_invalid_table_scan_is_group_uniform_and_legacy_fallback_is_collective():
    source, _ = transformed()
    scan = source.index("p8_scan_pages = total_pages")
    publish = source.index("s_relay[0] = p8_ok", scan)
    sync = source.index("cute.arch.sync_threads()", publish)
    read = source.index("p8_direct_short = Int32(s_relay[0])", sync)
    branch = source.index("if p8_direct_short != Int32(0):", read)
    legacy = source.index("# ---- in-kernel cross-CTA merge (relay) ----", branch)
    assert scan < publish < sync < read < branch < legacy
    assert "p8_pid >= Int32(k_quant_bytes.shape[0])" in source[scan:publish]
    assert "p8_pid >= Int32(k_scales.shape[0])" in source[scan:publish]
    long_guard = source.index("if seq_len > topk_static:", scan)
    zero_scan = source.index("p8_scan_pages = Int32(0)", long_guard)
    loop = source.index("while p8_page < p8_scan_pages:", zero_scan)
    assert long_guard < zero_scan < loop


def test_direct_write_has_post_append_cta_fence_and_no_merge_state_access():
    source, _ = transformed()
    begin = source.index("if p8_direct_short != Int32(0):")
    end = source.index("                else:\n                    # ---- in-kernel", begin)
    direct = source[begin:end]
    assert direct.index("cute.arch.sync_threads()") < direct.index("s_c0_values[i]")
    assert "merge_state" not in direct
    assert "atomic_add" not in direct
    assert "pack_values" not in direct and "pack_indices" not in direct
    assert "p8_begin > seq_len" in direct  # Empty CTAs clamp to an empty range.
    assert "if cta_in_group == Int32(0):" in direct


def test_cache_suffix_is_only_for_exact_compiled_contract():
    source, _ = transformed()
    start = source.index("p8_logical_short_v1 = (")
    end = source.index("    cache_key =", start)
    key = source[start:end]
    assert "kernel.kv_layout == KV_LAYOUT_PAGED" in key
    assert "not kernel.paged_output" in key
    assert "kernel.num_heads_static == 32" in key
    assert "kernel.topk == 512" in key
    assert "kernel.merge_in_kernel" in key
    assert key.count(CACHE_SUFFIX) == 1


def test_compile_registration_makes_inspect_read_transformed_source(tmp_path):
    origin = str(tmp_path / "fused_indexer.py")
    source = "def p8_transformed_probe():\n    return 'logical-short-v1'\n"
    code, synthetic = loader._compile_transformed(source, origin)
    module = types.ModuleType("probe")
    exec(code, module.__dict__)
    assert synthetic in linecache.cache
    assert inspect.getsource(module.p8_transformed_probe) == source
    assert module.p8_transformed_probe.__code__.co_filename == synthetic


def test_loader_makes_class_source_inspection_use_transformed_text(tmp_path, monkeypatch):
    origin = tmp_path / "fused_indexer.py"
    origin.write_text("class Probe:\n    marker = 'original'\n")
    source = "class Probe:\n    marker = 'logical-short-v1'\n"
    emitted = hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setattr(loader, "transform", lambda name, raw: (
        source,
        {"module": name, "original_sha256": "0" * 64, "emitted_sha256": emitted},
    ))
    loader.TRANSFORMATIONS.clear()
    name = "p8_index_order_inspect_probe"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        loader._Loader(name, str(origin)).exec_module(module)
        assert inspect.getsource(module.Probe) == source
        receipt = loader.TRANSFORMATIONS[name]
        assert receipt["inspect_source_sha256"] == emitted
        assert module.__file__ == receipt["synthetic_filename"]
    finally:
        sys.modules.pop(name, None)
        loader.TRANSFORMATIONS.clear()


def test_loader_uses_dont_inherit_future_annotations(tmp_path, monkeypatch):
    origin = tmp_path / "fused_indexer.py"
    origin.write_text("ignored\n")
    source = "def annotation_probe(x: MissingType):\n    return x\n"
    monkeypatch.setattr(loader, "transform", lambda name, raw: (
        source,
        {"module": name, "original_sha256": "0" * 64,
         "emitted_sha256": hashlib.sha256(source.encode()).hexdigest()},
    ))
    loader.TRANSFORMATIONS.clear()
    module = types.ModuleType(MODULE)
    with pytest.raises(NameError, match="MissingType"):
        loader._Loader(MODULE, str(origin)).exec_module(module)


def test_install_requires_exact_opt_in_and_preimport(monkeypatch):
    original_meta = list(sys.meta_path)
    try:
        monkeypatch.delenv("GLM53_P8_INDEX_ORDER", raising=False)
        with pytest.raises(RuntimeError, match="requires"):
            loader.install()
        monkeypatch.setenv("GLM53_P8_INDEX_ORDER", MODE)
        monkeypatch.setitem(sys.modules, MODULE, types.ModuleType(MODULE))
        with pytest.raises(RuntimeError, match="before"):
            loader.install()
        monkeypatch.delitem(sys.modules, MODULE)
        loader.install()
        with pytest.raises(RuntimeError, match="duplicate"):
            loader.install()
    finally:
        sys.meta_path[:] = original_meta
