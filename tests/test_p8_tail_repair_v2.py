"""CPU contract tests for tail-after-valid-history V2."""
from pathlib import Path

import numpy as np

from glm53_nvfp4 import p8_tail_repair as v1
from glm53_nvfp4 import p8_tail_repair_v2 as v2


REPO = Path(__file__).resolve().parents[1]


def baseline_row(length: int, topk: int = 2048, pool_size: int = 4) -> np.ndarray:
    out = np.full(topk + pool_size - 1, -1, dtype=np.int32)
    tail = length % pool_size
    tail_start = length - tail
    out[:tail_start] = np.arange(tail_start, dtype=np.int32)
    out[topk:topk + tail] = np.arange(tail_start, length, dtype=np.int32)
    return out


def repaired_row(length: int, topk: int = 2048, pool_size: int = 4) -> np.ndarray:
    out = np.full(topk + pool_size - 1, -1, dtype=np.int32)
    out[:length] = np.arange(length, dtype=np.int32)
    return out


def test_per_row_diff_is_exactly_nonzero_L_mod_4():
    differing = []
    for length in range(1, 2048):
        changed = not np.array_equal(baseline_row(length), repaired_row(length))
        assert changed == (length % 4 != 0)
        if changed:
            differing.append(length - 1)
    assert len(differing) == 1536
    assert {mod: sum((row + 1) % 4 == mod for row in differing) for mod in range(4)} == {0: 0, 1: 512, 2: 512, 3: 512}


def test_patch_places_tail_after_valid_history_and_preserves_overflow_rule():
    text = v2.PATCH.read_text()
    assert "valid_history_count = pool_len * POOL_SIZE" in text
    assert "history_count = tl.where(" in text
    assert "all_complete_pools_fit, valid_history_count, displaced_history_count" in text
    assert "tail_off = cols - history_count" in text
    assert text.count("P8 tail-after-valid-history-v2") == 1


def test_v2_runner_has_distinct_identity_and_safe_v1_binding():
    assert v2.PREFIX != v1.PREFIX
    assert v2.PORT != v1.PORT
    assert v2.IMAGE_RECEIPT != v1.IMAGE_RECEIPT
    assert v2._V1_SOURCE_FILES is not v2.source_files
    with v2.patched_prior():
        assert v1.source_files is v2.source_files
        assert "glm53_nvfp4/p8_tail_repair_v2.py" in v1.source_files()
    assert v1.source_files is v2._V1_SOURCE_FILES
