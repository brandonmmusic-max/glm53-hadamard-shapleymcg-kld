"""CPU-only contract tests for the single-variable KPool tail repair."""
from pathlib import Path

import numpy as np

from glm53_nvfp4 import p8_tail_repair as runner


REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "runtime_patch/p8_tail_repair/kpool-tail-consumed-v1.patch"


def repaired_order(history: np.ndarray, seq_len: int, pool_size: int = 4) -> np.ndarray:
    tail_count = seq_len % pool_size
    tail_start = seq_len - tail_count
    keep = history.size - tail_count
    return np.concatenate((history[:keep], np.arange(tail_start, seq_len, dtype=history.dtype)))


def test_tail_replaces_exactly_the_final_ordered_history_entries():
    history = np.arange(2048, dtype=np.int32) + 10000
    for tail_count in range(4):
        seq_len = 4096 + tail_count
        result = repaired_order(history, seq_len)
        assert result.shape == (2048,)
        assert np.array_equal(result[: 2048 - tail_count], history[: 2048 - tail_count])
        assert np.array_equal(result[2048 - tail_count :], np.arange(4096, seq_len))


def test_patch_changes_only_tail_placement_expressions():
    text = PATCH.read_text()
    assert "history_count = topk - tail_count" in text
    assert "+    is_history = cols < history_count" in text
    assert "+    tail_off = cols - history_count" in text
    assert "pool_ids_ptr + row * pid_s0 + g" in text
    assert text.count("P8 tail-consumed-v1") == 1


def test_runner_binds_the_unwrapped_clone_before_adapter_installation():
    assert runner._BASE_CLONE_ARGV is not runner.clone_argv
