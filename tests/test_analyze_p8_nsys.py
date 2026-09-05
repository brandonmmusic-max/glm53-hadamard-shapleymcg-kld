import pytest

from glm53_nvfp4.analyze_p8_nsys import (
    chunk_replays,
    infer_active_m,
    interval_union_ns,
)


def test_interval_union_preserves_overlap_without_double_counting():
    assert interval_union_ns([(0, 10), (5, 20), (30, 35)]) == 25


def test_replay_chunking_requires_repeated_node_sequence():
    rows = [
        (0, 1, 0, 0, 1, 1, 10),
        (1, 2, 0, 0, 1, 1, 11),
        (3, 4, 0, 0, 1, 1, 10),
        (4, 5, 0, 0, 1, 1, 11),
    ]
    chunks = chunk_replays(rows, 2)
    assert len(chunks) == 2
    rows[-1] = (*rows[-1][:-1], 12)
    with pytest.raises(ValueError, match='inventory differs'):
        chunk_replays(rows, 2)


def test_replay_chunking_rejects_nonintegral_inventory():
    with pytest.raises(ValueError, match='cannot be divided'):
        chunk_replays([(0, 1, 0, 0, 1, 1, 10)], 2)


def test_topk_grid_is_exact_m1_witness():
    assert infer_active_m({16}) == 1
    assert infer_active_m({32}) == 2
    with pytest.raises(ValueError, match='differs'):
        infer_active_m({16, 32})
