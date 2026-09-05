from __future__ import annotations

import copy
import numpy as np
import pytest

from glm53_nvfp4.p8_index_trace_analysis import bitwise_equal, compare_row, decode, verify_row


def example(position=259, swap=False):
    length, plen = position + 1, (position + 1) // 4
    pools = np.full(512, -1, dtype=np.int32)
    ids = np.arange(plen)
    if swap:
        ids = np.roll(ids, 1)
    pools[:plen] = ids
    logical = np.full(2051, -1, dtype=np.int32)
    logical[:2048] = np.where(pools[:, None] >= 0, pools[:, None] * 4 + np.arange(4), -1).reshape(-1)
    logical[2048:2048 + length % 4] = np.arange(plen * 4, length)
    table = np.array([3, 1, 4, 0, 2], dtype=np.int32)
    physical = np.full(2048, -1, dtype=np.int32)
    consumed = logical[:2048]
    good = consumed >= 0
    physical[good] = table[consumed[good] // 64] * 64 + consumed[good] % 64
    return {'pool_ids': pools, 'logical_tokens': logical, 'physical_slots': physical,
        'pool_lengths': np.array([plen]), 'attention_lengths': np.array([length]),
        'attention_table': table, 'pool_table': np.array([2, 0]),
        'pool_cache': np.zeros((66, 132), np.uint8), 'attention_cache': np.zeros((264, 288), np.uint8),
        'index_query': np.array([1]), 'index_weights': np.array([2]),
        'attention_query': np.array([3]), 'attention_output': np.array([4 if not swap else 5])}


@pytest.mark.parametrize('position', range(255, 264))
def test_complete_prefix_and_page_mapping(position):
    for swap in (False, True):
        verify_row(example(position, swap), position, {'page_size': 64})


def test_same_input_order_difference_links_output():
    result = compare_row(example(), example(swap=True))
    assert result['index_order_divergence_with_same_inputs']
    assert result['order_linked_attention_divergence']


def test_unconsumed_tail_is_reported_not_conflated_with_mapped_width():
    row = example(260)
    verify_row(row, 260, {'page_size': 64})
    result = compare_row(row, copy.deepcopy(row))
    assert result['expanded_tail_tokens_a'] == [260]
    assert result['tail_columns_consumed'] is False
    row['physical_slots'] = np.pad(row['physical_slots'], (0, 3), constant_values=-1)
    with pytest.raises(ValueError, match='2048'):
        verify_row(row, 260, {'page_size': 64})


def test_changed_inputs_do_not_masquerade_as_same_input_race():
    a, b = example(), example(swap=True)
    b['index_query'][0] += 1
    b['attention_cache'][0, 0] = 1
    result = compare_row(a, b)
    assert not result['index_order_divergence_with_same_inputs']
    assert not result['order_linked_attention_divergence']


def test_linkage_requires_same_scorer_inputs_in_the_same_row():
    a, b = example(), example(swap=True)
    b['index_query'][0] += 1
    result = compare_row(a, b)
    assert result['attention_content_bitwise_equal']
    assert not result['index_order_divergence_with_same_inputs']
    assert not result['order_linked_attention_divergence']


@pytest.mark.parametrize('label', ['pool_ids', 'logical_tokens', 'physical_slots'])
def test_bad_index_data_rejected(label):
    row = example()
    row[label][0] = 999
    with pytest.raises(ValueError):
        verify_row(row, 259, {'page_size': 64})


def test_invalid_cache_bytes_rejected():
    row = example(255)
    row['pool_cache'][65, 0] = 1
    with pytest.raises(ValueError, match='masked'):
        verify_row(row, 255, {'page_size': 64})


def test_bf16_is_bits_not_fp16_values():
    raw = np.full((9, 4), 128, dtype=np.uint8)
    data = decode(raw, {'torch_dtype': 'torch.bfloat16', 'logical_shape': [1, 2]})
    assert data.dtype == np.dtype('<u2') and data.shape == (9, 1, 2)


def test_bitwise_comparison_distinguishes_signed_zero_and_dtypes():
    assert not bitwise_equal(np.array([0.0], np.float32), np.array([-0.0], np.float32))
    assert not bitwise_equal(np.array([0], np.int32), np.array([0], np.int64))
    a, b = example(), example(swap=True)
    a['index_weights'] = np.array([0.0], np.float32)
    b['index_weights'] = np.array([-0.0], np.float32)
    assert not compare_row(a, b)['index_order_divergence_with_same_inputs']
