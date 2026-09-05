import json

import numpy as np
import pytest

from scripts import preflight_p8_index_order_device as probe


def outputs(length):
    ids = np.full((1, 512), -1, dtype=np.int32)
    values = np.full((1, 512), -np.inf, dtype=np.float32)
    chosen = np.arange(max(0, length - 512), length, dtype=np.int32)
    ids[0, :len(chosen)] = chosen
    values[0, :len(chosen)] = 32 * (chosen + 1)
    return ids, values


@pytest.mark.parametrize('length', probe.LENGTHS)
def test_exact_reference_and_suffix(length):
    ids, values = outputs(length)
    assert probe.assert_result(ids, values, length, ordered_short=True)['valid_count'] == min(length, 512)


def test_original_order_is_not_mistaken_for_changed_selection():
    ids, values = outputs(65)
    ids[0, :65] = np.roll(ids[0, :65], 1)
    values[0, :65] = np.roll(values[0, :65], 1)
    probe.assert_result(ids, values, 65, ordered_short=False)
    with pytest.raises(AssertionError, match='logical ascending'):
        probe.assert_result(ids, values, 65, ordered_short=True)


@pytest.mark.parametrize('bad', ['duplicate', 'score-bit', 'suffix-index', 'suffix-score'])
def test_rejects_any_pair_or_suffix_drift(bad):
    ids, values = outputs(65)
    if bad == 'duplicate':
        ids[0, 0] = 1
    elif bad == 'score-bit':
        values.view(np.uint32)[0, 0] ^= 1
    elif bad == 'suffix-index':
        ids[0, 511] = 0
    else:
        values[0, 511] = 0
    with pytest.raises(AssertionError):
        probe.assert_result(ids, values, 65, ordered_short=True)


def test_shuffled_pages_preserve_exact_logical_scale_and_split_layout():
    data = probe.synthetic_inputs(20260905)
    packed = data['packed']
    assert packed.shape == (17, 8448)
    assert not np.array_equal(data['table'][0], np.arange(17))
    for logical, physical in enumerate(data['table'][0]):
        k = packed[physical, :8192].reshape(64, 128)
        scales = packed[physical, 8192:].view(np.float32)
        assert np.all(k[:, 0] == 0x38) and not np.any(k[:, 1:])
        assert np.array_equal(scales, np.arange(logical * 64 + 1, (logical + 1) * 64 + 1))
    assert np.array_equal(data['packed'], probe.synthetic_inputs(20260905)['packed'])


def test_counter_guard_ignores_histogram_but_rejects_live_state():
    state = np.zeros(772, dtype=np.int32)
    state[:768] = 13
    probe.assert_counters(state)
    state[771] = 1
    with pytest.raises(AssertionError, match='reset'):
        probe.assert_counters(state)


def test_failure_receipt_without_gpu_or_model(tmp_path, monkeypatch):
    def fail(record, seed):
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(probe, 'run_device', fail)
    path = tmp_path / 'result.json'
    assert probe.main(['--output', str(path), '--image-id', probe.IMAGE]) == 1
    result = json.loads(path.read_text())
    assert result['status'] == 'failed' and result['model_loaded'] is False
    assert result['teacher_logits_opened'] is False and result['kld_measured'] is False
    assert result['failure']['type'] == 'RuntimeError'
    with pytest.raises(ValueError, match='fresh'):
        probe.main(['--output', str(path), '--image-id', probe.IMAGE])


def test_declared_dynamic_transition_has_no_reset_between_lengths():
    assert probe.TRANSITIONS == (512, 513) * 5 + (513, 512) * 5
    assert probe.THRESHOLD - 1 in probe.LENGTHS
    assert probe.THRESHOLD in probe.LENGTHS and probe.THRESHOLD + 1 in probe.LENGTHS


@pytest.mark.parametrize('valid_ids', [[64], list(range(64))])
def test_negative_page_reference_keeps_only_valid_logical_rows(valid_ids):
    ids = np.full((1, 512), -1, dtype=np.int32)
    values = np.full((1, 512), -np.inf, dtype=np.float32)
    ids[0, :len(valid_ids)] = valid_ids
    values[0, :len(valid_ids)] = 32 * (np.array(valid_ids) + 1)
    probe.assert_result(ids, values, 65, ordered_short=False, valid_ids=valid_ids)
    with pytest.raises(AssertionError, match='index set'):
        probe.assert_result(ids, values, 65, ordered_short=False)
