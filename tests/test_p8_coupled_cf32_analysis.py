import copy

import pytest

from glm53_nvfp4.analyze_p8_coupled_cf32 import analyze, ARMS


def inputs():
    windows = [{'id': f'conditional-fit-{i:04d}', 'domain': f'd{i % 4}'} for i in range(32)]
    arms = {arm: {
        'conditions': dict(attention='fixture', kv_dtype='fixture', moe_backend='fixture',
                           activation_precision='fixture', bpw=4.25),
        'windows': [dict(window_id=w['id'], domain=w['domain'],
                         mean_kld=0.5, true_decode_mean_kld=0.25) for w in windows],
    } for arm in ARMS}
    return windows, arms


def test_ci_is_nonblocking_and_rows_are_paired_by_id():
    windows, arms = inputs()
    for i, row in enumerate(arms['coupled_p8']['windows']):
        row['true_decode_mean_kld'] += -0.01 if i < 17 else 0.01
    result = analyze(windows, arms)
    assert result['decision'] == 'pass'
    assert result['stock'] == {'status': 'not-tested', 'historical_only': True, 'metric': None,
                               'claim': 'No fresh stock baseline; no equivalence or stock delta is inferred'}
    assert result['comparisons']['identity_p8']['paired_bca95'][1] > 0
    assert result['comparisons']['identity_p8']['paired_window_wins'] == 17
    arms['coupled_p8']['windows'].reverse()
    assert analyze(windows, arms) == result


def test_prefill_is_secondary_and_constant_delta_is_disclosed():
    windows, arms = inputs()
    for row in arms['coupled_p8']['windows']:
        row['mean_kld'] = 0.001
    result = analyze(windows, arms)
    assert result['decision'] == 'fail'
    assert result['comparisons']['identity_p8']['degenerate_delta_distribution']
    assert result['comparisons']['identity_p8']['paired_bca95'] == [0, 0]


@pytest.mark.parametrize('corruption', ['duplicate', 'domain', 'nan', 'negative', 'label', 'missing'])
def test_malformed_arm_rejected(corruption):
    windows, arms = inputs()
    entry = arms['coupled_p8']
    if corruption == 'duplicate':
        entry['windows'][-1] = copy.deepcopy(entry['windows'][0])
    elif corruption == 'domain':
        entry['windows'][0]['domain'] = 'wrong'
    elif corruption == 'nan':
        entry['windows'][0]['true_decode_mean_kld'] = float('nan')
    elif corruption == 'negative':
        entry['windows'][0]['mean_kld'] = -0.1
    elif corruption == 'label':
        del entry['conditions']['kv_dtype']
    else:
        entry['windows'].pop()
    with pytest.raises(ValueError):
        analyze(windows, arms)


def test_unbalanced_manifest_rejected():
    windows, arms = inputs()
    windows[0]['domain'] = 'd1'
    with pytest.raises(ValueError):
        analyze(windows, arms)
