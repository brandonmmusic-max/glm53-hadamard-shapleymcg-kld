import copy

import pytest

from glm53_nvfp4.analyze_p8_smallm_device import analyze


def payload(tokens, *, candidate=False, timing=False):
    cells = []
    for token in tokens:
        cells.append({'tokens': token, 'output_sha256': f'hash-{token}',
                      'timing': ({'warmups': 20, 'repeats': 100,
                                  'median_ms': .25 if candidate else 1.0}
                                 if timing else None)})
    return {'sidecar': {'sha256': 'sidecar'}, 'dense': {'sha256': 'dense'},
            'rank': 0, 'layer': 3, 'decision': 'pass',
            'deterministic_output': True, 'small_m_scheduler': candidate,
            'cells': cells}


def test_accepts_exact_outputs_and_reports_timing():
    base, cand = payload([1]), payload([1], candidate=True)
    base_t, cand_t = payload([1], timing=True), payload([1], candidate=True, timing=True)
    base_f, cand_f = payload([2, 3]), payload([2, 3], candidate=True)
    result = analyze(base, cand, base_t, cand_t, base_f, cand_f)
    assert result['status'] == 'pass'
    assert result['device_time_reduction_percent'] == 75
    assert result['device_time_speedup'] == 4


def test_rejects_output_or_identity_mismatch():
    args = [payload([1]), payload([1], candidate=True), payload([1], timing=True),
            payload([1], candidate=True, timing=True), payload([2, 3]),
            payload([2, 3], candidate=True)]
    changed = copy.deepcopy(args)
    changed[1]['cells'][0]['output_sha256'] = 'different'
    with pytest.raises(ValueError, match='M1 output differs'):
        analyze(*changed)
    changed = copy.deepcopy(args)
    changed[5]['dense']['sha256'] = 'different'
    with pytest.raises(ValueError, match='identities differ'):
        analyze(*changed)
