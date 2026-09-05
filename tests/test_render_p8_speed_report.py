import copy

import pytest

from glm53_nvfp4.render_p8_speed_report import figure_svg, render


def sample():
    analysis = {'plan': {'sha256': 'plan'}, 'decision': {
        'passed': False, 'allocation_game': 'stop', 'rule': 'both medians must win'}}
    audit = {'status': 'pass', 'missing': [], 'plan_sha256': 'plan', 'runs': []}
    for arm, speed in [('p8', 20.), ('exl3', 90.)]:
        runs = []
        for i in range(1, 6):
            digest = f'{arm}-{i}'
            run = {'sha256': digest,
                   'prefill': {c: {'prompt_tokens': int(c)-100, 'server_tps': 6000.} for c in ('32768', '65536')},
                   'decode': {c: {'aggregate_tps': speed} for c in ('32768', '65536')}}
            runs.append(run)
            audit['runs'].append({'round': i, 'arm': arm, 'container_id': digest,
                                  'benchmark_sha256': digest, 'measured_max_temp_c': 88.})
        analysis[arm] = {'runs': runs, 'median': {m: {c: {f: v} for c in ('32768', '65536')}
                         for m, f, v in [('prefill', 'server_tps', 6000.), ('decode', 'aggregate_tps', speed)]}}
    return analysis, audit


def test_renders_all_runs_and_claim_boundaries():
    report = render(*sample())
    assert 'FAIL — allocation game stopped' in report
    assert '| 5 | P8 |' in report and '| 5 | EXL3 |' in report
    assert 'twice the NVFP4 MMA issue count' in report
    assert 'not a matched NVFP4 win' in report


def test_svg_is_repeatable_and_not_rendered_for_partial_evidence():
    pytest.importorskip('matplotlib')
    a, b = sample()
    first, second = figure_svg(a, b), figure_svg(a, b)
    assert first == second
    assert b'<svg' in first
    b['status'] = 'incomplete'
    with pytest.raises(ValueError): figure_svg(a, b)


@pytest.mark.parametrize('failure', ['partial', 'duplicate', 'hash', 'median', 'decision'])
def test_inconsistent_inputs_are_not_publishable(failure):
    a, b = copy.deepcopy(sample())
    if failure == 'partial': b['status'] = 'incomplete'
    if failure == 'duplicate': b['runs'][1]['container_id'] = b['runs'][0]['container_id']
    if failure == 'hash': b['runs'][0]['benchmark_sha256'] = 'changed'
    if failure == 'median': a['p8']['median']['decode']['32768']['aggregate_tps'] = 100.
    if failure == 'decision': a['decision']['passed'] = True
    with pytest.raises(ValueError): render(a, b)
