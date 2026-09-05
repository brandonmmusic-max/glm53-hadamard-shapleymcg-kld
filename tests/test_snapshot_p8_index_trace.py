import json

import pytest

from scripts import snapshot_p8_index_trace as snap


def test_log_sanitizer_keeps_only_typed_proofs():
    text = ('SECRET=do-not-publish\nUsing V2 Model Runner\n'
            'GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=0 max_num_reqs=1 real_vocab=154880 expected_outputs=2047 SECRET=bad\n'
            'GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=0 registrations=1 samples=2\n'
            'GLM53_P8_INDEX_TRACE_COMPLETE tp_rank=0 window=conditional-fit-0056 rows=9 layers=11\n')
    safe = snap.sanitize_log(text).decode()
    assert 'SECRET' not in safe and 'do-not-publish' not in safe
    assert 'Using V2 Model Runner' in safe
    assert 'rows=9 layers=11' in safe
    assert 'registrations=1 samples=2' in safe


def test_thermal_projection_excludes_unknown_fields():
    data = json.dumps({'at': '2026-09-05T09:00:00+00:00', 'temperatures': [40, 41, 42, 43],
                       'private': 'SECRET'}).encode()
    assert json.loads(snap.thermal_projection(data)) == {
        'at': '2026-09-05T09:00:00+00:00', 'temperatures': [40, 41, 42, 43]}


@pytest.mark.parametrize('values', [[40], [40, 41, 42, float('nan')], [40, 41, 42, True]])
def test_thermal_refuses_bad_rows(values):
    with pytest.raises(ValueError):
        snap.thermal_projection(json.dumps({'at': '2026-09-05T09:00:00+00:00', 'temperatures': values}).encode())


def test_no_overwrite_before_live_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, 'terminal_state', lambda: pytest.fail('must refuse before live check'))
    with pytest.raises(ValueError, match='no overwrite'):
        snap.snapshot(tmp_path)


def test_terminal_refuses_remaining_container(monkeypatch):
    monkeypatch.setattr(snap.subprocess, 'check_output', lambda argv, **kwargs:
        'ActiveState=inactive\nResult=success\nMainPID=0\n' if argv[0] == 'systemctl' else 'container\n')
    with pytest.raises(ValueError, match='container remains'):
        snap.terminal_state()
