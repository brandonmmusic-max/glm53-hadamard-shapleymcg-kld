import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('v2_stop', Path(__file__).resolve().parents[1] / 'scripts/snapshot_p8_capture_v2_stop.py')
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, dest, plan = tmp_path / 'raw', tmp_path / 'snapshot', tmp_path / 'plan.json'
    root.mkdir()
    plan.write_text(json.dumps({'output': str(root), 'capture_image': stop.IMAGE}))
    digest = stop.receipt(plan)['sha256']
    plan.with_suffix('.sha256').write_text(digest)
    stages = []
    for index, name in enumerate(('canary-n128', 'canary-n64')):
        slot = root / name
        (slot / 'captures').mkdir(parents=True)
        (slot / 'requests').mkdir()
        for filename in (stop.N128_FILES if index == 0 else stop.N64_FILES):
            p = slot / filename
            p.parent.mkdir(exist_ok=True)
            text = 'PRIVATE_SECRET'
            if filename == 'container-final.private.json':
                text = json.dumps({'Id': 'a' * 64, 'Image': stop.IMAGE, 'State': {'Running': False}, 'Env': ['PRIVATE_SECRET']})
            elif filename == 'error.private.txt':
                text = stop.PORT_ERROR
            elif filename in (stop.META_NAME, 'models.json', 'runtime-ready-audit.json', 'runtime-final-audit.json'):
                text = '{}'
            p.write_text(text)
        rows = [] if index else [{'id': stop.WINDOW, 'rows': 2047, 'raw_sha256': stop.receipt(slot / stop.RAW_NAME)['sha256']}]
        stage = {'exit_code': index, 'capture_image': stop.IMAGE, 'plan_sha256': digest,
                 'cleanup': {'ok': True, 'errors': []}, 'protected_roles_opened': [], 'unreadable_artifacts': [],
                 'windows': rows, 'files': {p: stop.receipt(slot / p) for p in (stop.N128_FILES if index == 0 else stop.N64_FILES)}}
        if index:
            stage['error_type'] = 'OSError'
        else:
            stage['container_id'] = 'a' * 64
        (slot / 'execution.json').write_text(json.dumps(stage))
        stages.append({'slot': name, 'execution_sha256': stop.receipt(slot / 'execution.json')['sha256']})
    execution = {'exit_code': 1, 'plan_sha256': digest, 'capture_image': stop.IMAGE,
                 'protected_roles_opened': [], 'allocation_restart': False, 'speed_measurement_valid': False,
                 'final_identity_audit': {'ok': True}, 'restoration_safety': {'ok': True, 'errors': [], 'containers': []},
                 'prior': {'backend': True, 'timer': False}, 'restoration': {'backend': True, 'timer': False, 'errors': []},
                 'stages': stages}
    (root / 'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(stop, 'SOURCE', root)
    monkeypatch.setattr(stop, 'PLAN', plan)
    monkeypatch.setattr(stop, 'PLAN_SHA', digest)
    monkeypatch.setattr(stop, 'ROOT_SHA', stop.receipt(root / 'execution.json')['sha256'])
    monkeypatch.setattr(stop, 'terminal_state', lambda: {'ActiveState': 'failed'})
    monkeypatch.setattr(stop, 'validate_original', lambda: {'rows': 2047, 'raw_sha256': stop.receipt(root / 'canary-n128' / stop.RAW_NAME)['sha256']})
    return root, dest


def test_preserves_unpaired_capture_and_excludes_private_bytes(case):
    root, dest = case
    stop.snapshot(dest)
    report = json.loads((dest / 'report.json').read_text())
    assert report['n128_causal_rows'] == 2047 and report['n64_started'] is False
    assert report['paired_closure_passed'] is False and report['kld_measured'] is False
    assert report['raw_logits']['copied_to_repository'] is False
    assert (dest / 'execution.json').read_bytes() == (root / 'execution.json').read_bytes()
    assert not list(dest.rglob('*.f32')) and not list(dest.rglob('*.private.*'))
    manifest = json.loads((dest / 'snapshot.json').read_text())
    for path, expected in manifest['outputs'].items():
        assert stop.receipt(dest / path) == expected
    for p in dest.rglob('*'):
        if p.is_file():
            assert b'PRIVATE_SECRET' not in p.read_bytes()
    diagnostic = json.loads((dest / 'port-stop.sanitized.json').read_text())['diagnostic']
    assert 'main-agent-observed' in diagnostic['evidence_level']


@pytest.mark.parametrize('name', ['execution.json', 'canary-n128/execution.json',
                                'canary-n128/' + stop.RAW_NAME, 'canary-n64/error.private.txt',
                                'canary-exact.json', 'canary-n64/requests/request.json',
                                'canary-n64/captures/raw.f32'])
def test_tamper_or_extra_evidence_fails_before_creating_output(case, name):
    root, dest = case
    (root / name).write_text('changed')
    with pytest.raises(ValueError):
        stop.snapshot(dest)
    assert not dest.exists()


def test_original_validator_failure_cannot_publish(case, monkeypatch):
    _, dest = case
    def fail():
        raise ValueError('original runtime proof mismatch')
    monkeypatch.setattr(stop, 'validate_original', fail)
    with pytest.raises(ValueError):
        stop.snapshot(dest)
    assert not dest.exists()


def test_no_overwrite(case):
    _, dest = case
    dest.mkdir()
    (dest / 'keep').write_text('unchanged')
    with pytest.raises(ValueError, match='fresh'):
        stop.snapshot(dest)
    assert (dest / 'keep').read_text() == 'unchanged'


def test_rejects_running_unit(monkeypatch):
    monkeypatch.setattr(stop.subprocess, 'check_output', lambda *a, **k: 'ActiveState=active\nResult=success\nExecMainStatus=0\n')
    with pytest.raises(ValueError, match='terminal'):
        stop.terminal_state()
