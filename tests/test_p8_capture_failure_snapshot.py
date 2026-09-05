import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('capture_failure_snapshot', Path(__file__).resolve().parents[1] / 'scripts/snapshot_p8_capture_failure.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, dest, plan = tmp_path / 'raw', tmp_path / 'snapshot', tmp_path / 'plan.json'
    slot = root / snapshot.SLOT
    (slot / 'requests').mkdir(parents=True)
    (slot / 'captures').mkdir()
    plan.write_text(json.dumps({'output': str(root), 'capture_image': snapshot.IMAGE}))
    plan_sha = snapshot.sha(plan.read_bytes())
    plan.with_suffix('.sha256').write_text(plan_sha)
    container = {'Id': 'a' * 64, 'Image': snapshot.IMAGE,
                 'Name': '/glm53-p8-forced-m1-v2-v1-canary-n128',
                 'State': {'Running': False, 'ExitCode': 1}, 'Env': ['SECRET_DO_NOT_COPY']}
    log = '\n'.join(f'File "/opt/{p}", line {n}, in {f}' for p, n, f in snapshot.FRAMES)
    log += '\nRuntimeError: ' + snapshot.ERROR + '\nSECRET_DO_NOT_COPY'
    xml = '<nvidia_smi_log><driver_version>test</driver_version>' + ''.join(
        '<gpu><product_name>test</product_name><processes>SECRET_DO_NOT_COPY</processes></gpu>' for _ in range(4)) + '</nvidia_smi_log>'
    for name in snapshot.FILES:
        if name == 'container-final.private.json':
            text = json.dumps(container)
        elif name == 'server-final.private.log':
            text = log
        elif name.endswith('.xml'):
            text = xml
        elif name.endswith('.jsonl'):
            text = json.dumps({'at': '2026-09-05T08:00:00+00:00', 'temperatures': [60] * 4, 'private': 'SECRET_DO_NOT_COPY'}) + '\n'
        else:
            text = 'SECRET_DO_NOT_COPY'
        (slot / name).write_text(text)
    stage = {'exit_code': 1, 'windows': [], 'plan_sha256': plan_sha,
             'capture_image': snapshot.IMAGE, 'cleanup': {'ok': True, 'errors': []},
             'protected_roles_opened': [], 'unreadable_artifacts': [], 'container_id': 'a' * 64,
             'files': {name: {'bytes': (slot / name).stat().st_size, 'sha256': snapshot.sha((slot / name).read_bytes())} for name in snapshot.FILES}}
    (slot / 'execution.json').write_text(json.dumps(stage))
    execution = {'exit_code': 1, 'plan_sha256': plan_sha, 'capture_image': snapshot.IMAGE,
                 'protected_roles_opened': [], 'allocation_restart': False, 'speed_measurement_valid': False,
                 'final_identity_audit': {'ok': True}, 'restoration_safety': {'ok': True, 'containers': [], 'errors': []},
                 'prior': {'backend': True, 'timer': False}, 'restoration': {'backend': True, 'timer': False, 'errors': []},
                 'stages': [{'slot': snapshot.SLOT, 'execution_sha256': snapshot.sha((slot / 'execution.json').read_bytes())}]}
    (root / 'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(snapshot, 'SOURCE', root)
    monkeypatch.setattr(snapshot, 'PLAN', plan)
    monkeypatch.setattr(snapshot, 'PLAN_SHA', plan_sha)
    monkeypatch.setattr(snapshot, 'EXECUTION_SHA', snapshot.sha((root / 'execution.json').read_bytes()))
    monkeypatch.setattr(snapshot, 'terminal_state', lambda: {'ActiveState': 'failed', 'Result': 'exit-code', 'ExecMainStatus': '1'})
    return root, slot, dest


def test_preserves_failure_without_private_content(case):
    root, slot, dest = case
    assert snapshot.snapshot(dest)['status'] == 'failed-startup-preserved'
    assert (dest / 'execution.json').read_bytes() == (root / 'execution.json').read_bytes()
    report = json.loads((dest / 'report.json').read_text())
    assert report['captured_rows'] == 0 and report['kld_available'] is False
    assert report['numerical_closure_passed'] is False and report['allocation_restart'] is False
    manifest = json.loads((dest / 'snapshot.json').read_text())
    for path, receipt in manifest['outputs'].items():
        data = (dest / path).read_bytes()
        assert {'bytes': len(data), 'sha256': snapshot.sha(data)} == receipt
    for path in dest.rglob('*'):
        if path.is_file():
            assert b'SECRET_DO_NOT_COPY' not in path.read_bytes()
    assert not list(dest.rglob('*.private.*'))


@pytest.mark.parametrize('mutation', ['log', 'request', 'capture', 'next_stage', 'root_receipt', 'stage_receipt'])
def test_rejects_contrary_evidence_before_output(case, mutation):
    root, slot, dest = case
    path = {'log': slot / 'server-final.private.log', 'request': slot / 'requests/request.json',
            'capture': slot / 'captures/logits.f32', 'next_stage': root / 'full-exact.json',
            'root_receipt': root / 'execution.json', 'stage_receipt': slot / 'execution.json'}[mutation]
    path.write_text('tampered')
    with pytest.raises(ValueError):
        snapshot.snapshot(dest)
    assert not dest.exists()


def test_no_overwrite(case):
    _, _, dest = case
    dest.mkdir()
    marker = dest / 'keep'
    marker.write_text('unchanged')
    with pytest.raises(ValueError, match='fresh'):
        snapshot.snapshot(dest)
    assert marker.read_text() == 'unchanged'


def test_rejects_active_unit(case, monkeypatch):
    _, _, dest = case
    monkeypatch.undo()
    monkeypatch.setattr(snapshot.subprocess, 'check_output', lambda *args, **kwargs: 'ActiveState=active\nResult=success\nExecMainStatus=0\n')
    with pytest.raises(ValueError, match='terminal'):
        snapshot.snapshot(dest)
    assert not dest.exists()


def test_rejects_terminal_state_race(case, monkeypatch):
    _, _, dest = case
    states = iter([{'ActiveState': 'failed'}, {'ActiveState': 'active'}])
    monkeypatch.setattr(snapshot, 'terminal_state', lambda: next(states))
    with pytest.raises(ValueError, match='changed'):
        snapshot.snapshot(dest)
    assert not dest.exists()
