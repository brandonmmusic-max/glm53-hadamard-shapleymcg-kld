import json
from pathlib import Path

import pytest

from scripts import snapshot_p8_index_trace_failure as snap


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo, raw = tmp_path / 'repo', tmp_path / 'raw'
    repo.mkdir()
    raw.mkdir()
    slot = raw / snap.SLOT
    slot.mkdir()
    for name in ('requests', 'captures', 'index-traces'):
        (slot / name).mkdir()
    loader = repo / 'runtime_patch/p8_index_trace/__init__.py'
    loader.parent.mkdir(parents=True)
    loader.write_text("from __future__ import annotations\ncompile(source, self.origin, 'exec')\n")
    plan = repo / 'plan.json'
    plan.write_bytes(snap.encoded({'output': str(raw), 'capture_image': snap.IMAGE,
        'source_sha256': {str(loader.relative_to(repo)): snap.sha(loader.read_bytes())}}))
    plan_sha = snap.sha(plan.read_bytes())
    plan.with_suffix('.sha256').write_text(plan_sha + '  plan.json\n')
    for name in snap.FILES:
        (slot / name).write_text('PRIVATE-SECRET-NOT-PUBLISHABLE')
    for name in ('cooldown.jsonl', 'thermal.jsonl'):
        (slot / name).write_text(json.dumps({'at': '2026-09-05T08:51:00+00:00',
            'temperatures': [40, 41, 42, 43], 'private': 'PRIVATE-SECRET-NOT-PUBLISHABLE'}) + '\n')
    (slot / 'server-final.private.log').write_text('\n'.join((*snap.ERRORS,
        'torch/_library/infer_schema.py', 'PRIVATE-SECRET-NOT-PUBLISHABLE')))
    (slot / 'container-final.private.json').write_bytes(snap.encoded({
        'Id': 'a' * 64, 'Image': snap.IMAGE, 'Name': '/' + snap.PREFIX + '-repeat-01-canary-n128',
        'State': {'Running': False, 'ExitCode': 1}, 'Env': ['PRIVATE-SECRET-NOT-PUBLISHABLE']}))
    stage = {'exit_code': 1, 'windows': [], 'plan_sha256': plan_sha, 'capture_image': snap.IMAGE,
             'cleanup': {'ok': True, 'errors': []}, 'protected_roles_opened': [],
             'unreadable_artifacts': [], 'arm': 'n128', 'container_id': 'a' * 64,
             'files': {name: {'bytes': (slot / name).stat().st_size,
                             'sha256': snap.sha((slot / name).read_bytes())} for name in snap.FILES}}
    (slot / 'execution.json').write_bytes(snap.encoded(stage))
    stage_sha = snap.sha((slot / 'execution.json').read_bytes())
    root = {'exit_code': 1, 'plan_sha256': plan_sha,
            'repeats': [{'index': 1, 'execution_sha256': stage_sha}],
            'prior': {'backend': True, 'timer': False},
            'restoration': {'backend': True, 'timer': False, 'errors': []},
            'restoration_safety': {'ok': True, 'errors': [], 'containers': []},
            'final_identity_audit': {'ok': True}, 'protected_roles_opened': [],
            'teacher_logits_opened': False, 'speed_measurement_valid': False, 'allocation_restart': False}
    (raw / 'execution.json').write_bytes(snap.encoded(root))
    for name in ('error.private.txt', 'hardware-inventory.json'):
        (raw / name).write_text('PRIVATE-SECRET-NOT-PUBLISHABLE')
    for key, value in {'REPO': repo, 'SOURCE': raw, 'PLAN': plan, 'PLAN_SHA': plan_sha,
                       'STAGE_SHA': stage_sha, 'ROOT_SHA': snap.sha((raw / 'execution.json').read_bytes())}.items():
        monkeypatch.setattr(snap, key, value)
    monkeypatch.setattr(snap, 'terminal_state', lambda: {'ActiveState': 'failed', 'MainPID': '0'})
    return raw, tmp_path / 'snapshot'


def test_preserves_failure_without_private_content_or_overwrite(fixture):
    raw, out = fixture
    assert snap.snapshot(out)['files'] == 7
    report = json.loads((out / 'report.json').read_text())
    assert report['evaluation_requests'] == 0 and report['stage_files_verified'] == 11
    assert report['numerical_closure_passed'] is False
    for path in out.rglob('*'):
        if path.is_file():
            assert b'PRIVATE-SECRET-NOT-PUBLISHABLE' not in path.read_bytes()
    manifest = json.loads((out / 'snapshot.json').read_text())
    assert str(raw / snap.SLOT / 'server-final.private.log') in manifest['inputs']
    with pytest.raises(ValueError, match='no overwrite'):
        snap.snapshot(out)


@pytest.mark.parametrize('directory', ['requests', 'captures', 'index-traces'])
def test_refuses_any_evaluation_artifact(fixture, directory):
    raw, out = fixture
    (raw / snap.SLOT / directory / 'unexpected').write_text('x')
    with pytest.raises(ValueError, match='artifacts exist'):
        snap.snapshot(out)
    assert not out.exists()


def test_refuses_hash_drift_before_output(fixture):
    raw, out = fixture
    (raw / snap.SLOT / 'launch-spec.private.json').write_text('changed')
    with pytest.raises(ValueError, match='raw artifact identity'):
        snap.snapshot(out)
    assert not out.exists()


def test_refuses_nonterminal_unit_or_remaining_container(monkeypatch):
    monkeypatch.setattr(snap.subprocess, 'check_output', lambda argv, **k:
        'ActiveState=failed\nResult=exit-code\nMainPID=0\n' if argv[0] == 'systemctl' else 'live-id\n')
    with pytest.raises(ValueError, match='container remains'):
        snap.terminal_state()


def test_refuses_symlink_input(fixture):
    raw, out = fixture
    source = raw / snap.SLOT / 'launch-spec.private.json'
    target = raw / 'outside'
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(ValueError):
        snap.snapshot(out)
    assert not out.exists()
