import importlib.util
import json
from pathlib import Path
import shutil

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v3_gate', REPO / 'scripts/snapshot_p8_capture_v3_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
spec2 = importlib.util.spec_from_file_location('prior_test_fixture', REPO / 'tests/test_p8_capture_v2_stop.py')
oldtests = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(oldtests)
case = oldtests.case


@pytest.fixture
def campaign(case, tmp_path, monkeypatch):
    oldroot, _ = case
    old = oldtests.stop
    root, plan, dest = tmp_path / 'v3', tmp_path / 'v3-plan.json', tmp_path / 'v3-evidence'
    root.mkdir()
    monkeypatch.setattr(gate, 'prior', old)
    plan.write_text(json.dumps({'output': str(root), 'capture_image': old.IMAGE}))
    digest = old.receipt(plan)['sha256']
    plan.with_suffix('.sha256').write_text(digest)
    stages = []
    # Valid typed thermal rows, then rebuild original immutable chain.
    oldslot = oldroot / 'canary-n128'
    a = json.loads((oldslot / 'execution.json').read_text())
    for name in ('thermal.jsonl', 'cooldown.jsonl'):
        (oldslot / name).write_text(json.dumps({'at': '2026-09-05T09:00:00+00:00', 'temperatures': [60] * 4}) + '\n')
        a['files'][name] = old.receipt(oldslot / name)
    (oldslot / 'execution.json').write_text(json.dumps(a))
    oldexec = json.loads((oldroot / 'execution.json').read_text())
    oldexec['stages'][0]['execution_sha256'] = old.receipt(oldslot / 'execution.json')['sha256']
    (oldroot / 'execution.json').write_text(json.dumps(oldexec))
    monkeypatch.setattr(old, 'ROOT_SHA', old.receipt(oldroot / 'execution.json')['sha256'])
    for name in ('canary-n128', 'canary-n64'):
        slot = root / name
        shutil.copytree(oldslot, slot)
        stage = json.loads((slot / 'execution.json').read_text())
        stage['plan_sha256'] = digest
        (slot / 'execution.json').write_text(json.dumps(stage))
        stages.append({'slot': name, 'execution_sha256': old.receipt(slot / 'execution.json')['sha256']})
    execution = {**oldexec, 'plan_sha256': digest, 'stages': stages}
    (root / 'execution.json').write_text(json.dumps(execution))
    (root / 'canary-exact.json').write_text(json.dumps({'all_exact': False}))
    monkeypatch.setattr(gate, 'SOURCE', root)
    monkeypatch.setattr(gate, 'PLAN', plan)
    monkeypatch.setattr(gate, 'PLAN_SHA', digest)
    monkeypatch.setattr(gate, 'ROOT_SHA', old.receipt(root / 'execution.json')['sha256'])
    monkeypatch.setattr(gate, 'terminal_state', lambda: {'ActiveState': 'failed'})
    diagnostic = {'primary_gate_replayed': True, 'gate_passed': False, 'comparisons': {
        name: {'differing_rows': [259], 'maximum_absolute_difference': 7.0, 'unequal_values': 4}
        for name in ('v3_n128_vs_v3_n64', 'v2_n128_vs_v3_n128', 'v2_n128_vs_v3_n64')}}
    monkeypatch.setattr(gate, 'compare_verified', lambda: diagnostic)
    return root, dest, diagnostic


def test_preserves_failure_and_diagnostics_without_logits(campaign):
    root, dest, diagnostic = campaign
    result = gate.snapshot(dest)
    assert result['status'] == 'v3-failed-gate-preserved'
    assert (dest / 'canary-exact.json').read_bytes() == (root / 'canary-exact.json').read_bytes()
    report = json.loads((dest / 'report.json').read_text())
    assert report['paired_gate_passed'] is False and report['full_stage_started'] is False
    assert report['kld_measured'] is False and report['allocation_restart'] is False
    assert len(report['raw_logits']) == 3 and 'post-hoc' in report['repeat_comparison_role']
    assert not list(dest.rglob('*.f32')) and not list(dest.rglob('*.private.*'))
    for path in dest.rglob('*'):
        if path.is_file():
            assert b'PRIVATE_SECRET' not in path.read_bytes()
    manifest = json.loads((dest / 'snapshot.json').read_text())
    for name, expected in manifest['outputs'].items():
        assert gate.prior.receipt(dest / name) == expected


@pytest.mark.parametrize('name', ['execution.json', 'canary-n64/execution.json',
                                'canary-n128/' + oldtests.stop.RAW_NAME,
                                'canary-n64/requests/unreceipted.json', 'full-exact.json'])
def test_contradictory_or_changed_evidence_rejected(campaign, name):
    root, dest, _ = campaign
    (root / name).write_text('tampered')
    with pytest.raises(ValueError):
        gate.snapshot(dest)
    assert not dest.exists()


def test_cannot_substitute_passing_reanalysis(campaign):
    _, dest, diagnostic = campaign
    diagnostic['gate_passed'] = True
    with pytest.raises(ValueError, match='failed primary gate'):
        gate.snapshot(dest)
    assert not dest.exists()


def test_fresh_output_required(campaign):
    _, dest, _ = campaign
    dest.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        gate.snapshot(dest)


def test_terminal_unit_required(monkeypatch):
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *a, **k: 'ActiveState=active\nResult=success\nExecMainStatus=0\n')
    with pytest.raises(ValueError, match='terminal'):
        gate.terminal_state()
