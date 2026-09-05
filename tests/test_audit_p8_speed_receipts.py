import copy
import json

import pytest

from glm53_nvfp4 import audit_p8_speed_receipts as mod


def test_config_normalizes_docker_order_not_command_order():
    a = {'Config': {'Cmd': ['a', 'b'], 'Env': ['A=1', 'B=2']},
         'HostConfig': {'Binds': ['/a:/a:ro', '/b:/b:rw'], 'DeviceRequests': []}}
    b = copy.deepcopy(a)
    b['Config']['Env'].reverse()
    b['HostConfig']['Binds'].reverse()
    assert mod.config_fingerprint(a) == mod.config_fingerprint(b)
    b['Config']['Cmd'].reverse()
    assert mod.config_fingerprint(a) != mod.config_fingerprint(b)


def test_ambiguous_env_rejected():
    a = {'Config': {'Cmd': [], 'Env': ['A=1', 'A=2']},
         'HostConfig': {'Binds': [], 'DeviceRequests': []}}
    with pytest.raises(ValueError, match='ambiguous'):
        mod.config_fingerprint(a)


@pytest.fixture
def series(tmp_path, monkeypatch):
    orders = [['exl3', 'p8'], ['p8', 'exl3']]
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'benchmark': {
        'arm_order_by_round': orders, 'cold_process_runs_per_arm': 2}}))
    rows = {}
    for i, (round_no, arm) in enumerate((r, a) for r, order in enumerate(orders, 1) for a in order):
        slot = tmp_path / f'round-{round_no:02d}-{arm}'
        slot.mkdir()
        (slot / 'receipt.json').write_text('{}')
        rows[slot.name] = {
            'round': round_no, 'arm': arm, 'container_id': f'container-{i}',
            'started_at': f'2026-09-05T04:{i * 2:02d}:00Z',
            'completed_at': f'2026-09-05T04:{i * 2 + 1:02d}:00Z',
            'config_sha256': arm,
        }
    monkeypatch.setattr(mod, 'audit_slot', lambda slot, *args: copy.deepcopy(rows[slot.name]))
    return tmp_path, plan, rows


def test_complete_independent_ordered_runs(series):
    root, plan, _ = series
    result = mod.audit(root, plan)
    assert result['status'] == 'pass'
    assert len(result['runs']) == 4


def test_partial_inventory_is_not_pass(series):
    root, plan, _ = series
    (root / 'round-02-exl3/receipt.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        mod.audit(root, plan)
    assert mod.audit(root, plan, True)['status'] == 'incomplete'


def test_missing_predecessor_rejected(series):
    root, plan, _ = series
    (root / 'round-01-p8/receipt.json').unlink()
    with pytest.raises(ValueError, match='predecessor'):
        mod.audit(root, plan, True)


def test_duplicate_container_rejected(series):
    root, plan, rows = series
    rows['round-02-p8']['container_id'] = rows['round-01-p8']['container_id']
    with pytest.raises(ValueError, match='reused'):
        mod.audit(root, plan)


def test_configuration_drift_rejected(series):
    root, plan, rows = series
    rows['round-02-p8']['config_sha256'] = 'changed'
    with pytest.raises(ValueError, match='configuration changed'):
        mod.audit(root, plan)


def test_overlapping_runs_rejected(series):
    root, plan, rows = series
    rows['round-02-p8']['started_at'] = rows['round-01-p8']['started_at']
    with pytest.raises(ValueError, match='chronology'):
        mod.audit(root, plan)


def test_plan_requires_each_arm_once(series):
    root, plan, _ = series
    data = json.loads(plan.read_text())
    data['benchmark']['arm_order_by_round'][0] = ['p8', 'p8']
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='arm inventory'):
        mod.audit(root, plan)


def test_slot_checks_receipt_identity_before_reading_inputs(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'candidate': {}}))
    (tmp_path / 'receipt.json').write_text(json.dumps({'round': 1, 'arm': 'exl3'}))
    with pytest.raises(ValueError, match='slot identity'):
        mod.audit_slot(tmp_path, plan, 'p8', 1)


def test_slot_checks_protected_role_receipt(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'candidate': {}}))
    (tmp_path / 'receipt.json').write_text(json.dumps({
        'round': 1, 'arm': 'p8', 'protected_roles_opened': ['confirmation']}))
    with pytest.raises(ValueError, match='protected-role'):
        mod.audit_slot(tmp_path, plan, 'p8', 1)


def test_slot_checks_hash_before_parsing_data(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'candidate': {}}))
    (tmp_path / 'benchmark.json').write_text('{}')
    (tmp_path / 'receipt.json').write_text(json.dumps({
        'round': 1, 'arm': 'p8', 'protected_roles_opened': [],
        'files': {'benchmark.json': {'sha256': 'wrong', 'bytes': 2}}}))
    with pytest.raises(ValueError, match='hash/size'):
        mod.audit_slot(tmp_path, plan, 'p8', 1)
