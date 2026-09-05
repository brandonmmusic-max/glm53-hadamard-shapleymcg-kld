import importlib.util
import json
from pathlib import Path

import pytest

from glm53_nvfp4 import p8_fc1_integration as integration

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_fc1', REPO / 'scripts/snapshot_p8_fc1_integration.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def proof_log():
    return '\n'.join([
        'secret=DO_NOT_PUBLISH enforce_eager=False secret=DO_NOT_PUBLISH',
        'Breakable CUDA graph enabled secret=DO_NOT_PUBLISH',
        'Capturing CUDA graphs (FULL): 100% secret=DO_NOT_PUBLISH',
        *[f'(Worker_TP{rank} pid=12) Graph capturing finished secret=DO_NOT_PUBLISH' for rank in range(4)],
        *[f'GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} private=DO_NOT_PUBLISH small_m_scheduler=true'
          for layer in range(3, 45) for rank in range(4)],
        *[f'GLM53_P8_M1_DISPATCH layer={layer} rank={rank} fc1_tile_n=64 fused_scratch_zero=true secret=DO_NOT_PUBLISH'
          for layer in range(3, 45) for rank in range(4)],
        'Entirely private DO_NOT_PUBLISH error stack',
    ])


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    source = root / 'fc1-integrated-v1'
    source.mkdir(parents=True)
    dest = tmp_path / 'evidence' / 'p8-fc1-integrated-v1'
    for key, value in (('ROOT', root), ('SOURCE', source), ('DEST', dest)):
        monkeypatch.setattr(snapshot, key, value)
    plan_path = tmp_path / 'plan.json'
    plan = {'candidate_image': 'sha256:' + 'a' * 64, 'tile_n': 64, 'fused_scratch_zero': True}
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(snapshot, 'authenticate', lambda path: (plan, {}))
    state = {'ActiveState': 'inactive', 'Result': 'success'}
    monkeypatch.setattr(snapshot, 'terminal_state', lambda: state.copy())
    result = json.loads((REPO / 'evidence/opened/codec-v2/p8-smallm-integrated-v1/result.json').read_text())
    result['metadata'].update(model=integration.NAME, server=f'127.0.0.1:{integration.PORT}')
    files = {'result.json': json.dumps(result), 'summary.json': json.dumps(integration.summarize(result)),
             'runtime-audit.json': json.dumps(integration.verify_runtime(proof_log(), 64, True)),
             'thermal.jsonl': '{"temperatures":[60,60,60,60]}\n',
             'device-prerequisite.json': '{}', 'models.json': '{}',
             'server-ready.log': proof_log(), 'server-final.log': proof_log(),
             'launch-spec.private.json': '{"secret":"DO_NOT_PUBLISH"}',
             'container.private.json': '{"secret":"DO_NOT_PUBLISH"}',
             'unknown.log': 'DO_NOT_PUBLISH'}
    for name, data in files.items():
        (source / name).write_text(data)
    execution = {'schema': 'glm53-p8-fc1-integration-execution.v1',
                 'plan_sha256': snapshot.digest(plan_path.read_bytes()), 'image': plan['candidate_image'],
                 'tile_n': 64, 'fused_scratch_zero': True, 'protected_roles_opened': [],
                 'allocation_restart': False, 'single_run_diagnostic': True,
                 'five_cold_run_product_comparison': False, 'exit_code': 0, 'finished_at': 'terminal',
                 'cleanup': {'ok': True}, 'prior': {'backend': True, 'timer': True},
                 'restoration': {'backend': True, 'timer': True, 'errors': []}}

    def refresh():
        execution['files'] = {path.name: {'sha256': snapshot.digest(path.read_bytes()), 'bytes': path.stat().st_size}
                              for path in source.iterdir() if path.name != 'execution.json'}
        (source / 'execution.json').write_text(json.dumps(execution))

    refresh()
    return source, dest, plan_path, execution, state, refresh


def test_log_sanitizer_preserves_proofs_not_arbitrary_text():
    safe = snapshot.sanitize_log(proof_log()).decode()
    assert 'DO_NOT_PUBLISH' not in safe and 'pid=' not in safe
    assert integration.verify_runtime(safe, 64, True)['m1_pairs'] == 168


def test_success_snapshot_authenticates_and_omits_private_material(source):
    raw, dest, plan, _, _, _ = source
    before = {path.name: path.read_bytes() for path in raw.iterdir()}
    manifest = snapshot.snapshot(plan)
    assert manifest['status'] == 'terminal-success-diagnostic'
    assert not manifest['allocation_restart'] and not manifest['five_cold_run_product_comparison']
    assert (dest / 'result.json').read_bytes() == (raw / 'result.json').read_bytes()
    assert not (dest / 'server-final.log').exists()
    assert not (dest / 'launch-spec.private.json').exists()
    assert not (dest / 'container.private.json').exists()
    assert not (dest / 'unknown.log').exists()
    assert all(b'DO_NOT_PUBLISH' not in path.read_bytes() for path in dest.iterdir())
    assert before == {path.name: path.read_bytes() for path in raw.iterdir()}
    final = next(row for row in manifest['files'] if row['path'] == 'server-final.sanitized.log')
    assert final['raw_sha256'] == snapshot.digest((raw / 'server-final.log').read_bytes())
    with pytest.raises(ValueError, match='no overwrite'):
        snapshot.snapshot(plan)


def test_failure_snapshot_preserves_incomplete_attempt(source):
    raw, dest, plan, execution, state, refresh = source
    execution['exit_code'] = 1
    execution['cleanup'] = {'ok': False}
    state.update(ActiveState='failed', Result='exit-code')
    for name in ('summary.json', 'result.json', 'runtime-audit.json', 'server-final.log'):
        (raw / name).unlink()
    refresh()
    manifest = snapshot.snapshot(plan)
    assert manifest['status'] == 'terminal-failure-preserved'
    assert {'summary.json', 'result.json', 'runtime-audit.json', 'server-final.log'} <= set(manifest['missing_artifacts'])
    assert json.loads((dest / 'execution.json').read_text())['cleanup']['ok'] is False


def test_source_hash_mismatch_prevents_any_snapshot_write(source):
    raw, dest, plan, _, _, _ = source
    (raw / 'thermal.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='raw artifact identity'):
        snapshot.snapshot(plan)
    assert not dest.exists()


def test_unreviewed_benchmark_environment_is_not_published(source):
    raw, dest, plan, _, _, refresh = source
    result = json.loads((raw / 'result.json').read_text())
    result['startup_diagnostics']['env'] = {'VLLM_API_KEY': 'DO_NOT_PUBLISH'}
    (raw / 'result.json').write_text(json.dumps(result))
    refresh()
    with pytest.raises(ValueError, match='privacy review'):
        snapshot.snapshot(plan)
    assert not dest.exists()


def test_wrong_plan_identity_and_missing_success_artifact_fail(source):
    raw, dest, plan, execution, _, refresh = source
    execution['tile_n'] = 32
    refresh()
    with pytest.raises(ValueError, match='identity'):
        snapshot.snapshot(plan)
    execution['tile_n'] = 64
    (raw / 'summary.json').unlink()
    refresh()
    with pytest.raises(ValueError, match='missing required'):
        snapshot.snapshot(plan)
    assert not dest.exists()


def test_active_unit_is_rejected(monkeypatch):
    class Result:
        stdout = 'ActiveState=active\nResult=success\n'

    monkeypatch.setattr(snapshot.subprocess, 'run', lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match='terminal'):
        snapshot.terminal_state()
