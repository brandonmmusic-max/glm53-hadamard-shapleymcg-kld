import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from glm53_nvfp4 import p8_index_order_full as full


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'capture_base_tests', REPO / 'tests/test_p8_decode_launcher.py')
base_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_tests)


def _environment(argv):
    return dict(argv[index + 1].split('=', 1)
                for index, value in enumerate(argv) if value == '--env')


def test_full_source_inventory_extends_the_authenticated_52_by_exactly_four():
    inherited = full.canary.source_files()
    assert len(inherited) == full.CANARY_SOURCE_COUNT == 52
    assert full.source_files() - inherited == {
        'glm53_nvfp4/p8_index_order_full.py',
        'tests/test_p8_index_order_full.py',
        'glm53_nvfp4/p8_index_order_full_analysis.py',
        'tests/test_p8_index_order_full_analysis.py',
    }
    assert len(full.source_files()) == 56


def test_adapter_is_full_only_sets_both_arms_and_restores_after_exceptions(tmp_path):
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities',
             'clone_argv', 'runtime_audit')
    before = {name: getattr(full.base, name) for name in names}
    windows = [{'id': f'conditional-fit-{number:04d}'} for number in range(32)]
    plan = {'windows': windows}

    for entry in full.ORDER:
        with pytest.raises(RuntimeError, match='sentinel'):
            with full.stage_adapter(plan, entry):
                assert full.base.PREFIX == full.PREFIX
                assert full.base.stage_windows(plan, 'full') == windows
                with pytest.raises(ValueError, match='only full32'):
                    full.base.stage_windows(plan, 'canary')
                argv = full.base.clone_argv(
                    base_tests.pilot_recipe(), full.IMAGE, entry, tmp_path,
                    windows, 'sealed-owner')
                env = _environment(argv)
                assert env['GLM53_P8_INDEX_ORDER'] == 'logical-short-v1'
                assert env['GLM53_P8_INDEX_ORDER_RECEIPT'] == '1'
                assert env['GLM53_P8_INDEX_TRACE'] == ''
                assert env['GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS'] == '2047'
                assert env['GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS'] == ','.join(
                    window['id'] for window in windows)
                assert env['GLM53_P8_FC1_TILE_N'] == (
                    '128' if entry['arm'] == 'n128' else '64')
                assert env['GLM53_P8_FUSED_SCRATCH'] == (
                    '' if entry['arm'] == 'n128' else '1')
                assert not any('/p8-index-traces' in value for value in argv)
                raise RuntimeError('sentinel')

    assert {name: getattr(full.base, name) for name in names} == before


@pytest.mark.parametrize('collision', ['observer_env', 'order_env', 'observer_mount'])
def test_clone_rejects_recipe_that_preloads_observer_or_index_correction(
        tmp_path, collision):
    recipe = base_tests.pilot_recipe()
    if collision == 'observer_env':
        recipe['Config']['Env'].append('GLM53_P8_INDEX_TRACE=1')
    elif collision == 'order_env':
        recipe['Config']['Env'].append('GLM53_P8_INDEX_ORDER=foreign')
    else:
        recipe['HostConfig']['Binds'].append('/tmp/traces:/p8-index-traces:rw')
    with pytest.raises(ValueError):
        full.clone_argv(recipe, full.IMAGE, full.ORDER[0], tmp_path, [], 'owner')


def test_approved_inputs_requires_teacher_byte_verification(tmp_path, monkeypatch):
    roles = tmp_path / 'roles.json'
    teacher = tmp_path / 'teacher'
    token = tmp_path / 'tokens.npy'
    teacher_file = teacher / 'window.bf16'
    teacher.mkdir()
    for path in (roles, token, teacher_file):
        path.write_bytes(b'pinned')
    calls = []
    windows = [{'token_path': str(token), 'teacher_path': teacher_file.name}]

    def load(actual_roles, actual_teacher, *, verify_teacher_bytes):
        calls.append((actual_roles, actual_teacher, verify_teacher_bytes))
        return windows

    monkeypatch.setattr(full.protocol, 'load_role_inputs', load)
    observed, stats = full.approved_inputs(roles, teacher)
    assert observed == windows
    assert calls == [(roles, teacher, True)]
    assert set(stats) == {str(roles), str(token), str(teacher_file)}


def _make_isolated_plan(tmp_path, monkeypatch):
    repo, root = tmp_path / 'repo', tmp_path / 'raw'
    repo.mkdir()
    root.mkdir()
    source = repo / 'source.py'
    source.write_text('pinned source')
    roles = tmp_path / 'roles.json'
    roles.write_text('pinned roles')
    teacher = tmp_path / 'teacher'
    teacher.mkdir()
    canary_plan = tmp_path / 'canary-plan.json'
    image_receipt = tmp_path / 'image.json'
    image_receipt.write_text('pinned image receipt')
    canary_plan.write_text(json.dumps({
        'image_receipt': str(image_receipt),
        'image_receipt_sha256': full.pilot.sha(image_receipt),
    }))
    receipts = {'external': 'authenticated'}
    windows = [{'id': f'conditional-fit-{number:04d}',
                'token_path': str(tmp_path / f'token-{number:04d}.npy'),
                'teacher_path': f'teacher-{number:04d}.bf16'}
               for number in range(32)]
    stats = {'approved': {'sha256': 'b' * 64}}
    fixed = {**copy.deepcopy(full.FIXED),
             'analysis_output': str(root / 'index-order-full-v1-kld')}
    monkeypatch.setattr(full, 'REPO', repo)
    monkeypatch.setattr(full, 'ROOT', root)
    monkeypatch.setattr(full, 'FIXED', fixed)
    monkeypatch.setattr(full, 'CANARY_PLAN', canary_plan)
    monkeypatch.setattr(full, 'source_files', lambda: {'source.py'})
    monkeypatch.setattr(full.protocol, 'ROLES_SHA256', full.pilot.sha(roles))
    monkeypatch.setattr(full, 'replay_canary', lambda: receipts)
    monkeypatch.setattr(full, 'approved_inputs', lambda r, t: (windows, stats))
    monkeypatch.setattr(full, 'verify_input_stats', lambda plan: None)
    monkeypatch.setattr(full, 'verify_canary_receipts', lambda plan: None)
    plan_path = tmp_path / 'plan.json'
    output = root / 'index-order-full-v1'
    plan = full.make_plan(plan_path, output, roles, teacher)
    return plan_path, plan, source


def test_plan_authentication_is_sealed_and_fails_closed_on_source_or_protocol_drift(
        tmp_path, monkeypatch):
    path, plan, source = _make_isolated_plan(tmp_path, monkeypatch)
    assert full.authenticate(path) == plan
    assert plan['order'] == full.ORDER
    assert plan['independent_processes'] == 2
    assert plan['raw_capture_bytes'] == 2 * 32 * 2047 * 154880 * 4
    assert plan['observer_enabled'] is False
    assert plan['canary_receipts'] == {'external': 'authenticated'}

    original = path.read_text()
    path.write_text(original + '\n')
    with pytest.raises(ValueError, match='seal'):
        full.authenticate(path)
    path.write_text(original)

    for key, bad in (
            ('order', list(reversed(full.ORDER))),
            ('independent_processes', 3),
            ('observer_enabled', True),
            ('raw_capture_bytes', 1),
            ('full_windows', 31),
            ('analysis_output', str(tmp_path / 'foreign-analysis'))):
        with pytest.raises(ValueError, match='fixed full32'):
            full.verify_identities({**plan, key: bad})

    source.write_text('source drift')
    with pytest.raises(ValueError, match='source identity'):
        full.verify_identities(plan)


def test_planning_and_authentication_each_require_external_canary_replay(
        tmp_path, monkeypatch):
    calls = []
    path, plan, _ = _make_isolated_plan(tmp_path, monkeypatch)

    def replay():
        calls.append('replay')
        raise ValueError('external canary no longer authenticates')

    monkeypatch.setattr(full, 'replay_canary', replay)
    with pytest.raises(ValueError, match='no longer authenticates'):
        full.authenticate(path)
    assert calls == ['replay']


def test_replay_canary_rejects_nonpassing_terminal_execution(tmp_path, monkeypatch):
    repo, root = tmp_path / 'canary-repo', tmp_path / 'canary-root'
    repo.mkdir()
    root.mkdir()
    source = repo / 'source.py'
    source.write_text('pinned')
    plan_path = repo / 'plan.json'
    plan_sha = '1' * 64
    plan = {'schema': full.canary.FIXED['schema'], 'output': str(root),
            'source_sha256': {'source.py': full.pilot.sha(source)}}
    plan_path.write_text(json.dumps(plan))
    plan_path.with_suffix('.sha256').write_text(plan_sha + '  plan.json\n')
    execution = {
        'schema': 'glm53-p8.index-order-integration-execution.v1',
        'plan_sha256': plan_sha, 'exit_code': 0,
        'canary_exact_gate_passed': False,
        'full_model_kld_measured': False, 'full_panel_authorized': False,
        'teacher_logits_opened': False, 'protected_roles_opened': [],
        'observer_enabled': False, 'speed_measurement_valid': False,
        'allocation_restart': False, 'final_identity_audit': {'ok': True},
        'restoration_safety': {'ok': True, 'errors': [], 'containers': []},
        'prior': {'backend': False, 'timer': False},
        'restoration': {'backend': False, 'timer': False, 'errors': []},
        'stages': full.canary.ORDER,
    }
    execution_path = root / 'execution.json'
    execution_path.write_text(json.dumps(execution))
    monkeypatch.setattr(full, 'CANARY_REPO', repo)
    monkeypatch.setattr(full, 'CANARY_ROOT', root)
    monkeypatch.setattr(full, 'CANARY_PLAN', plan_path)
    monkeypatch.setattr(full, 'CANARY_PLAN_SHA256', plan_sha)
    monkeypatch.setattr(full, 'CANARY_EXECUTION_SHA256', '2' * 64)
    monkeypatch.setattr(full, 'CANARY_SOURCE_COUNT', 1)
    monkeypatch.setattr(full, '_unit_terminal', lambda: None)
    monkeypatch.setattr(full, '_small_file', lambda path, expected=None: expected or '3' * 64)
    monkeypatch.setattr(full.canary, 'source_files', lambda: {'source.py'})
    monkeypatch.setattr(full.canary, 'authenticate', lambda path: plan)

    with pytest.raises(ValueError, match='terminal execution contract'):
        full.replay_canary()


def test_stage_boundary_rejects_any_external_canary_receipt_drift(monkeypatch):
    monkeypatch.setattr(full, '_unit_terminal', lambda: None)
    monkeypatch.setattr(full, '_small_file', lambda path, expected=None: expected or 'f' * 64)
    with pytest.raises(ValueError, match='receipt changed after replay'):
        full.verify_canary_receipts({'canary_receipts': {}})


@pytest.mark.parametrize('mutation', ['count', 'order', 'rows', 'vocabulary', 'exact_type'])
def test_full_comparison_requires_all_32_declared_full_rows(monkeypatch, mutation):
    windows = [{'id': f'conditional-fit-{number:04d}'} for number in range(32)]
    rows = [{'window_id': window['id'], 'rows': 2047,
             'vocabulary': 154880, 'exact': True} for window in windows]
    if mutation == 'count':
        rows.pop()
    elif mutation == 'order':
        rows.reverse()
    elif mutation == 'rows':
        rows[0]['rows'] = 2046
    elif mutation == 'vocabulary':
        rows[0]['vocabulary'] = 154879
    else:
        rows[0]['exact'] = 1
    calls = []
    monkeypatch.setattr(
        full.base, 'compare_stage',
        lambda plan, root, stage: calls.append((root, stage)) or {
            'all_exact': False, 'windows': rows})
    plan = {'output': '/canonical/mock-output', 'windows': windows}
    with pytest.raises(ValueError, match='every declared causal row'):
        full.compare_full(plan)
    assert calls == [(Path(plan['output']), 'full')]


def _run_lifecycle(tmp_path, monkeypatch, failure='none', all_exact=True):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    output = tmp_path / 'raw' / 'index-order-full-v1'
    output.parent.mkdir()
    analysis_output = tmp_path / 'raw' / 'index-order-full-v1-kld'
    plan = {'output': str(output), 'analysis_output': str(analysis_output),
            'canary_receipts': {}}
    fixed = {**copy.deepcopy(full.FIXED), 'analysis_output': str(analysis_output)}
    monkeypatch.setattr(full, 'FIXED', fixed)
    monkeypatch.setattr(full, 'authenticate', lambda value: plan)
    monkeypatch.setattr(full, 'prelaunch', lambda value: base_tests.pilot_recipe())
    monkeypatch.setattr(full, 'verify_identities',
                        lambda value: (_ for _ in ()).throw(ValueError('identity drift'))
                        if failure == 'identity' else None)
    monkeypatch.setattr(full, 'open',
                        lambda value, mode: (tmp_path / 'model-stack.lock').open(mode),
                        raising=False)
    monkeypatch.setattr(full.fcntl, 'flock', lambda *args: None)
    monkeypatch.setattr(full.pilot, 'active', lambda *args: False)
    monkeypatch.setattr(full.cold, 'inventory', lambda: {'gpu': 'mocked'})
    commands = []

    def command(argv):
        commands.append(argv)
        stdout = 'deadbeef\n' if argv[:4] == ['git', '-C', str(full.REPO), 'rev-parse'] else ''
        return subprocess.CompletedProcess(argv, 0, stdout, '')

    monkeypatch.setattr(full.pilot, 'command', command)
    stages = []

    def capture(plan, entry, recipe, directory, digest, hardware):
        stages.append(entry['arm'])
        directory.mkdir()
        (directory / 'execution.json').write_text('{}')
        number = len(stages)
        return {'exit_code': int(failure == f'stage{number}'),
                'container_id': 'same' if failure == 'duplicate' else str(number)}

    monkeypatch.setattr(full.base, 'capture_stage', capture)
    comparisons = []

    def compare(plan):
        comparisons.append('full')
        if failure == 'compare':
            raise RuntimeError('comparison failed')
        return {'all_exact': all_exact, 'windows': []}

    monkeypatch.setattr(full, 'compare_full', compare)
    monkeypatch.setattr(full, 'restoration_safety',
                        lambda digest: {'ok': True, 'errors': [], 'containers': []})
    restores = []

    def restore(prior, safety):
        for sig in full.base.INTERRUPT_SIGNALS:
            assert signal.getsignal(sig) == signal.SIG_IGN
            os.kill(os.getpid(), sig)
        restores.append((prior, safety))
        if failure == 'restore':
            raise RuntimeError('restore failed')
        return {**prior, 'errors': []}

    monkeypatch.setattr(full.cold, 'restore_if_safe', restore)
    before = {sig: signal.getsignal(sig) for sig in full.base.INTERRUPT_SIGNALS}
    return path, output, stages, comparisons, restores, before


@pytest.mark.parametrize(
    'failure,expected_stages,comparison_count',
    [('stage1', ['n128'], 0), ('stage2', ['n128', 'n64'], 0),
     ('duplicate', ['n128', 'n64'], 0), ('compare', ['n128', 'n64'], 1),
     ('identity', ['n128', 'n64'], 1), ('restore', ['n128', 'n64'], 1)],
)
def test_two_stage_lifecycle_stops_on_failure_and_always_restores_signals(
        tmp_path, monkeypatch, failure, expected_stages, comparison_count):
    path, output, stages, comparisons, restores, before = _run_lifecycle(
        tmp_path, monkeypatch, failure)
    with pytest.raises(RuntimeError, match='partial evidence preserved'):
        full.run(path)
    assert stages == expected_stages
    assert len(comparisons) == comparison_count
    assert len(restores) == 1
    assert {sig: signal.getsignal(sig) for sig in before} == before
    record = json.loads((output / 'execution.json').read_text())
    assert record['exit_code'] == 1
    assert record['capture_protocol_complete'] is (failure in ('identity', 'restore'))


@pytest.mark.parametrize('all_exact', [True, False])
def test_two_stage_order_and_nonexact_full_are_completed_without_becoming_a_pass(
        tmp_path, monkeypatch, all_exact):
    path, output, stages, comparisons, restores, before = _run_lifecycle(
        tmp_path, monkeypatch, all_exact=all_exact)
    record = full.run(path)
    assert stages == ['n128', 'n64']
    assert comparisons == ['full']
    assert len(restores) == 1
    assert {sig: signal.getsignal(sig) for sig in before} == before
    assert record['exit_code'] == 0
    assert record['capture_protocol_complete'] is True
    assert record['full_exact'] is all_exact
    assert 'full_exact_gate_passed' not in record
    assert json.loads((output / 'full-exact.json').read_text())['all_exact'] is all_exact


def test_run_rejects_external_canary_auth_failure_before_any_lifecycle_action(
        tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    calls = []
    monkeypatch.setattr(full, 'authenticate',
                        lambda value: (_ for _ in ()).throw(ValueError('canary gate failed')))
    monkeypatch.setattr(full, 'prelaunch', lambda plan: calls.append('prelaunch'))
    with pytest.raises(ValueError, match='canary gate failed'):
        full.run(path)
    assert calls == []


def test_run_and_analyze_never_scores_after_capture_failure(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    calls = []
    from glm53_nvfp4 import p8_index_order_full_analysis as analysis
    monkeypatch.setattr(full, 'run',
                        lambda value: (_ for _ in ()).throw(RuntimeError('capture failed')))
    monkeypatch.setattr(analysis, 'run', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='capture failed'):
        full.run_and_analyze(path)
    assert calls == []


def test_run_and_analyze_scores_completed_nonexact_capture_at_fixed_path(
        tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    analysis_output = tmp_path / 'fixed-analysis'
    fixed = {**copy.deepcopy(full.FIXED), 'analysis_output': str(analysis_output)}
    capture = {'exit_code': 0, 'capture_protocol_complete': True, 'full_exact': False}
    calls = []
    monkeypatch.setattr(full, 'FIXED', fixed)
    monkeypatch.setattr(full, 'run', lambda value: capture)
    from glm53_nvfp4 import p8_index_order_full_analysis as analysis
    monkeypatch.setattr(analysis, 'run',
                        lambda plan, output: calls.append((plan, output)) or {'status': 'complete'})
    result = full.run_and_analyze(path)
    assert calls == [(path, analysis_output)]
    assert result == {'capture': capture, 'analysis': {'status': 'complete'}}
