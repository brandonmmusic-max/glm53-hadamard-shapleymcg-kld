"""CPU-only tests for the analysis-only corrected-prefix amendment."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from glm53_nvfp4 import p8_index_order_full_analysis_v2 as analysis


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def test_source_inventory_contains_only_the_wrapper_and_its_tests():
    assert analysis.source_files() == {
        'glm53_nvfp4/p8_index_order_full_analysis_v2.py',
        'tests/test_p8_index_order_full_analysis_v2.py',
    }


@pytest.mark.parametrize('fail', [False, True])
def test_corrected_prefix_is_lock_scoped_and_restored(fail, monkeypatch):
    original = 'legacy-four-stage-prefix'
    monkeypatch.setattr(analysis.launcher.base, 'PREFIX', original)
    observed = []

    with pytest.raises(RuntimeError, match='sentinel') if fail else analysis.corrected_prefix():
        if fail:
            with analysis.corrected_prefix():
                observed.append(analysis.launcher.base.PREFIX)
                raise RuntimeError('sentinel')
        else:
            observed.append(analysis.launcher.base.PREFIX)

    assert observed == [analysis.launcher.PREFIX]
    assert analysis.launcher.base.PREFIX == original


def test_corrected_prefix_rejects_ambiguous_preexisting_patch(monkeypatch):
    monkeypatch.setattr(analysis.launcher.base, 'PREFIX', analysis.launcher.PREFIX)
    with pytest.raises(ValueError, match='preexisting'):
        with analysis.corrected_prefix():
            pass


def _capture_tree(tmp_path, monkeypatch):
    repo = tmp_path / 'capture-repo'
    root = tmp_path / 'capture-root'
    plan_path = repo / 'experiments/p8-index-order-full-v1.json'
    source_hashes = {}
    for number in range(56):
        relative = f'sources/source-{number:02d}.py'
        source = repo / relative
        save(source, f'frozen source {number}\n')
        source_hashes[relative] = analysis.sha(source)
    capture_plan = {'source_sha256': source_hashes}
    save(plan_path, capture_plan)
    plan_sha = analysis.sha(plan_path)
    save(plan_path.with_suffix('.sha256'), f'{plan_sha}  {plan_path.name}\n')
    execution = root / 'execution.json'
    exact_path = root / 'full-exact.json'
    save(execution, {'capture_protocol_complete': True})
    save(exact_path, {'all_exact': True})

    monkeypatch.setattr(analysis, 'CAPTURE_REPO', repo)
    monkeypatch.setattr(analysis, 'CAPTURE_PLAN', plan_path)
    monkeypatch.setattr(analysis, 'CAPTURE_PLAN_SHA256', plan_sha)
    monkeypatch.setattr(analysis, 'CAPTURE_ROOT', root)
    monkeypatch.setattr(analysis, 'CAPTURE_EXECUTION_SHA256', analysis.sha(execution))
    monkeypatch.setattr(analysis, 'FULL_EXACT_SHA256', analysis.sha(exact_path))
    failure_state = {'ActiveState': 'failed', 'MainPID': '0', 'Result': 'exit-code',
                     'ExecMainStatus': '1', 'InvocationID': analysis.FAILED_INVOCATION}
    unit_calls = []
    evidence_calls = []
    failed_evidence = {'journal_sha256': analysis.FAILED_JOURNAL_SHA256,
                       'analysis_output': '/absent/v1-output', 'analysis_output_absent': True,
                       'error': 'request does not match approved causal sequence',
                       'failure_before_output_creation': True}
    monkeypatch.setattr(
        analysis, 'unit_failure',
        lambda: unit_calls.append('checked') or copy.deepcopy(failure_state),
    )
    monkeypatch.setattr(
        analysis, 'failure_evidence',
        lambda: evidence_calls.append('checked') or copy.deepcopy(failed_evidence),
    )
    monkeypatch.setattr(analysis.launcher, 'authenticate', lambda path: capture_plan)
    return (capture_plan, plan_path, root, unit_calls, evidence_calls,
            failure_state, failed_evidence)


def test_capture_prerequisite_binds_old_receipts_and_corrects_only_verifier_prefix(
        tmp_path, monkeypatch):
    (capture_plan, plan_path, _, unit_calls, evidence_calls,
     failure_state, failed_evidence) = _capture_tree(tmp_path, monkeypatch)
    original = 'legacy-four-stage-prefix'
    monkeypatch.setattr(analysis.launcher.base, 'PREFIX', original)
    verifier_calls = []

    def verify(plan, path):
        verifier_calls.append((plan, path, analysis.launcher.base.PREFIX,
                               analysis.launcher.base.model_name('n128')))
        exact = {'all_exact': True, 'windows': [
            {'rows': 2047, 'vocabulary': 154880, 'exact': True}
            for _ in range(32)
        ]}
        return {'capture_protocol_complete': True}, exact

    monkeypatch.setattr(analysis.v1, 'verify_execution', verify)
    receipt = analysis.capture_prerequisite()

    assert verifier_calls == [(capture_plan, plan_path, analysis.launcher.PREFIX,
                               f'{analysis.launcher.PREFIX}-n128')]
    assert analysis.launcher.base.PREFIX == original
    assert unit_calls == ['checked']
    assert evidence_calls == ['checked']
    assert receipt == {
        'capture_plan_sha256': analysis.CAPTURE_PLAN_SHA256,
        'capture_execution_sha256': analysis.CAPTURE_EXECUTION_SHA256,
        'full_exact_sha256': analysis.FULL_EXACT_SHA256,
        'source_count': 56,
        'windows': 32,
        'causal_rows': 65504,
        'all_exact': True,
        'failed_unit': failure_state,
        'failed_evidence': failed_evidence,
        'teacher_scored_by_failed_attempt': False,
    }


@pytest.mark.parametrize('drift', ['plan', 'source', 'execution', 'exact', 'unit', 'closure'])
def test_capture_prerequisite_fails_closed_on_old_evidence_drift(tmp_path, monkeypatch, drift):
    capture_plan, _, root, _, _, _, _ = _capture_tree(tmp_path, monkeypatch)
    exact = {'all_exact': True, 'windows': [
        {'rows': 2047, 'vocabulary': 154880, 'exact': True} for _ in range(32)
    ]}
    monkeypatch.setattr(
        analysis.v1, 'verify_execution',
        lambda plan, path: ({'capture_protocol_complete': True}, exact),
    )
    if drift == 'plan':
        analysis.CAPTURE_PLAN.write_text(analysis.CAPTURE_PLAN.read_text() + '\n')
    elif drift == 'source':
        (analysis.CAPTURE_REPO / next(iter(capture_plan['source_sha256']))).write_text('changed')
    elif drift == 'execution':
        save(root / 'execution.json', {'changed': True})
    elif drift == 'exact':
        save(root / 'full-exact.json', {'changed': True})
    elif drift == 'unit':
        monkeypatch.setattr(
            analysis, 'unit_failure',
            lambda: (_ for _ in ()).throw(ValueError('original failure state differs')),
        )
    else:
        exact['windows'][0]['exact'] = False
    with pytest.raises(ValueError):
        analysis.capture_prerequisite()


def _failed_journal():
    return b'\n'.join((
        b'Traceback (most recent call last):',
        b'p8_index_order_full.py", line 570, in run_and_analyze',
        b'p8_index_order_full_analysis.py", line 168, in run',
        b'p8_index_order_full_analysis.py", line 139, in verify_execution',
        b'ValueError: request does not match approved causal sequence',
    )) + b'\n'


def test_failure_evidence_binds_invocation_journal_and_absent_v1_output(tmp_path, monkeypatch):
    journal = _failed_journal()
    failed_output = tmp_path / 'absent-v1-output'
    calls = []
    monkeypatch.setattr(analysis, 'FAILED_OUTPUT', failed_output)
    monkeypatch.setattr(
        analysis, 'FAILED_JOURNAL_SHA256', analysis.hashlib.sha256(journal).hexdigest(),
    )

    def command(argv):
        calls.append(argv)
        return journal

    receipt = analysis.failure_evidence(command)
    assert calls == [[
        'journalctl', '--user', f'_SYSTEMD_INVOCATION_ID={analysis.FAILED_INVOCATION}',
        '--no-pager', '-o', 'cat',
    ]]
    assert receipt == {
        'journal_sha256': analysis.FAILED_JOURNAL_SHA256,
        'analysis_output': str(failed_output),
        'analysis_output_absent': True,
        'error': 'request does not match approved causal sequence',
        'failure_before_output_creation': True,
    }


@pytest.mark.parametrize('drift', ['hash', 'content', 'output'])
def test_failure_evidence_rejects_journal_or_output_drift(tmp_path, monkeypatch, drift):
    journal = _failed_journal()
    failed_output = tmp_path / 'v1-output'
    monkeypatch.setattr(analysis, 'FAILED_OUTPUT', failed_output)
    if drift == 'hash':
        monkeypatch.setattr(analysis, 'FAILED_JOURNAL_SHA256', '0' * 64)
    elif drift == 'content':
        journal = b'unrelated failure\n'
        monkeypatch.setattr(
            analysis, 'FAILED_JOURNAL_SHA256', analysis.hashlib.sha256(journal).hexdigest(),
        )
    else:
        failed_output.mkdir()
        monkeypatch.setattr(
            analysis, 'FAILED_JOURNAL_SHA256', analysis.hashlib.sha256(journal).hexdigest(),
        )
    with pytest.raises(ValueError, match='journal differs|traceback differs|unexpectedly exists'):
        analysis.failure_evidence(lambda argv: journal)


def test_unit_failure_requires_exact_terminal_invocation():
    expected = {'ActiveState': 'failed', 'MainPID': '0', 'Result': 'exit-code',
                'ExecMainStatus': '1', 'InvocationID': analysis.FAILED_INVOCATION}
    calls = []

    def command(argv, *, text):
        calls.append((argv, text))
        return '\n'.join(f'{key}={value}' for key, value in reversed(expected.items())) + '\n'

    assert analysis.unit_failure(command) == expected
    assert calls == [([
        'systemctl', '--user', 'show', analysis.FAILED_UNIT,
        '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus',
        '-p', 'InvocationID',
    ], True)]

    bad = {**expected, 'InvocationID': 'foreign'}
    with pytest.raises(ValueError, match='failure state'):
        analysis.unit_failure(lambda *args, **kwargs: '\n'.join(
            f'{key}={value}' for key, value in bad.items()))


def _analysis_plan_tree(tmp_path, monkeypatch):
    repo = tmp_path / 'analysis-repo'
    for relative in analysis.source_files():
        save(repo / relative, f'pinned {relative}\n')
    output = tmp_path / 'analysis-output'
    prerequisite = {'capture': 'immutable', 'all_exact': True}
    monkeypatch.setattr(analysis, 'REPO', repo)
    monkeypatch.setattr(analysis, 'OUTPUT', output)
    monkeypatch.setattr(analysis, 'FIXED', {**copy.deepcopy(analysis.FIXED), 'output': str(output)})
    monkeypatch.setattr(analysis, 'capture_prerequisite', lambda: copy.deepcopy(prerequisite))
    monkeypatch.setattr(analysis.pilot, 'now', lambda: '2026-09-05T12:00:00+00:00')
    path = tmp_path / 'analysis-plan.json'
    return path, output, prerequisite


def test_make_plan_and_authenticate_bind_sources_prerequisite_and_seal(tmp_path, monkeypatch):
    path, _, prerequisite = _analysis_plan_tree(tmp_path, monkeypatch)
    plan = analysis.make_plan(path)
    assert analysis.authenticate(path) == plan
    assert plan['capture_prerequisite'] == prerequisite
    assert set(plan['source_sha256']) == analysis.source_files()

    original = path.read_text()
    path.write_text(original + '\n')
    with pytest.raises(ValueError, match='sealed'):
        analysis.authenticate(path)
    path.write_text(original)

    source = analysis.REPO / next(iter(analysis.source_files()))
    source.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        analysis.authenticate(path)


@pytest.mark.parametrize('drift', ['fixed', 'prerequisite'])
def test_authenticate_rejects_resealed_protocol_or_prerequisite_drift(
        tmp_path, monkeypatch, drift):
    path, _, _ = _analysis_plan_tree(tmp_path, monkeypatch)
    analysis.make_plan(path)
    plan = json.loads(path.read_text())
    if drift == 'fixed':
        plan['gpu_launch_authorized'] = True
    else:
        plan['capture_prerequisite'] = {'capture': 'substituted'}
    save(path, plan)
    save(path.with_suffix('.sha256'), f'{analysis.sha(path)}  {path.name}\n')
    with pytest.raises(ValueError, match='fixed analysis|capture prerequisite'):
        analysis.authenticate(path)


def _run_plan(tmp_path, monkeypatch):
    output = tmp_path / 'analysis-output'
    plan_path = tmp_path / 'plan.json'
    plan = {**copy.deepcopy(analysis.FIXED), 'output': str(output),
            'capture_prerequisite': {'capture': 'immutable'}}
    save(plan_path, plan)
    monkeypatch.setattr(analysis, 'OUTPUT', output)
    monkeypatch.setattr(analysis, 'authenticate', lambda path: plan)
    monkeypatch.setattr(analysis.pilot, 'now', lambda: '2026-09-05T12:00:00+00:00')
    original = 'legacy-four-stage-prefix'
    monkeypatch.setattr(analysis.launcher.base, 'PREFIX', original)

    def forbidden(*args, **kwargs):
        raise AssertionError('capture/GPU launcher must not run in analysis amendment')

    monkeypatch.setattr(analysis.launcher, 'run', forbidden)
    monkeypatch.setattr(analysis.launcher.base, 'run', forbidden)
    return plan_path, output, original


def test_run_scores_existing_capture_under_corrected_prefix_without_target_rerun(
        tmp_path, monkeypatch):
    plan_path, output, original = _run_plan(tmp_path, monkeypatch)
    calls = []

    def score(capture_plan, scored):
        calls.append((capture_plan, scored, analysis.launcher.base.PREFIX))
        save(scored / 'analysis.json', {'status': 'complete'})
        return {'schema': 'glm53-p8.index-order-full-kld.v1', 'status': 'complete',
                'windows': 32, 'causal_rows': 65504, 'exact_logits': True,
                'protected_roles_opened': [], 'reference_n128': {'mean_kld': 0.125}}

    monkeypatch.setattr(analysis.v1, 'run', score)
    record = analysis.run(plan_path)

    assert calls == [(analysis.CAPTURE_PLAN, output / 'scored', analysis.launcher.PREFIX)]
    assert analysis.launcher.base.PREFIX == original
    assert record['exit_code'] == 0
    assert record['gpu_launched'] is False
    assert record['windows_scored'] == 32
    assert record['mean_kld'] == 0.125
    assert record['analysis_sha256'] == analysis.sha(output / 'scored/analysis.json')
    assert record['files'] == {
        'scored/analysis.json': {
            'bytes': (output / 'scored/analysis.json').stat().st_size,
            'sha256': analysis.sha(output / 'scored/analysis.json'),
        }
    }
    assert json.loads((output / 'execution.json').read_text())['exit_code'] == 0

    with pytest.raises(ValueError, match='fresh fixed'):
        analysis.run(plan_path)
    assert len(calls) == 1


def test_run_restores_prefix_and_preserves_partial_evidence_on_scorer_failure(
        tmp_path, monkeypatch):
    plan_path, output, original = _run_plan(tmp_path, monkeypatch)
    calls = []

    def fail(capture_plan, scored):
        calls.append(analysis.launcher.base.PREFIX)
        save(scored / 'partial.json', {'windows_scored': 7})
        save(scored / 'analysis.failed.json', {'status': 'failed', 'windows_scored': 7})
        raise ArithmeticError('numerical failure')

    monkeypatch.setattr(analysis.v1, 'run', fail)
    with pytest.raises(RuntimeError, match='partial evidence preserved'):
        analysis.run(plan_path)

    assert calls == [analysis.launcher.PREFIX]
    assert analysis.launcher.base.PREFIX == original
    assert (output / 'error.private.txt').read_text() == 'numerical failure'
    assert json.loads((output / 'scored/partial.json').read_text()) == {'windows_scored': 7}
    receipt = json.loads((output / 'execution.json').read_text())
    assert receipt['exit_code'] == 1
    assert receipt['gpu_launched'] is False
    assert receipt['windows_scored'] == 7
    assert receipt['error_type'] == 'ArithmeticError'
    assert set(receipt['files']) == {
        'error.private.txt', 'scored/analysis.failed.json', 'scored/partial.json',
    }


@pytest.mark.parametrize('bad', [True, -1, 33, '7', None])
def test_preserved_windows_scored_requires_bounded_exact_integer(tmp_path, bad):
    scored = tmp_path / 'scored'
    save(scored / 'analysis.failed.json', {'windows_scored': bad})
    with pytest.raises(ValueError, match='malformed'):
        analysis.preserved_windows_scored(scored)


def test_preserved_windows_scored_rejects_symlink_and_allows_absence(tmp_path):
    scored = tmp_path / 'scored'
    scored.mkdir()
    assert analysis.preserved_windows_scored(scored) == 0
    target = tmp_path / 'foreign.json'
    save(target, {'windows_scored': 1})
    (scored / 'analysis.failed.json').symlink_to(target)
    with pytest.raises(ValueError, match='regular file'):
        analysis.preserved_windows_scored(scored)
