"""Analysis-only amendment for the corrected-index full32 P8 captures.

V1 capture completed and passed exact closure. Its chained analyzer stopped
before teacher scoring because a verifier reconstructed request model names
after the launcher's scoped prefix had been restored. This amendment changes
only that verification context, reuses the frozen V1 metric and captures, and
records a new outer receipt. It never launches or controls a GPU service.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import threading

from . import p8_index_order_full as launcher
from . import p8_index_order_full_analysis as v1

pilot = launcher.pilot
REPO = Path(__file__).resolve().parents[1]
CAPTURE_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-full-v1')
CAPTURE_PLAN = CAPTURE_REPO / 'experiments/p8-index-order-full-v1.json'
CAPTURE_PLAN_SHA256 = '6873cd1c066ad2845635cdb0db25df2352196240930d99ec213dc16b6a15133a'
CAPTURE_ROOT = launcher.ROOT / 'index-order-full-v1'
CAPTURE_EXECUTION_SHA256 = '3a08163738655452c0c430a36ecb906e014b0bec7f5d8aeef80d70849ccac158'
FULL_EXACT_SHA256 = '89b628ec6519f3a2e070a2226e62140bbd9608e2f5ea57966a1c7e4ad1f53e10'
FAILED_UNIT = 'glm53-p8-index-order-full-v1.service'
FAILED_INVOCATION = 'c6509947a453419d900ffb39cee70b7a'
FAILED_OUTPUT = launcher.ROOT / 'index-order-full-v1-kld'
FAILED_JOURNAL_SHA256 = '423aa90e503e91455fcc35196414fc678ea4e38f531abfde47e43b91bec01d9d'
OUTPUT = launcher.ROOT / 'index-order-full-v1-kld-v2'
PREFIX_LOCK = threading.RLock()

FIXED = {
    'schema': 'glm53-p8.index-order-full-analysis-v2-plan.v1',
    'capture_plan': str(CAPTURE_PLAN),
    'capture_plan_sha256': CAPTURE_PLAN_SHA256,
    'capture_execution_sha256': CAPTURE_EXECUTION_SHA256,
    'full_exact_sha256': FULL_EXACT_SHA256,
    'failed_analysis_unit': FAILED_UNIT,
    'failed_analysis_invocation': FAILED_INVOCATION,
    'failed_analysis_journal_sha256': FAILED_JOURNAL_SHA256,
    'failed_analysis_error_type': 'ValueError',
    'failed_analysis_error': 'request does not match approved causal sequence',
    'failure_stage': 'receipt verification before teacher scoring',
    'failed_analysis_windows_scored': 0,
    'amendment': ('Reconstruct request model names under the full campaign prefix while reusing the '
                  'unchanged frozen V1 receipt verifier, FP64 metric, aggregation and output schema.'),
    'output': str(OUTPUT),
    'capture_reused': True,
    'gpu_launch_authorized': False,
    'teacher_role': 'conditional-fit',
    'protected_roles_opened': [],
    'metric': 'KL(teacher || student), nats, full vocabulary, CPU FP64',
    'windows': 32,
    'causal_rows': 65504,
    'bootstrap': {'unit': 'paired window', 'replicates': 20000, 'seed': 20260905},
    'decision': 'Bitwise closure remains primary; KLD cannot override closure. No matched NVFP4 claim.',
    'tail_limit': 'Incomplete-KPool-tail omission remains unchanged and unqualified.',
}


def source_files():
    return {
        'glm53_nvfp4/p8_index_order_full_analysis_v2.py',
        'tests/test_p8_index_order_full_analysis_v2.py',
    }


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unit_failure(command=subprocess.check_output):
    text = command(['systemctl', '--user', 'show', FAILED_UNIT, '-p', 'ActiveState', '-p', 'MainPID',
                    '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'InvocationID'], text=True)
    state = dict(line.split('=', 1) for line in text.strip().splitlines())
    expected = {'ActiveState': 'failed', 'MainPID': '0', 'Result': 'exit-code',
                'ExecMainStatus': '1', 'InvocationID': FAILED_INVOCATION}
    if state != expected:
        raise ValueError('original chained analysis failure state differs')
    return state


def failure_evidence(command=subprocess.check_output):
    """Authenticate the invocation-scoped traceback and pre-output failure stage."""
    journal = command([
        'journalctl', '--user', f'_SYSTEMD_INVOCATION_ID={FAILED_INVOCATION}',
        '--no-pager', '-o', 'cat',
    ])
    if hashlib.sha256(journal).hexdigest() != FAILED_JOURNAL_SHA256:
        raise ValueError('original chained analysis journal differs')
    text = journal.decode('utf-8', errors='strict')
    required = (
        'p8_index_order_full.py", line 570, in run_and_analyze',
        'p8_index_order_full_analysis.py", line 168, in run',
        'p8_index_order_full_analysis.py", line 139, in verify_execution',
        'ValueError: request does not match approved causal sequence',
    )
    if any(fragment not in text for fragment in required):
        raise ValueError('original chained analysis traceback differs')
    if FAILED_OUTPUT.exists():
        raise ValueError('failed V1 analysis output unexpectedly exists')
    return {'journal_sha256': FAILED_JOURNAL_SHA256,
            'analysis_output': str(FAILED_OUTPUT), 'analysis_output_absent': True,
            'error': 'request does not match approved causal sequence',
            'failure_before_output_creation': True}


@contextmanager
def corrected_prefix():
    """Narrow single-process compatibility context; always restore V1 global."""
    with PREFIX_LOCK:
        original = launcher.base.PREFIX
        if original == launcher.PREFIX:
            raise ValueError('unexpected preexisting full prefix')
        launcher.base.PREFIX = launcher.PREFIX
        try:
            yield
        finally:
            launcher.base.PREFIX = original


def capture_prerequisite():
    if (CAPTURE_PLAN != CAPTURE_PLAN.resolve() or sha(CAPTURE_PLAN) != CAPTURE_PLAN_SHA256
            or CAPTURE_PLAN.with_suffix('.sha256').read_text().split()
            != [CAPTURE_PLAN_SHA256, CAPTURE_PLAN.name]):
        raise ValueError('sealed V1 capture plan differs')
    capture_plan = json.loads(CAPTURE_PLAN.read_text())
    if len(capture_plan.get('source_sha256', {})) != 56:
        raise ValueError('V1 capture source inventory differs')
    for name, expected in capture_plan['source_sha256'].items():
        source = CAPTURE_REPO / name
        if source != source.resolve() or not source.is_relative_to(CAPTURE_REPO) or sha(source) != expected:
            raise ValueError('V1 frozen source changed')
    failed_unit = unit_failure()
    failed_evidence = failure_evidence()
    if (sha(CAPTURE_ROOT / 'execution.json') != CAPTURE_EXECUTION_SHA256
            or sha(CAPTURE_ROOT / 'full-exact.json') != FULL_EXACT_SHA256):
        raise ValueError('V1 capture or exact receipt differs')
    # Authenticate teacher/input identities and replay every stage and all-row
    # comparison using frozen V1 code. Only the naming context is amended.
    plan = launcher.authenticate(CAPTURE_PLAN)
    with corrected_prefix():
        execution, exact = v1.verify_execution(plan, CAPTURE_PLAN)
    if (execution.get('capture_protocol_complete') is not True or exact.get('all_exact') is not True
            or len(exact.get('windows', [])) != 32
            or any(row.get('rows') != 2047 or row.get('vocabulary') != 154880 or row.get('exact') is not True
                   for row in exact['windows'])):
        raise ValueError('V1 full capture did not pass complete exact closure')
    return {'capture_plan_sha256': CAPTURE_PLAN_SHA256,
            'capture_execution_sha256': CAPTURE_EXECUTION_SHA256,
            'full_exact_sha256': FULL_EXACT_SHA256, 'source_count': 56,
            'windows': 32, 'causal_rows': 65504, 'all_exact': True,
            'failed_unit': failed_unit, 'failed_evidence': failed_evidence,
            'teacher_scored_by_failed_attempt': False}


def make_plan(path):
    if path != path.resolve() or path.exists() or path.with_suffix('.sha256').exists() or OUTPUT.exists():
        raise ValueError('fresh canonical analysis amendment required')
    missing = [name for name in source_files() if not (REPO / name).is_file()]
    if missing:
        raise ValueError('analysis amendment source inventory incomplete')
    prerequisite = capture_prerequisite()
    plan = {**FIXED, 'created_at': pilot.now(), 'capture_prerequisite': prerequisite,
            'source_sha256': {name: sha(REPO / name) for name in sorted(source_files())}}
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), sha(path) + '  ' + path.name + '\n')
    return plan


def authenticate(path):
    if (path != path.resolve() or not path.is_file() or not path.with_suffix('.sha256').is_file()
            or sha(path) != path.with_suffix('.sha256').read_text().split()[0]):
        raise ValueError('sealed analysis amendment differs')
    plan = json.loads(path.read_text())
    if any(plan.get(key) != value for key, value in FIXED.items()):
        raise ValueError('fixed analysis amendment changed')
    if set(plan.get('source_sha256', {})) != source_files():
        raise ValueError('analysis amendment source inventory differs')
    for name, expected in plan['source_sha256'].items():
        if sha(REPO / name) != expected:
            raise ValueError('analysis amendment source changed')
    if capture_prerequisite() != plan.get('capture_prerequisite'):
        raise ValueError('capture prerequisite changed after seal')
    return plan


def output_inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink forbidden in analysis output')
        if path.is_file():
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def preserved_windows_scored(scored):
    """Recover frozen scorer progress without inferring it from loose files."""
    receipt = scored / 'analysis.failed.json'
    if not receipt.exists():
        return 0
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError('frozen scorer failure receipt is not a regular file')
    value = json.loads(receipt.read_text()).get('windows_scored')
    if type(value) is not int or not 0 <= value <= 32:
        raise ValueError('frozen scorer windows_scored is malformed')
    return value


def run(plan_path):
    plan = authenticate(plan_path)
    output = Path(plan['output'])
    if output != OUTPUT or output.exists():
        raise ValueError('fresh fixed analysis output required')
    output.mkdir(mode=0o700, parents=True)
    scored = output / 'scored'
    record = {'schema': 'glm53-p8.index-order-full-analysis-v2-execution.v1',
              'plan_sha256': sha(plan_path), 'started_at': pilot.now(), 'exit_code': 1,
              'capture_reused': True, 'gpu_launched': False, 'windows_scored': 0,
              'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
              'capture_prerequisite': plan['capture_prerequisite']}
    try:
        with corrected_prefix():
            result = v1.run(CAPTURE_PLAN, scored)
        if (result.get('schema') != 'glm53-p8.index-order-full-kld.v1'
                or result.get('status') != 'complete' or result.get('windows') != 32
                or result.get('causal_rows') != 65504 or result.get('exact_logits') is not True
                or result.get('protected_roles_opened') != []):
            raise ValueError('frozen scorer result contract differs')
        record.update(exit_code=0, windows_scored=32, analysis_sha256=sha(scored / 'analysis.json'),
                      exact_logits=True, mean_kld=result['reference_n128']['mean_kld'])
    except BaseException as error:
        record['error_type'] = type(error).__name__
        pilot.private_save(output / 'error.private.txt', str(error))
        try:
            record['windows_scored'] = preserved_windows_scored(scored)
        except BaseException as progress_error:
            record['progress_receipt_error_type'] = type(progress_error).__name__
            record['progress_receipt_error'] = str(progress_error)
    finally:
        record['finished_at'] = pilot.now()
        record['files'] = output_inventory(output)
        pilot.save(output / 'execution.json', record)
    if record['exit_code']:
        raise RuntimeError('analysis amendment failed; partial evidence preserved without retry')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan'); prepare.add_argument('--plan', type=Path, required=True)
    execute = commands.add_parser('run'); execute.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.plan) if args.command == 'plan' else run(args.plan), sort_keys=True))
