#!/usr/bin/env python3
"""Create a sanitized terminal snapshot of full32 capture plus V2 CPU KLD.

The snapshot copies JSON plans and receipts only. Raw logits and score NPZs are
rehash-verified in place; private logs and teacher/token payloads are not read
or copied. This script never launches a service, container, model, or scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4 import p8_decode_analysis as metric

CAPTURE_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-full-v1')
CAPTURE_PLAN = CAPTURE_REPO / 'experiments/p8-index-order-full-v1.json'
CAPTURE_PLAN_SHA256 = '6873cd1c066ad2845635cdb0db25df2352196240930d99ec213dc16b6a15133a'
CAPTURE_ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-order-full-v1')
CAPTURE_EXECUTION_SHA256 = '3a08163738655452c0c430a36ecb906e014b0bec7f5d8aeef80d70849ccac158'
FULL_EXACT_SHA256 = '89b628ec6519f3a2e070a2226e62140bbd9608e2f5ea57966a1c7e4ad1f53e10'

V2_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-full-analysis-v2')
V2_PLAN = V2_REPO / 'experiments/p8-index-order-full-analysis-v2.json'
V2_PLAN_SHA256 = '79e958c6463c647b24fc8d48640ff0c85f0e65393e2a7839bcf45103834abc9c'
V2_ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-order-full-v1-kld-v2')
V2_EXECUTION_SHA256 = 'a67f466084947759156737be705ccdfedc965d33c3d4bee9f172ab8e359faa23'
ANALYSIS_CONTRACT_SHA256 = 'd8c3b211d4f5d5f8f922d619fce5d1d78cdd15b1bb7be957923eb8f00a118ffd'
ANALYSIS_SHA256 = '83fa9f3700c8ad2ad7e2be77c42f4c682afb64d95a8bedf486ec02b3fbbfcb2c'
UNIT = 'glm53-p8-index-order-full-analysis-v2.service'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-order-full-analysis-v2'
SLOTS = ('full-n128', 'full-n64')


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def signature(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ino, value.st_dev


def receipt(path):
    path = Path(path)
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError('canonical regular input required')
    before = signature(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if signature(path) != before:
        raise ValueError('input changed while hashing')
    return {'bytes': before[0], 'sha256': digest}


def sealed_plan(path, expected_sha256):
    value = receipt(path)
    seal = path.with_suffix('.sha256')
    if (value['sha256'] != expected_sha256
            or seal.read_text().split() != [expected_sha256, path.name]):
        raise ValueError('sealed plan identity differs')
    receipt(seal)
    return json.loads(path.read_text())


def authenticate_sources(repo, mapping, expected_count):
    if not isinstance(mapping, dict) or len(mapping) != expected_count:
        raise ValueError('sealed source inventory count differs')
    observed = {}
    for name, expected in mapping.items():
        path = repo / name
        if (path != path.resolve() or not path.is_relative_to(repo)
                or receipt(path)['sha256'] != expected):
            raise ValueError('sealed source identity differs')
        observed[str(path)] = signature(path)
    return observed


def validate_input_stats(plan):
    """Check the scorer inputs still have their sealed stat identities.

    Teacher and token payloads are deliberately not opened here. Their hashes
    were verified by the completed scorer and are bound into each window JSON.
    """
    observed = {}
    for name, expected in plan.get('input_stats', {}).items():
        path = Path(name)
        if path != path.resolve() or path.is_symlink() or not path.is_file():
            raise ValueError('approved input is no longer a canonical regular file')
        stat = path.stat()
        actual = {'resolved_path': str(path.resolve()), 'bytes': stat.st_size,
                  'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
                  'device': stat.st_dev}
        if actual != expected:
            raise ValueError('approved teacher/token/role input stat changed')
        observed[str(path)] = signature(path)
    if len(observed) != 65:
        raise ValueError('expected role plus32 teacher and32 token stat receipts')
    return observed


def validate_capture(plan, execution, exact, check):
    window_ids = [window['id'] for window in plan.get('windows', [])]
    if (len(window_ids) != 32 or len(set(window_ids)) != 32
            or Counter(window['domain'] for window in plan['windows'])
            != Counter({name: 8 for name in ('axis1_general', 'axis2_legal',
                                              'axis3_code_agentic',
                                              'axis4_reasoning_termination')})):
        raise ValueError('capture plan is not the balanced full32 panel')
    if (execution.get('schema') != 'glm53-p8.index-order-full-execution.v1'
            or execution.get('plan_sha256') != CAPTURE_PLAN_SHA256
            or execution.get('exit_code') != 0
            or execution.get('capture_protocol_complete') is not True
            or execution.get('full_exact') is not True
            or execution.get('full_exact_sha256') != FULL_EXACT_SHA256
            or [row.get('slot') for row in execution.get('stages', [])] != list(SLOTS)
            or execution.get('opened_roles') != ['conditional-fit']
            or execution.get('protected_roles_opened') != []
            or execution.get('observer_enabled') is not False
            or execution.get('allocation_restart') is not False
            or execution.get('speed_measurement_valid') is not False
            or execution.get('final_identity_audit') != {'ok': True}
            or execution.get('restoration_safety') != {'ok': True, 'errors': [], 'containers': []}
            or execution.get('restoration') != {**execution.get('prior', {}), 'errors': []}):
        raise ValueError('completed full32 capture contract differs')
    if (exact.get('schema') != 'glm53-p8.forced-m1-v2-stage-exact.v1'
            or exact.get('stage') != 'full' or exact.get('all_exact') is not True
            or exact.get('allocation_restart') is not False
            or [row.get('window_id') for row in exact.get('windows', [])] != window_ids
            or any(row.get('rows') != 2047 or row.get('vocabulary') != 154880
                   or row.get('exact') is not True or row.get('unequal_values') != 0
                   or row.get('differing_rows') != []
                   for row in exact.get('windows', []))):
        raise ValueError('all-row N128=N64 exact result differs')

    stage_rows = {}
    stage_receipts = {}
    for slot, root_row in zip(SLOTS, execution['stages'], strict=True):
        path = CAPTURE_ROOT / slot / 'execution.json'
        if check(path)['sha256'] != root_row.get('execution_sha256'):
            raise ValueError('capture root-to-stage receipt chain differs')
        stage = json.loads(path.read_text())
        arm = slot.split('-', 1)[1]
        if (stage.get('schema') != 'glm53-p8.forced-m1-v2-stage.v1'
                or stage.get('stage') != 'full' or stage.get('arm') != arm
                or stage.get('plan_sha256') != CAPTURE_PLAN_SHA256
                or stage.get('exit_code') != 0
                or stage.get('cleanup') != {'ok': True, 'errors': []}
                or stage.get('unreadable_artifacts') != []
                or stage.get('protected_roles_opened') != []
                or stage.get('allocation_restart') is not False
                or [row.get('id') for row in stage.get('windows', [])] != window_ids):
            raise ValueError('full32 stage receipt differs')
        stage_receipts[slot] = stage
        stage_rows[slot] = {row['id']: row for row in stage['windows']}

    raw_receipts = {}
    for window in plan['windows']:
        wid = window['id']
        hashes = []
        for slot in SLOTS:
            row = stage_rows[slot][wid]
            relative = f'captures/{wid}.logits.f32'
            expected = stage_receipts[slot]['files'].get(relative)
            actual = check(CAPTURE_ROOT / slot / relative)
            if (row.get('rows') != 2047 or expected != actual
                    or row.get('raw_sha256') != actual['sha256']):
                raise ValueError('raw capture no longer matches its stage receipt')
            hashes.append(actual['sha256'])
        if len(set(hashes)) != 1:
            raise ValueError('current N128 and N64 raw logits are not bit-identical')
        raw_receipts[wid] = hashes[0]
    return window_ids, stage_receipts, raw_receipts


def validate_analysis(v2_plan, outer, contract, analysis, window_ids,
                      raw_receipts, check):
    scored = V2_ROOT / 'scored'
    if (v2_plan.get('schema') != 'glm53-p8.index-order-full-analysis-v2-plan.v1'
            or v2_plan.get('capture_plan') != str(CAPTURE_PLAN)
            or v2_plan.get('capture_plan_sha256') != CAPTURE_PLAN_SHA256
            or v2_plan.get('capture_execution_sha256') != CAPTURE_EXECUTION_SHA256
            or v2_plan.get('full_exact_sha256') != FULL_EXACT_SHA256
            or v2_plan.get('output') != str(V2_ROOT)
            or v2_plan.get('windows') != 32 or v2_plan.get('causal_rows') != 65504
            or v2_plan.get('gpu_launch_authorized') is not False
            or v2_plan.get('protected_roles_opened') != []
            or v2_plan.get('capture_prerequisite', {}).get('all_exact') is not True):
        raise ValueError('sealed V2 analysis scope differs')
    if (outer.get('schema') != 'glm53-p8.index-order-full-analysis-v2-execution.v1'
            or outer.get('plan_sha256') != V2_PLAN_SHA256
            or outer.get('exit_code') != 0 or outer.get('windows_scored') != 32
            or outer.get('capture_reused') is not True
            or outer.get('gpu_launched') is not False
            or outer.get('exact_logits') is not True
            or outer.get('protected_roles_opened') != []
            or outer.get('analysis_sha256') != ANALYSIS_SHA256):
        raise ValueError('successful V2 outer execution receipt differs')
    actual_files = {str(path.relative_to(V2_ROOT)): check(path)
                    for path in sorted(V2_ROOT.rglob('*'))
                    if path.is_file() and path != V2_ROOT / 'execution.json'}
    if actual_files != outer.get('files'):
        raise ValueError('V2 analysis output inventory differs')
    expected_names = {'scored/analysis-contract.json', 'scored/analysis.json'}
    expected_names |= {f'scored/{wid}.json' for wid in window_ids}
    expected_names |= {f'scored/{wid}.{arm}.npz'
                       for wid in window_ids for arm in ('n128', 'n64')}
    if set(actual_files) != expected_names:
        raise ValueError('V2 output contains missing or undeclared artifacts')
    if (actual_files['scored/analysis-contract.json']['sha256'] != ANALYSIS_CONTRACT_SHA256
            or actual_files['scored/analysis.json']['sha256'] != ANALYSIS_SHA256):
        raise ValueError('analysis contract or result identity differs')
    if (contract.get('plan_sha256') != CAPTURE_PLAN_SHA256
            or contract.get('source_receipt_sha256') != {
                str(CAPTURE_PLAN): CAPTURE_PLAN_SHA256,
                str(CAPTURE_ROOT / 'execution.json'): CAPTURE_EXECUTION_SHA256,
                str(CAPTURE_ROOT / 'full-exact.json'): FULL_EXACT_SHA256,
            }
            or contract.get('rows') != 65504 or contract.get('window_count') != 32
            or contract.get('bootstrap_seed') != 20260905
            or contract.get('bootstrap_replicates') != 20000
            or contract.get('torch_cpu_threads') != 8):
        raise ValueError('frozen analysis contract differs')
    if (analysis.get('schema') != 'glm53-p8.index-order-full-kld.v1'
            or analysis.get('status') != 'complete'
            or analysis.get('plan_sha256') != CAPTURE_PLAN_SHA256
            or analysis.get('exact_logits') is not True
            or analysis.get('windows') != 32 or analysis.get('causal_rows') != 65504
            or analysis.get('opened_roles') != ['conditional-fit']
            or analysis.get('protected_roles_opened') != []
            or analysis.get('speed_measurement_valid') is not False
            or analysis.get('allocation_restart') is not False):
        raise ValueError('completed KLD result contract differs')

    records = []
    by_id = {window['window_id']: window for window in analysis.get('window_records', [])}
    if list(by_id) != window_ids or len(by_id) != 32:
        raise ValueError('analysis window order or uniqueness differs')
    for wid in window_ids:
        path = scored / f'{wid}.json'
        row = json.loads(path.read_text())
        if (row != by_id[wid] or row.get('rows') != 2047
                or row.get('exact_logits') is not True
                or row.get('candidate_metric_reused_from_bitexact_reference') is not True
                or row.get('n128_raw_sha256') != raw_receipts[wid]
                or row.get('n64_raw_sha256') != raw_receipts[wid]
                or not all(type(row.get(key)) is float and math.isfinite(row[key])
                           for key in ('n128_mean_kld', 'n64_mean_kld'))
                or row['n128_mean_kld'] != row['n64_mean_kld']):
            raise ValueError('per-window exact/KLD receipt differs')
        for arm in ('n128', 'n64'):
            name = f'{wid}.{arm}.npz'
            if check(scored / name)['sha256'] != analysis['files'].get(name):
                raise ValueError('score array hash differs from result receipt')
        if analysis['files'][f'{wid}.n128.npz'] != analysis['files'][f'{wid}.n64.npz']:
            raise ValueError('bit-exact arms produced different score arrays')
        if check(path)['sha256'] != analysis['files'].get(path.name):
            raise ValueError('window JSON hash differs from result receipt')
        records.append(row)
    replayed = metric.aggregate(records, True)
    if any(analysis.get(key) != value for key, value in replayed.items()):
        raise ValueError('32-window aggregate does not replay')
    return records, replayed


def successful_invocation(window_ids, outer, command=subprocess.check_output):
    """Prefer retained systemd state; fall back to invocation-correlated journal."""
    fields = ('LoadState', 'ActiveState', 'MainPID', 'Result', 'ExecMainStatus',
              'InvocationID')
    raw = command(['systemctl', '--user', 'show', UNIT,
                   *sum((['-p', field] for field in fields), [])], text=True)
    state = dict(line.split('=', 1) for line in raw.strip().splitlines())
    if state.get('LoadState') != 'not-found':
        expected = {'ActiveState': 'inactive', 'MainPID': '0', 'Result': 'success',
                    'ExecMainStatus': '0'}
        if any(state.get(key) != value for key, value in expected.items()) or not re.fullmatch(
                r'[0-9a-f]{32}', state.get('InvocationID', '')):
            raise ValueError('analysis V2 systemd unit is not terminal success')
        return {'method': 'retained-systemd-state', 'unit': UNIT,
                'invocation_id': state['InvocationID'], 'terminal_success': True}

    journal = command(['journalctl', '--user', '-u', UNIT, '--no-pager', '-o', 'json'])
    if isinstance(journal, str):
        journal = journal.encode()
    entries = [json.loads(line) for line in journal.splitlines() if line.strip()]
    invocations = {entry.get('_SYSTEMD_INVOCATION_ID') for entry in entries
                   if entry.get('_SYSTEMD_USER_UNIT') == UNIT}
    invocations.discard(None)
    started = [entry for entry in entries if isinstance(entry.get('MESSAGE'), str)
               and entry['MESSAGE'].startswith('Started ' + UNIT + ' - ')]
    progress = []
    for entry in entries:
        message = entry.get('MESSAGE')
        if entry.get('_SYSTEMD_USER_UNIT') != UNIT or not isinstance(message, str):
            continue
        try:
            value = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            continue
        if set(('windows_scored', 'window_id')) <= set(value):
            progress.append((value['windows_scored'], value['window_id']))
    expected_command = ('/usr/bin/python3 -u -m glm53_nvfp4.p8_index_order_full_analysis_v2 '
                        f'run --plan {V2_PLAN}')
    if (len(invocations) != 1 or len(started) != 1
            or expected_command not in started[0]['MESSAGE']
            or progress != list(enumerate(window_ids, 1))
            or outer.get('exit_code') != 0 or outer.get('windows_scored') != 32):
        raise ValueError('collected systemd invocation cannot be correlated to success')
    invocation = next(iter(invocations))
    if not re.fullmatch(r'[0-9a-f]{32}', invocation):
        raise ValueError('invalid systemd invocation identity')
    canonical = {'unit': UNIT, 'invocation_id': invocation,
                 'start_command': expected_command,
                 'progress': [{'windows_scored': number, 'window_id': wid}
                              for number, wid in progress]}
    return {'method': 'collected-unit-journal-correlation', 'unit': UNIT,
            'invocation_id': invocation, 'terminal_success': True,
            'progress_events': 32,
            'canonical_journal_evidence_sha256': hashlib.sha256(encode(canonical)).hexdigest()}


def snapshot(destination=DEST):
    destination = Path(destination)
    if (not destination.is_absolute() or destination != destination.resolve()
            or destination.exists()):
        raise ValueError('fresh canonical snapshot destination required')
    inputs, stats = {}, {}

    def check(path, expected=None):
        path = Path(path)
        value = receipt(path)
        if expected is not None and value['sha256'] != expected:
            raise ValueError('pinned evidence identity differs')
        inputs[str(path)] = value
        stats[str(path)] = signature(path)
        return value

    capture_plan = sealed_plan(CAPTURE_PLAN, CAPTURE_PLAN_SHA256)
    for path in (CAPTURE_PLAN, CAPTURE_PLAN.with_suffix('.sha256')):
        check(path)
    stats.update(authenticate_sources(CAPTURE_REPO, capture_plan.get('source_sha256'), 56))
    input_stats = validate_input_stats(capture_plan)
    stats.update(input_stats)
    if (capture_plan.get('output') != str(CAPTURE_ROOT)
            or capture_plan.get('analysis_output') is None):
        raise ValueError('capture plan output scope differs')

    v2_plan = sealed_plan(V2_PLAN, V2_PLAN_SHA256)
    for path in (V2_PLAN, V2_PLAN.with_suffix('.sha256')):
        check(path)
    stats.update(authenticate_sources(V2_REPO, v2_plan.get('source_sha256'), 2))

    capture_execution_path = CAPTURE_ROOT / 'execution.json'
    exact_path = CAPTURE_ROOT / 'full-exact.json'
    capture_execution = json.loads(capture_execution_path.read_text())
    exact = json.loads(exact_path.read_text())
    check(capture_execution_path, CAPTURE_EXECUTION_SHA256)
    check(exact_path, FULL_EXACT_SHA256)
    window_ids, stages, raw_receipts = validate_capture(
        capture_plan, capture_execution, exact, check)

    outer_path = V2_ROOT / 'execution.json'
    contract_path = V2_ROOT / 'scored/analysis-contract.json'
    analysis_path = V2_ROOT / 'scored/analysis.json'
    outer = json.loads(outer_path.read_text())
    contract = json.loads(contract_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    check(outer_path, V2_EXECUTION_SHA256)
    check(contract_path, ANALYSIS_CONTRACT_SHA256)
    check(analysis_path, ANALYSIS_SHA256)
    records, replayed = validate_analysis(
        v2_plan, outer, contract, analysis, window_ids, raw_receipts, check)
    invocation = successful_invocation(window_ids, outer)

    report = {
        'schema': 'glm53-p8.index-order-full-analysis-v2-snapshot.v1',
        'status': 'full32-exact-and-analysis-complete',
        'capture_plan_sha256': CAPTURE_PLAN_SHA256,
        'capture_execution_sha256': CAPTURE_EXECUTION_SHA256,
        'full_exact_sha256': FULL_EXACT_SHA256,
        'analysis_v2_plan_sha256': V2_PLAN_SHA256,
        'analysis_v2_execution_sha256': V2_EXECUTION_SHA256,
        'analysis_contract_sha256': ANALYSIS_CONTRACT_SHA256,
        'analysis_sha256': ANALYSIS_SHA256,
        'source_counts': {'capture': 56, 'analysis_v2': 2},
        'windows': 32, 'causal_rows': 65504,
        'n128_n64_all_rows_exact': True,
        'aggregation_replayed': True,
        'mean_kld': replayed['reference_n128']['mean_kld'],
        'paired_delta_n64_minus_n128': replayed['paired_candidate_minus_reference']['mean_delta_kld'],
        'systemd_invocation': invocation,
        'capture_reused': True, 'gpu_launched_by_analysis': False,
        'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
        'speed_measurement_valid': False, 'allocation_restart': False,
        'raw_logits_copied': False, 'score_npz_copied': False,
        'private_logs_copied': False, 'teacher_or_token_contents_copied': False,
        'limits': analysis['limits'],
    }

    outputs = {
        'capture-plan.json': CAPTURE_PLAN.read_bytes(),
        'capture-plan.sha256': CAPTURE_PLAN.with_suffix('.sha256').read_bytes(),
        'analysis-v2-plan.json': V2_PLAN.read_bytes(),
        'analysis-v2-plan.sha256': V2_PLAN.with_suffix('.sha256').read_bytes(),
        'receipts/capture-execution.json': capture_execution_path.read_bytes(),
        'receipts/full-n128-execution.json': (CAPTURE_ROOT / 'full-n128/execution.json').read_bytes(),
        'receipts/full-n64-execution.json': (CAPTURE_ROOT / 'full-n64/execution.json').read_bytes(),
        'full-exact.json': exact_path.read_bytes(),
        'receipts/analysis-v2-execution.json': outer_path.read_bytes(),
        'analysis-contract.json': contract_path.read_bytes(),
        'analysis.json': analysis_path.read_bytes(),
        'report.json': encode(report),
    }
    for row in records:
        wid = row['window_id']
        outputs[f'windows/{wid}.json'] = (V2_ROOT / 'scored' / f'{wid}.json').read_bytes()
    forbidden = ('.npz', '.f32', '.npy', '.safetensors', '.private.', 'server-', 'request.json',
                 'response.json')
    if any(any(marker in name for marker in forbidden) for name in outputs):
        raise ValueError('snapshot copy allowlist includes forbidden payload')

    for name, before in stats.items():
        if signature(name) != before:
            raise ValueError('evidence input changed during snapshot validation')
    if successful_invocation(window_ids, outer) != invocation:
        raise ValueError('systemd invocation evidence changed during snapshot validation')
    manifest = {
        'schema': 'glm53-p8.index-order-full-analysis-v2-snapshot-manifest.v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'generator': receipt(Path(__file__)),
        'inputs': inputs,
        'outputs': {name: {'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}
                    for name, data in outputs.items()},
        'copy_policy': ('JSON plans, seals, execution receipts, exact result, analysis contract, '
                        'analysis result, and32 per-window JSON only'),
    }
    destination.mkdir(parents=True)
    for name, data in outputs.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with (destination / 'snapshot.json').open('xb') as stream:
        stream.write(encode(manifest))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output), sort_keys=True))
