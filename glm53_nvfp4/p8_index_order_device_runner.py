"""Sealed synthetic index-order gate; no model, corpus, or teacher is mounted."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1')
IMAGE = 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9'
PREFIX = 'glm53-p8-index-order-device-v2'
LABEL = 'org.klc.p8-index-order-plan'
SOURCE_FILES = (
    'glm53_nvfp4/p8_index_order_device_runner.py',
    'runtime_patch/sitecustomize.py',
    'runtime_patch/p8_index_order/__init__.py',
    'runtime_patch/p8_index_order/patches.py',
    'scripts/preflight_p8_index_order_device.py',
    'scripts/preflight_p8_index_order_import.py',
    'tests/test_p8_index_order_patch.py',
    'tests/test_p8_index_order_device.py',
    'tests/test_p8_index_order_device_runner.py',
)
PRIOR = ROOT / 'index-trace-v2/comparison.json'
PRIOR_SHA = '29111348a0ac0b1d35c6c8613400bb55ac8a545e753c3e0adc4332f40ac3fdab'
V1_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-fix-v1')
V1_PLAN_SHA = '5ed0f8c3268dbdc05d039335b08884f033c6642f2c7f3781fc502bc83873e021'
V1_ROOT_SHA = '23b973a5225823fd228941ae72200e52c67b1148ebc0885058879e17bf7339d3'
V1_RESULT_SHA = 'fe6f56c73dc1d90fff28cc41535b8b4dfd3c4d7c390f017e9804ac8489104574'
IMPORT_SOURCES = {'scripts/preflight_p8_index_order_import.py',
    'scripts/preflight_p8_index_order_device.py', 'runtime_patch/sitecustomize.py',
    'runtime_patch/p8_index_order/__init__.py', 'runtime_patch/p8_index_order/patches.py'}
LENGTHS = (0, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025)
TRANSITIONS = (512, 513) * 5 + (513, 512) * 5
FIXED = {
    'schema': 'glm53-p8.index-order-device-plan.v2', 'image': IMAGE,
    'amends_plan_sha256': V1_PLAN_SHA, 'amends_execution_sha256': V1_ROOT_SHA,
    'amendment': 'Preserve first kernel pass and stopped v1 campaign; canonicalize NVIDIA optional GPU- UUID prefix only. New five-process campaign, unchanged kernel/probe/numerical gate.',
    'physical_gpu': 0, 'independent_processes': 5, 'seed': 20260905,
    'timeout_per_process_seconds': 2400, 'thermal_start_max_c': 75,
    'thermal_abort_c': 90, 'minimum_free_bytes': 5 * 2**30,
    'model_loaded': False, 'teacher_logits_opened': False, 'protected_roles_opened': [],
    'speed_measurement_valid': False, 'allocation_restart': False,
    'prior_comparison_sha256': PRIOR_SHA,
    'experimental_unit': 'fresh GPU0 process; cases and replays are correlated subsamples',
    'decision': 'All fixed probe assertions pass in five processes and short-case byte signatures agree; any failure stops with no retry. This is a kernel gate, not full-model closure.',
    'cache': 'fresh campaign XDG cache, reused across processes; cold processes, not cold JIT caches',
    'next_stage': 'No automatic full-model launch; inspect and pre-register the integrated intervention separately.',
    'isa_cost': 'P8 E4M3 mxf8f6f4 has twice NVFP4 MMA issue count; no new throughput or KLD result.',
}


def command(args, *, check=True, timeout=60):
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=timeout)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def active(unit, user=False):
    value = command(['systemctl', *(['--user'] if user else []), 'is-active', unit], check=False).stdout.strip()
    if value not in {'active', 'inactive', 'failed'}:
        raise ValueError('ambiguous service state')
    return value == 'active'


def temperatures():
    values = [int(v) for v in command(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits']).stdout.splitlines()]
    if len(values) != 4:
        raise ValueError('thermal inventory differs')
    return values


def inventory():
    rows = command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,power.limit', '--format=csv,noheader']).stdout.strip().splitlines()
    if len(rows) != 4 or [int(row.split(',')[0]) for row in rows] != [0, 1, 2, 3]:
        raise ValueError('GPU inventory differs')
    return rows


def validate(plan):
    if any(plan.get(key) != value for key, value in FIXED.items()):
        raise ValueError('fixed device protocol differs')
    output = Path(plan['output'])
    if output.parent != ROOT or output != output.resolve() or output.name != 'index-order-device-v2':
        raise ValueError('unexpected output target')
    if set(plan['source_sha256']) != set(SOURCE_FILES):
        raise ValueError('sealed source inventory differs')
    for name, expected in plan['source_sha256'].items():
        if sha(REPO / name) != expected:
            raise ValueError(f'sealed source differs: {name}')
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError('prior diagnostic receipt differs')
    verify_v1()
    preflight = Path(plan['import_preflight'])
    if (preflight != preflight.resolve() or not preflight.is_relative_to(REPO)
            or sha(preflight) != plan['import_preflight_sha256']):
        raise ValueError('CPU import preflight identity differs')
    imported = json.loads(preflight.read_text())
    if (imported.get('schema') != 'glm53-p8.index-order-import.v1'
            or imported.get('status') != 'passed' or imported.get('image_id') != IMAGE
            or imported.get('gpu_used') is not False
            or any(imported.get(k) is not True for k in ('module_source_exact',
                       'cute_class_source_isolated', 'compile_cache_identity_isolated'))):
        raise ValueError('CPU import preflight gate failed')
    if set(imported.get('source_sha256', {})) != IMPORT_SOURCES:
        raise ValueError('CPU import preflight source inventory incomplete')
    for name, digest in imported['source_sha256'].items():
        if plan['source_sha256'].get(name) != digest:
            raise ValueError('CPU import preflight source stale')
    if inventory() != plan['gpu_inventory']:
        raise ValueError('GPU identity or power setting differs')
    return output


def make_plan(path):
    if path.exists() or path.with_suffix('.sha256').exists():
        raise ValueError('no plan overwrite')
    preflight = REPO / 'evidence/opened/codec-v2/p8-index-order-import-v1/import.json'
    plan = {**FIXED, 'created_at': now(), 'output': str(ROOT / 'index-order-device-v2'),
            'import_preflight': str(preflight), 'import_preflight_sha256': sha(preflight),
            'gpu_inventory': inventory(), 'source_sha256': {name: sha(REPO / name) for name in SOURCE_FILES}}
    validate(plan)
    save(path, plan)
    save(path.with_suffix('.sha256'), sha(path) + '  ' + path.name + '\n')
    return plan


def argv(plan, output, index, digest):
    uuid = plan['gpu_inventory'][0].split(',')[1].strip()
    name = f'{PREFIX}-{index:02d}'
    return ['docker', 'create', '--name', name, '--label', f'{LABEL}={digest}:{index}',
            '--gpus', f'device={uuid}', '--network', 'none', '--ipc', 'private', '--shm-size', '1g',
            '--entrypoint', '/opt/venv/bin/python',
            '-e', 'PYTHONPATH=/work:/runtime-patch:/opt/infernal-invocation/b12x',
            '-e', 'GLM53_P8_INDEX_ORDER=logical-short-v1', '-e', 'GLM53_P8_INDEX_TRACE=',
            '-e', 'GLM53_P8_NATIVE=', '-e', 'GLM53_P4_NATIVE=', '-e', 'OMP_NUM_THREADS=2',
            '-e', 'XDG_CACHE_HOME=/cache',
            '-v', f'{REPO}:/work:ro', '-v', f'{REPO}/runtime_patch:/runtime-patch:ro',
            '-v', f'{output}/cache:/cache:rw', '-v', f'{output}/repeat-{index:02d}:/out:rw',
            IMAGE, '/work/scripts/preflight_p8_index_order_device.py', '--output', '/out/result.json',
            '--image-id', IMAGE, '--seed', str(plan['seed'])]


def owned(name, digest, index):
    result = command(['docker', 'inspect', name], check=False)
    if result.returncode:
        if command(['docker', 'ps', '-a', '--filter', f'name=^/{name}$', '--format', '{{.ID}}']).stdout.strip():
            raise RuntimeError('container exists but cannot be inspected')
        return None
    item = json.loads(result.stdout)[0]
    if (not re.fullmatch('[a-f0-9]{64}', item['Id']) or item['Name'] != '/' + name
            or item['Image'] != IMAGE or item['Config'].get('Labels', {}).get(LABEL) != f'{digest}:{index}'):
        raise RuntimeError('container ownership mismatch')
    return item


def cleanup(name, digest, index, slot):
    item = owned(name, digest, index)
    if item is None:
        return
    if item['State']['Running']:
        command(['docker', 'stop', '--time', '20', item['Id']], check=False, timeout=35)
        item = owned(name, digest, index)
    if item['State']['Running']:
        command(['docker', 'kill', item['Id']], timeout=30)
        item = owned(name, digest, index)
    if item['State']['Running']:
        raise RuntimeError('owned container remains running')
    logs = command(['docker', 'logs', item['Id']], check=False)
    save(slot / 'probe.log', logs.stdout + logs.stderr)
    save(slot / 'container-state.json', item['State'])
    command(['docker', 'rm', item['Id']])


def validate_result(result, plan):
    if (result.get('schema') != 'glm53-p8.index-order-device.v1' or result.get('status') != 'passed'
            or result.get('image_id') != IMAGE or result.get('model_loaded') is not False
            or result.get('teacher_logits_opened') is not False
            or result.get('speed_measurement_valid') is not False
            or not re.fullmatch('[a-f0-9]{64}', result.get('determinism_sha256', ''))):
        raise ValueError('kernel gate failed or result contract differs')
    for name in ('scripts/preflight_p8_index_order_device.py', 'runtime_patch/p8_index_order/__init__.py',
                 'runtime_patch/p8_index_order/patches.py'):
        if result.get('source_sha256', {}).get(name) != plan['source_sha256'][name]:
            raise ValueError('probe source receipt differs')
    gpu = result.get('gpu', {})
    if (canonical_gpu_uuid(gpu.get('uuid', '')) != canonical_gpu_uuid(plan['gpu_inventory'][0].split(',')[1].strip())
            or gpu.get('compute_capability') != [12, 0]):
        raise ValueError('actual probe GPU does not match authorized physical GPU0')
    if (result.get('prefix_gate_points') != list(LENGTHS)
            or result.get('dynamic_length_transitions') != list(TRANSITIONS)
            or result.get('eager_repeats') != 5 or result.get('graph_repeats') != 5):
        raise ValueError('declared probe matrix differs')
    expected_cases, expected_transitions = expected_case_inventory()
    fields = ('arm', 'policy', 'scenario', 'length', 'mode', 'repeat')
    for key, expected in (('cases', expected_cases), ('state_transitions', expected_transitions)):
        rows = result.get(key, [])
        if Counter(tuple(row.get(f) for f in fields) for row in rows) != Counter(expected):
            raise ValueError('actual probe case inventory differs')
        for row in rows:
            if row.get('counter_zero') is not True:
                raise ValueError('counter reset not proven')
            expected_ctas = 4 if row['policy'] == 'forced-boundary' else gpu['multiprocessors']
            if row.get('ctas_per_group') != expected_ctas:
                raise ValueError('probe CTA grid differs')
            if row['policy'] == 'forced-boundary' and row.get('merge_threshold') != 1024:
                raise ValueError('forced boundary differs')
    return result['determinism_sha256']


def canonical_gpu_uuid(value):
    """Accept only a canonical UUID with NVIDIA's optional literal GPU- prefix."""
    if not isinstance(value, str):
        raise ValueError('GPU UUID must be text')
    raw = value[4:] if value.startswith('GPU-') else value
    if not re.fullmatch('[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', raw):
        raise ValueError('malformed GPU UUID')
    return str(uuid.UUID(raw))


def verify_v1():
    plan_path = V1_REPO / 'experiments/p8-index-order-device-v1.json'
    oldroot = ROOT / 'index-order-device-v1'
    if (sha(plan_path) != V1_PLAN_SHA or sha(oldroot / 'execution.json') != V1_ROOT_SHA
            or sha(oldroot / 'repeat-01/result.json') != V1_RESULT_SHA):
        raise ValueError('stopped v1 identities changed')
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(V1_REPO / name) != expected:
            raise ValueError('frozen v1 source changed')
    record = json.loads((oldroot / 'execution.json').read_text())
    if (record['exit_code'] != 1 or record['repeats'] != []
            or record['error'] != {'type': 'ValueError', 'message': 'actual probe GPU does not match authorized physical GPU0'}
            or record['restoration'] != {'backend': True, 'timer': False, 'safe': True, 'errors': []}
            or record['final_identity_audit'] is not True
            or (oldroot / 'repeat-02').exists()):
        raise ValueError('v1 terminal failure contract differs')
    # Diagnostic replay with corrected UUID parsing; does not rewrite the old failure.
    validate_result(json.loads((oldroot / 'repeat-01/result.json').read_text()), plan)


def expected_case_inventory():
    cases, transitions = [], []
    for arm in ('original', 'candidate'):
        for policy, lengths in (('forced-boundary', LENGTHS), ('serving-default-resolver', (65, 512, 513))):
            for mode in ('eager', 'graph'):
                for length in lengths:
                    for repeat in range(5):
                        cases.append((arm, policy, 'valid', length, mode, repeat))
                if policy == 'forced-boundary':
                    for page in (0, 1):
                        for repeat in range(5):
                            cases.append((arm, policy, f'negative-page-{page}', 65, mode, repeat))
            transitions.extend((arm, policy, 'valid', length, 'same-graph-512-513', repeat)
                               for repeat, length in enumerate(TRANSITIONS))
    return cases, transitions


def restore(prior, safe):
    errors = []
    try:
        if command(['docker', 'ps', '-a', '--filter', f'name=^/{PREFIX}-', '--format', '{{.ID}}']).stdout.strip():
            safe = False
    except Exception as error:
        safe = False
        errors.append('container-inventory:' + type(error).__name__)
    if safe:
        for needed, args in ((prior['backend'], ['systemctl', '--user', 'start', 'klc-backend.service']),
                             (prior['timer'], ['sudo', '-n', 'systemctl', 'start', 'klc-model-stack.timer'])):
            if needed:
                try:
                    command(args)
                except Exception as error:
                    errors.append('service-start:' + type(error).__name__)
    else:
        errors.append('UnsafeToRestoreWhileContainerOwnershipUnresolved')
    states = {}
    for key, unit, user in (('backend', 'klc-backend.service', True), ('timer', 'klc-model-stack.timer', False)):
        try:
            states[key] = active(unit, user)
        except Exception as error:
            states[key] = None
            errors.append('service-read:' + type(error).__name__)
    return {**states, 'safe': safe, 'errors': errors}


def run(path):
    plan = json.loads(path.read_text())
    digest = sha(path)
    if path.with_suffix('.sha256').read_text().split()[0] != digest:
        raise ValueError('plan seal differs')
    output = validate(plan)
    if output.exists() or command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip():
        raise ValueError('fresh output and clean committed checkout required')
    if shutil.disk_usage(ROOT).free < plan['minimum_free_bytes']:
        raise ValueError('disk headroom insufficient')
    if command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip() != IMAGE:
        raise ValueError('pinned image unavailable')
    terminal = command(['systemctl', '--user', 'show', 'glm53-p8-index-trace-v2.service', '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result']).stdout
    if set(terminal.splitlines()) != {'ActiveState=inactive', 'MainPID=0', 'Result=success'}:
        raise ValueError('preceding diagnostic not terminal successful')
    stopped = command(['systemctl', '--user', 'show', 'glm53-p8-index-order-device-v1.service', '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result']).stdout
    if set(stopped.splitlines()) != {'ActiveState=failed', 'MainPID=0', 'Result=exit-code'}:
        raise ValueError('v1 attempt must remain terminal failed')
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        (output / 'cache').mkdir()
        prior = {'backend': active('klc-backend.service', True), 'timer': active('klc-model-stack.timer')}
        record = {'schema': 'glm53-p8.index-order-device-execution.v2', 'started_at': now(),
                  'plan_sha256': digest, 'prior': prior, 'exit_code': 1, 'repeats': [],
                  'amends_plan_sha256': V1_PLAN_SHA, 'amends_execution_sha256': V1_ROOT_SHA,
                  'teacher_logits_opened': False, 'protected_roles_opened': [], 'allocation_restart': False}
        handlers, safe = {}, True
        def interrupted(signum, frame):
            raise RuntimeError(f'interrupted by signal {signum}')
        try:
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                handlers[sig] = signal.signal(sig, interrupted)
            if prior['timer']:
                command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            if command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip():
                raise RuntimeError('other GPU compute process active')
            signatures = []
            for index in range(1, 6):
                slot = output / f'repeat-{index:02d}'
                slot.mkdir()
                if max(temperatures()) > 75:
                    raise RuntimeError('start temperature above 75C; no automatic retry')
                name = f'{PREFIX}-{index:02d}'
                if owned(name, digest, index) is not None:
                    raise RuntimeError('fresh container required')
                args = argv(plan, output, index, digest)
                save(slot / 'launch.json', {'argv': args})
                try:
                    cid = command(args).stdout.strip()
                    if not re.fullmatch('[a-f0-9]{64}', cid):
                        raise RuntimeError('invalid container id')
                    save(slot / 'container-id.txt', cid + '\n')
                    command(['docker', 'start', cid])
                    deadline = time.monotonic() + 2400
                    with (slot / 'thermal.jsonl').open('x') as thermal:
                        while True:
                            values = temperatures()
                            thermal.write(json.dumps({'at': now(), 'temperatures': values}) + '\n')
                            thermal.flush()
                            if max(values) >= 90:
                                raise RuntimeError('thermal abort')
                            item = owned(name, digest, index)
                            if item is None:
                                raise RuntimeError('owned container unexpectedly disappeared')
                            if not item['State']['Running']:
                                if item['State']['ExitCode'] != 0:
                                    raise RuntimeError('probe process failed; no retry')
                                break
                            if time.monotonic() >= deadline:
                                raise RuntimeError('probe timeout; no retry')
                            time.sleep(2)
                    result = json.loads((slot / 'result.json').read_text())
                    signatures.append(validate_result(result, plan))
                    record['repeats'].append({'index': index, 'result_sha256': sha(slot / 'result.json'),
                                              'determinism_sha256': signatures[-1]})
                    if len(set(signatures)) != 1:
                        raise RuntimeError('cross-process short-case bytes differ')
                finally:
                    cleanup_handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in handlers}
                    try:
                        cleanup(name, digest, index, slot)
                    except BaseException:
                        safe = False
                        raise
                    finally:
                        for sig, handler in cleanup_handlers.items():
                            signal.signal(sig, handler)
            record['exit_code'] = 0
            record['kernel_gate_pass'] = True
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                validate(plan)
                record['final_identity_audit'] = True
            except Exception as error:
                record['final_identity_audit'] = False
                record['identity_error'] = type(error).__name__
                record['exit_code'] = 1
            try:
                restored = restore(prior, safe)
                record['restoration'] = restored
                if restored['errors'] or any(restored[key] != prior[key] for key in prior):
                    record['exit_code'] = 1
            except BaseException as error:
                record['restoration'] = {'safe': False, 'errors': [type(error).__name__]}
                record['exit_code'] = 1
            finally:
                record['finished_at'] = now()
                try:
                    save(output / 'execution.json', record)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
    if record['exit_code']:
        raise RuntimeError('device intervention gate failed; preserved without retry')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.plan) if args.action == 'plan' else run(args.plan)))
