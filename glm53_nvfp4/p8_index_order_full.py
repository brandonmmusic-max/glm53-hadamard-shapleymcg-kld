"""Sealed full32 corrected-index capture; reuses, but never repeats, the canary.

Planning and execution authenticate and replay the external three-process
canary.  Execution starts exactly two fresh serving processes, one per FC1
tile arm.  A non-exact full comparison is preserved as a completed numerical
result so the separately frozen analysis can report KLD without hiding the
closure failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal

from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol
from . import p8_index_order_integration as canary


pilot, cold = base.pilot, base.cold
REPO = Path(__file__).resolve().parents[1]
ROOT = base.ROOT
PREFIX = 'glm53-p8-index-order-full-v1'
IMAGE = canary.IMAGE
ORDER = [{'stage': 'full', 'arm': arm} for arm in ('n128', 'n64')]
RAW_CAPTURE_BYTES = 2 * 32 * protocol.ROWS * protocol.VOCAB_LIMIT * 4

CANARY_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-integration-v1')
CANARY_PLAN = CANARY_REPO / 'experiments/p8-index-order-integration-v1.json'
CANARY_PLAN_SHA256 = '5514b3f29db3aad7e88327f1af336013b35f951aa49c96d3846bd0d0a7833e0f'
CANARY_SOURCE_COUNT = 52
CANARY_UNIT = 'glm53-p8-index-order-integration-v1.service'
CANARY_ROOT = ROOT / 'index-order-integration-v1'
CANARY_EXECUTION_SHA256 = '4836ed7d6d94880c6bef90433adddaf55ff7fe5977a36b7dafa711ba6d1f30a6'

_BASE_CLONE_ARGV = base.clone_argv
_BASE_RUNTIME_AUDIT = base.runtime_audit

FIXED = {
    'schema': 'glm53-p8.index-order-full-plan.v1',
    'capture_image': IMAGE,
    'order': ORDER,
    'arms': base.ARMS,
    'topology': {'tp': 4, 'ep': False, 'dcp': 1},
    'kv_cache_dtype': 'nvfp4_ds_mla',
    'max_num_seqs': 1,
    'rows_per_window': protocol.ROWS,
    'full_windows': 32,
    'windows_per_domain': 8,
    'raw_capture_bytes': RAW_CAPTURE_BYTES,
    'thermal_start_max_c': 75,
    'thermal_abort_c': 90,
    'ready_timeout_seconds': 2400,
    'request_timeout_seconds': 900,
    'canary_plan': str(CANARY_PLAN),
    'canary_plan_sha256': CANARY_PLAN_SHA256,
    'canary_source_count': CANARY_SOURCE_COUNT,
    'canary_policy': 'authenticate and replay passed three-process canary; never repeat it in this protocol',
    'full_exact_policy': 'compare all65504 causal rows; preserve and analyze captures even when bitwise closure fails',
    'opened_roles': ['conditional-fit'],
    'protected_roles_opened': [],
    'speed_measurement_valid': False,
    'allocation_restart': False,
    'observer_enabled': False,
    'independent_processes': 2,
    'primary_metric': 'KL(teacher || student), nats, FP64 CPU, all154880 vocabulary entries',
    'analysis_contract': {'windows': 32, 'causal_rows': 65504, 'aggregation': 'equal-window mean',
                          'bootstrap_unit': 'paired window', 'bootstrap_replicates': 20000,
                          'bootstrap_seed': 20260905, 'cpu_threads': 8,
                          'decision': 'bitwise closure; KLD never overrides an exact failure'},
    'canary_execution_sha256': CANARY_EXECUTION_SHA256,
    'automatic_cpu_kld_after_successful_capture': True,
    'analysis_output': str(ROOT / 'index-order-full-v1-kld'),
    'experimental_unit': 'fresh serving process per arm; 32 windows and causal rows are correlated subsamples',
    'retry_policy': 'no retry or resume; preserve failures and require an explicit amended plan',
    'tail_contract': ('The correction orders retained compressed pools for pool_seq_len<=512. '
                      'The backend still consumes2048 of2051 expanded columns; incomplete-tail KLD is not isolated.'),
    'isa_cost': ('P8 E4M3 mxf8f6f4 issues twice the MMA instructions of NVFP4; '
                 'capture timings are not speed measurements.'),
}


def source_files():
    """Freeze the 52 inherited sources plus runner and separately owned analysis."""
    return canary.source_files() | {
        'glm53_nvfp4/p8_index_order_full.py',
        'tests/test_p8_index_order_full.py',
        'glm53_nvfp4/p8_index_order_full_analysis.py',
        'tests/test_p8_index_order_full_analysis.py',
    }


def slot_name(entry):
    if entry not in ORDER:
        raise ValueError('undeclared full32 stage')
    return f'full-{entry["arm"]}'


def _unit_terminal(command=pilot.command):
    state = command(['systemctl', '--user', 'show', CANARY_UNIT,
                     '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result']).stdout
    if set(state.strip().splitlines()) != {'ActiveState=inactive', 'MainPID=0', 'Result=success'}:
        raise ValueError('external canary unit must be observably terminal success')


def _small_file(path, expected=None):
    if path != path.resolve() or not path.is_file():
        raise ValueError('canonical canary receipt missing')
    digest = pilot.sha(path)
    if expected is not None and digest != expected:
        raise ValueError('canary receipt identity differs')
    return digest


def _validate_canary_stage(plan, entry, stage_path):
    receipt = json.loads(stage_path.read_text())
    expected_slot = canary.slot_name(entry)
    if (receipt.get('schema') != 'glm53-p8.forced-m1-v2-stage.v1'
            or receipt.get('plan_sha256') != CANARY_PLAN_SHA256
            or receipt.get('stage') != 'canary' or receipt.get('arm') != entry['arm']
            or receipt.get('capture_image') != IMAGE or receipt.get('exit_code') != 0
            or receipt.get('protected_roles_opened') != [] or receipt.get('allocation_restart') is not False
            or receipt.get('cleanup') != {'ok': True, 'errors': []}
            or receipt.get('unreadable_artifacts') != []
            or [row.get('id') for row in receipt.get('windows', [])] != [canary.WINDOW]
            or any(row.get('rows') != protocol.ROWS for row in receipt.get('windows', []))):
        raise ValueError(f'canary stage contract differs: {expected_slot}')
    final_path = stage_path.parent / 'runtime-final-audit.json'
    log_path = stage_path.parent / 'server-final.private.log'
    if (_small_file(final_path) != receipt.get('files', {}).get('runtime-final-audit.json', {}).get('sha256')
            or _small_file(log_path) != receipt.get('files', {}).get('server-final.private.log', {}).get('sha256')):
        raise ValueError('canary final runtime receipt is not bound to stage inventory')
    expected_runtime = runtime_audit(log_path.read_text(), entry['arm'], plan['windows'])
    if json.loads(final_path.read_text()) != expected_runtime:
        raise ValueError('canary runtime audit does not replay')
    for name, expected in receipt['files'].items():
        path = stage_path.parent / name
        if (path != path.resolve() or not path.is_relative_to(stage_path.parent)
                or path.stat().st_size != expected['bytes'] or pilot.sha(path) != expected['sha256']):
            raise ValueError('canary stage artifact changed')
    final = json.loads((stage_path.parent / 'container-final.private.json').read_text())
    if (final['Id'] != receipt.get('container_id') or final['Image'] != IMAGE
            or final['Name'] != f'/{canary.prefix(entry)}-canary-{entry["arm"]}'
            or final['State']['Running'] is not False
            or final['Config']['Labels'].get(cold.LABEL) != f'{CANARY_PLAN_SHA256}:canary-{entry["arm"]}'):
        raise ValueError('canary fresh owned container stop proof differs')
    for row in receipt['windows']:
        if (row['capture_sha256'] != receipt['files'][f'captures/{row["id"]}.capture.json']['sha256']
                or row['raw_sha256'] != receipt['files'][f'captures/{row["id"]}.logits.f32']['sha256']):
            raise ValueError('canary capture-to-stage chain differs')
    return receipt


def replay_canary():
    """Authenticate the external run and replay all three raw-logit comparisons.

    This reads the already-opened canary token and student captures only.  It
    does not resolve or read a teacher-logit path.
    """
    _unit_terminal()
    if CANARY_PLAN != CANARY_PLAN.resolve():
        raise ValueError('canonical external canary plan path required')
    _small_file(CANARY_PLAN, CANARY_PLAN_SHA256)
    _small_file(CANARY_PLAN.with_suffix('.sha256'))
    if CANARY_PLAN.with_suffix('.sha256').read_text().split()[0] != CANARY_PLAN_SHA256:
        raise ValueError('external canary plan seal differs')
    declared = json.loads(CANARY_PLAN.read_text())
    if (declared.get('schema') != canary.FIXED['schema']
            or declared.get('output') != str(CANARY_ROOT)
            or len(declared.get('source_sha256', {})) != CANARY_SOURCE_COUNT
            or set(declared.get('source_sha256', {})) != canary.source_files()):
        raise ValueError('external canary plan or 52-source inventory differs')
    for relative, expected in declared['source_sha256'].items():
        source = CANARY_REPO / relative
        if source != source.resolve() or not source.is_relative_to(CANARY_REPO) or pilot.sha(source) != expected:
            raise ValueError('external canary executed source differs')

    # Reuse the frozen verifier against this worktree too: every inherited
    # runtime byte must match the canary plan before any new plan is accepted.
    verified = canary.authenticate(CANARY_PLAN)
    if verified != declared:
        raise ValueError('external canary plan replay differs')

    execution_path = CANARY_ROOT / 'execution.json'
    execution_sha = _small_file(execution_path, CANARY_EXECUTION_SHA256)
    execution = json.loads(execution_path.read_text())
    if (execution.get('schema') != 'glm53-p8.index-order-integration-execution.v1'
            or execution.get('plan_sha256') != CANARY_PLAN_SHA256
            or execution.get('exit_code') != 0
            or execution.get('canary_exact_gate_passed') is not True
            or execution.get('full_model_kld_measured') is not False
            or execution.get('full_panel_authorized') is not False
            or execution.get('teacher_logits_opened') is not False
            or execution.get('protected_roles_opened') != []
            or execution.get('observer_enabled') is not False
            or execution.get('speed_measurement_valid') is not False
            or execution.get('allocation_restart') is not False
            or execution.get('final_identity_audit') != {'ok': True}
            or execution.get('restoration_safety') != {'ok': True, 'errors': [], 'containers': []}):
        raise ValueError('external canary terminal execution contract differs')
    prior, restored = execution.get('prior'), execution.get('restoration')
    if (not isinstance(prior, dict) or set(prior) != {'backend', 'timer'}
            or any(type(v) is not bool for v in prior.values()) or restored != {**prior, 'errors': []}):
        raise ValueError('external canary production restoration differs')
    containers = execution['restoration_safety'].get('containers', [])
    if any(row.get('running') is not False for row in containers):
        raise ValueError('external canary container remains live')
    if [{k: row.get(k) for k in ('index', 'arm')} for row in execution.get('stages', [])] != canary.ORDER:
        raise ValueError('external canary stage order differs')

    stage_hashes = {}
    container_ids = []
    for entry, recorded in zip(canary.ORDER, execution['stages']):
        path = CANARY_ROOT / canary.slot_name(entry) / 'execution.json'
        digest = _small_file(path, recorded.get('execution_sha256'))
        receipt = _validate_canary_stage(declared, entry, path)
        stage_hashes[canary.slot_name(entry)] = digest
        container_ids.append(receipt.get('container_id'))
    if (len(set(container_ids)) != 3
            or any(not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value) for value in container_ids)):
        raise ValueError('external canary did not use three fresh processes')

    n128_path = CANARY_ROOT / 'n128-repeat-exact.json'
    n64_path = CANARY_ROOT / 'n64-exact.json'
    n128_sha = _small_file(n128_path, execution.get('n128_repeat_sha256'))
    n64_sha = _small_file(n64_path, execution.get('n64_comparison_sha256'))
    n128 = canary.compare_pair(declared, canary.ORDER[0], canary.ORDER[1])
    if n128 != json.loads(n128_path.read_text()) or not canary.full_rows_exact(n128):
        raise ValueError('external N128 full-row comparison does not replay exactly')
    pairs = [canary.compare_pair(declared, reference, canary.ORDER[2]) for reference in canary.ORDER[:2]]
    n64 = {'pairs': pairs, 'all_exact': all(canary.full_rows_exact(pair) for pair in pairs)}
    if n64 != json.loads(n64_path.read_text()) or n64['all_exact'] is not True:
        raise ValueError('external N64 full-row comparison does not replay exactly')
    return {
        'plan_sha256': CANARY_PLAN_SHA256,
        'execution_sha256': execution_sha,
        'n128_repeat_sha256': n128_sha,
        'n64_comparison_sha256': n64_sha,
        'stage_execution_sha256': stage_hashes,
        'source_count': CANARY_SOURCE_COUNT,
        'raw_comparisons_replayed': True,
        'teacher_logits_opened': False,
    }


def verify_canary_receipts(plan):
    """Cheap stage-boundary verification; raw replay occurs in authenticate."""
    _unit_terminal()
    expected = plan['canary_receipts']
    observed = {
        'plan_sha256': _small_file(CANARY_PLAN, CANARY_PLAN_SHA256),
        'execution_sha256': _small_file(CANARY_ROOT / 'execution.json', CANARY_EXECUTION_SHA256),
        'n128_repeat_sha256': _small_file(CANARY_ROOT / 'n128-repeat-exact.json'),
        'n64_comparison_sha256': _small_file(CANARY_ROOT / 'n64-exact.json'),
        'stage_execution_sha256': {
            canary.slot_name(entry): _small_file(CANARY_ROOT / canary.slot_name(entry) / 'execution.json')
            for entry in canary.ORDER
        },
        'source_count': CANARY_SOURCE_COUNT,
        'raw_comparisons_replayed': True,
        'teacher_logits_opened': False,
    }
    if observed != expected:
        raise ValueError('external canary receipt changed after replay')


def approved_inputs(roles, teacher_root):
    windows = protocol.load_role_inputs(roles, teacher_root, verify_teacher_bytes=True)
    paths = {Path(roles)}
    for window in windows:
        paths.add(Path(window['token_path']))
        paths.add(Path(teacher_root) / window['teacher_path'])
    return windows, {str(path): base.fingerprint(path) for path in sorted(paths)}


def verify_input_stats(plan):
    for path, expected in plan['input_stats'].items():
        if base.fingerprint(Path(path)) != expected:
            raise ValueError('approved role/token/teacher metadata changed after byte verification')


def make_plan(path, output, roles, teacher_root):
    if any(p != p.resolve() for p in (path, output, roles, teacher_root)):
        raise ValueError('canonical planning paths required')
    if (path.exists() or path.with_suffix('.sha256').exists() or output.exists()
            or Path(FIXED['analysis_output']).exists() or output != ROOT / 'index-order-full-v1'):
        raise ValueError('fresh canonical plan, seal and output required')
    receipts = replay_canary()
    windows, stats = approved_inputs(roles, teacher_root)
    inherited = json.loads(CANARY_PLAN.read_text())
    files = source_files()
    missing = [name for name in files if not (REPO / name).is_file()]
    if missing:
        raise ValueError('full protocol source inventory incomplete')
    plan = {
        **copy.deepcopy(FIXED),
        'created_at': pilot.now(),
        'output': str(output),
        'roles': str(roles),
        'roles_sha256': protocol.ROLES_SHA256,
        'teacher_root': str(teacher_root),
        'windows': windows,
        'input_stats': stats,
        'canary_receipts': receipts,
        'image_receipt': inherited['image_receipt'],
        'image_receipt_sha256': inherited['image_receipt_sha256'],
        'capture_artifact_owner': {'uid': os.getuid(), 'gid': os.getgid()},
        'source_sha256': {name: pilot.sha(REPO / name) for name in sorted(files)},
        'limits': [
            'This is conditional-fit evidence, not a protected final qualification.',
            'KLD analysis is separate; this runner captures students and reports exact closure only.',
            'A bitwise mismatch remains a closure failure even if a later KLD comparison is favorable.',
            'The two serving processes are independent; windows and rows within each are subsamples.',
            'The unchanged incomplete-pool-tail behavior is not isolated by this experiment.',
        ],
    }
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), pilot.sha(path) + '  ' + path.name + '\n')
    return plan


def verify_identities(plan):
    if any(plan.get(key) != value for key, value in FIXED.items()):
        raise ValueError('fixed full32 protocol differs')
    if set(plan.get('source_sha256', {})) != source_files():
        raise ValueError('full32 source inventory differs')
    if Path(plan.get('output', '')) != (ROOT / 'index-order-full-v1').resolve():
        raise ValueError('full32 output scope differs')
    for name, expected in plan['source_sha256'].items():
        source = REPO / name
        if source != source.resolve() or not source.is_relative_to(REPO) or pilot.sha(source) != expected:
            raise ValueError('full32 source identity differs')
    for key in ('roles', 'teacher_root', 'image_receipt', 'output'):
        value = Path(plan[key])
        if value != value.resolve():
            raise ValueError('noncanonical full32 prerequisite path')
    if (pilot.sha(Path(plan['roles'])) != protocol.ROLES_SHA256
            or plan.get('roles_sha256') != protocol.ROLES_SHA256
            or pilot.sha(Path(plan['image_receipt'])) != plan.get('image_receipt_sha256')
            or plan.get('capture_artifact_owner') != {'uid': os.getuid(), 'gid': os.getgid()}):
        raise ValueError('full32 role, image, or artifact-owner identity differs')
    verify_input_stats(plan)
    verify_canary_receipts(plan)


def authenticate(path):
    if path != path.resolve() or not path.is_file() or not path.with_suffix('.sha256').is_file():
        raise ValueError('canonical sealed full32 plan required')
    if pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('full32 plan seal differs')
    plan = json.loads(path.read_text())
    verify_identities(plan)
    # Planning and running each independently replay the external raw canary.
    if replay_canary() != plan['canary_receipts']:
        raise ValueError('external canary replay differs from sealed plan')
    windows, stats = approved_inputs(Path(plan['roles']), Path(plan['teacher_root']))
    if windows != plan['windows'] or stats != plan['input_stats']:
        raise ValueError('full32 approved role/token/teacher inventory differs')
    return plan


def runtime_audit(log, arm, completed_windows=None):
    return {**_BASE_RUNTIME_AUDIT(log, arm, completed_windows), **canary.order_runtime_audit(log)}


def clone_argv(container, image, entry, out, windows, owner):
    env_names = [value.split('=', 1)[0] for value in container['Config']['Env']]
    if any(name.startswith(('GLM53_P8_INDEX_TRACE', 'GLM53_P8_INDEX_ORDER')) for name in env_names):
        raise ValueError('serving recipe unexpectedly enables observer or index correction')
    if any(bind.split(':')[1] == '/p8-index-traces' for bind in container['HostConfig']['Binds']):
        raise ValueError('observer mount forbidden')
    args = _BASE_CLONE_ARGV(container, image, entry, out, windows, owner)
    return [
        *args[:-3],
        '--env', 'GLM53_P8_INDEX_ORDER=logical-short-v1',
        '--env', 'GLM53_P8_INDEX_ORDER_RECEIPT=1',
        '--env', 'GLM53_P8_INDEX_TRACE=',
        *args[-3:],
    ]


@contextmanager
def stage_adapter(plan, entry):
    slot_name(entry)
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities', 'clone_argv', 'runtime_audit')
    original = {name: getattr(base, name) for name in names}
    try:
        base.PREFIX = PREFIX
        base.stage_windows = lambda value, stage: (
            value['windows'] if stage == 'full'
            else (_ for _ in ()).throw(ValueError('only full32 stage is declared'))
        )
        base.verify_stage_identities = verify_identities
        base.clone_argv = clone_argv
        base.runtime_audit = runtime_audit
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def compare_full(plan):
    """Pure full32 comparison; base globals are unmodified outside a stage."""
    result = base.compare_stage(plan, Path(plan['output']), 'full')
    if (len(result['windows']) != 32
            or [w['window_id'] for w in result['windows']] != [w['id'] for w in plan['windows']]
            or any(w['rows'] != 2047 or w['vocabulary'] != 154880 or type(w['exact']) is not bool
                   for w in result['windows'])):
        raise ValueError('full comparison must cover every declared causal row')
    return result


def restoration_safety(plan_hash, command=pilot.command):
    record = {'ok': False, 'containers': [], 'errors': []}
    try:
        expected = {f'/{PREFIX}-{slot_name(entry)}': entry for entry in ORDER}
        ids = command(['docker', 'ps', '-a', '--no-trunc', '--filter',
                       f'name=^/{PREFIX}-', '--format', '{{.ID}}']).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise ValueError('unresolved full32 container identity')
            container = json.loads(command(['docker', 'inspect', cid]).stdout)[0]
            entry = expected.get(container['Name'])
            if (entry is None or container['Id'] != cid or container['Image'] != IMAGE
                    or container['Config'].get('Labels', {}).get(cold.LABEL)
                    != f'{plan_hash}:{slot_name(entry)}'):
                raise ValueError('full32 capture ownership uncertain')
            record['containers'].append({'id': cid, 'name': container['Name'],
                                         'running': container['State']['Running']})
            if container['State']['Running']:
                raise RuntimeError('full32 capture container remains live')
        record['ok'] = True
    except Exception as error:
        record['errors'].append(type(error).__name__)
    return record


def prelaunch(plan):
    verify_identities(plan)
    if shutil.disk_usage(ROOT).free < RAW_CAPTURE_BYTES + 20 * 2**30:
        raise ValueError('two full32 captures plus20GiB headroom required')
    if pilot.command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip() != IMAGE:
        raise ValueError('exact capture image missing')
    receipt = base.verify_image_receipt(Path(plan['image_receipt']), IMAGE)
    installed = {**receipt['image_source_sha256'], **canary.lineage.UNMODIFIED_SOURCES}
    text = pilot.command([
        'docker', 'run', '--rm', '--network=none', '--runtime=runc',
        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--entrypoint', 'sha256sum', IMAGE, *installed,
    ]).stdout
    observed = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in text.splitlines()}
    if observed != installed:
        raise ValueError('installed full32 image source identity differs')
    recipe_path = cold.PILOT / 'container-final.private.json'
    if pilot.sha(recipe_path) != cold.PINNED[recipe_path]:
        raise ValueError('historical serving recipe differs')
    recipe = json.loads(recipe_path.read_text())
    cold.validate_recipe(recipe, 'p8')
    return recipe


def run(path):
    plan = authenticate(path)
    output, digest = Path(plan['output']), pilot.sha(path)
    if (output.exists() or output != output.resolve()
            or Path(plan['analysis_output']).exists()
            or pilot.command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip()):
        raise ValueError('fresh output and clean sealed checkout required')
    recipe = prelaunch(plan)
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {'backend': pilot.active('klc-backend.service', True),
                 'timer': pilot.active('klc-model-stack.timer')}
        record = {
            'schema': 'glm53-p8.index-order-full-execution.v1',
            'plan_sha256': digest,
            'started_at': pilot.now(),
            'prior': prior,
            'exit_code': 1,
            'stages': [],
            'capture_protocol_complete': False,
            'protected_roles_opened': [],
            'opened_roles': ['conditional-fit'],
            'allocation_restart': False,
            'speed_measurement_valid': False,
            'observer_enabled': False,
            'capture_image': IMAGE,
            'canary_receipts': plan['canary_receipts'],
            'full_model_kld_measured': False,
            'source_commit': pilot.command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).stdout.strip(),
        }
        handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f'full32 capture interrupted by signal {signum}')
            for sig in base.INTERRUPT_SIGNALS:
                handlers[sig] = signal.signal(sig, interrupted)
            if prior['timer']:
                pilot.command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                pilot.command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            hardware = cold.inventory()
            pilot.save(output / 'hardware-inventory.json', hardware)
            container_ids = []
            for entry in ORDER:
                directory = output / slot_name(entry)
                with stage_adapter(plan, entry):
                    stage = base.capture_stage(plan, entry, recipe, directory, digest, hardware)
                record['stages'].append({'slot': slot_name(entry),
                                         'execution_sha256': pilot.sha(directory / 'execution.json')})
                if stage['exit_code'] != 0:
                    raise RuntimeError('full32 capture stage failed; no retry or advance')
                if stage['container_id'] in container_ids:
                    raise ValueError('full32 arms did not use fresh serving processes')
                container_ids.append(stage['container_id'])

            exact = compare_full(plan)
            pilot.save(output / 'full-exact.json', exact)
            record['full_exact'] = exact['all_exact']
            record['full_exact_sha256'] = pilot.sha(output / 'full-exact.json')
            record['capture_protocol_complete'] = True
            record['exit_code'] = 0
        except BaseException as error:
            record['error_type'] = type(error).__name__
            pilot.private_save(output / 'error.private.txt', str(error))
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                verify_identities(plan)
                record['final_identity_audit'] = {'ok': True}
            except BaseException as error:
                record['final_identity_audit'] = {'ok': False, 'error_type': type(error).__name__}
                record['exit_code'] = 1
            try:
                safety = restoration_safety(digest)
                record['restoration_safety'] = safety
                record['restoration'] = cold.restore_if_safe(prior, safety)
                if (record['restoration']['errors']
                        or any(record['restoration'][key] != value for key, value in prior.items())):
                    record['exit_code'] = 1
            except BaseException as error:
                record['restoration'] = {'safe': False, 'errors': [type(error).__name__]}
                record['exit_code'] = 1
            finally:
                record['finished_at'] = pilot.now()
                try:
                    pilot.save(output / 'execution.json', record)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
    if record['exit_code']:
        raise RuntimeError('full32 capture protocol failed; partial evidence preserved without retry')
    return record


def run_and_analyze(path):
    """Serialized capture then CPU scoring; no analysis after a capture error."""
    captured = run(path)
    if captured.get('exit_code') != 0 or captured.get('capture_protocol_complete') is not True:
        raise ValueError('successful complete capture required before CPU KLD')
    from . import p8_index_order_full_analysis
    scored = p8_index_order_full_analysis.run(path, Path(FIXED['analysis_output']))
    return {'capture': captured, 'analysis': scored}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--roles', type=Path, required=True)
    prepare.add_argument('--teacher-root', type=Path, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--plan', type=Path, required=True)
    chained = commands.add_parser('run-and-analyze')
    chained.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = (make_plan(args.plan, args.output, args.roles, args.teacher_root)
              if args.command == 'plan' else
              run_and_analyze(args.plan) if args.command == 'run-and-analyze' else run(args.plan))
    print(json.dumps(result, sort_keys=True))
