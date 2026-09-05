"""Three fresh corrected serving captures; no teacher or observer is enabled.

The only lifecycle adaptation is process-local and restored in finally. The
two corrected N128 captures must match before the corrected N64 start is allowed.
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

import numpy as np

from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol
from . import p8_index_trace_launcher as lineage
from . import p8_index_order_device_runner as device

pilot, cold = base.pilot, base.cold
REPO = Path(__file__).resolve().parents[1]
ROOT = base.ROOT
PREFIX = 'glm53-p8-index-order-integration-v1'
IMAGE = device.IMAGE
WINDOW = 'conditional-fit-0056'
EMITTED = '655236be67b31d61a159fcce01ad85fe74b9c2445aedcbac3b474fb664e6de86'
ORIGINAL = '69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af'
DEVICE_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-fix-v2')
DEVICE_PLAN = DEVICE_REPO / 'experiments/p8-index-order-device-v2.json'
DEVICE_PLAN_SHA = 'fa2b503b9791fd9135240bee7a0b811084e98c2fe95f53bb12d2c53cc205db91'
DEVICE_ROOT = ROOT / 'index-order-device-v2'
DEVICE_ROOT_SHA = '251757fd24931fe2ca260fedd63ed1a3819aa5170194b0298dc95036c33390f5'
TRACE_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-index-trace-v2')
TRACE_PLAN = TRACE_REPO / 'experiments/p8-index-trace-v2.json'
TRACE_PLAN_SHA = 'd8e814b5a25ea7d557d2efeb297914aad4c3d98d338e1ef816c68d7c1ecddd2d'
TRACE_ROOT = ROOT / 'index-trace-v2'
TRACE_ROOT_SHA = 'cae40d0c7d10202fbb2a4251dcb44e795f2ff56facfa1323b520aace30d03a19'
TRACE_COMPARE_SHA = '29111348a0ac0b1d35c6c8613400bb55ac8a545e753c3e0adc4332f40ac3fdab'
ORDER = [{'index': 1, 'arm': 'n128'}, {'index': 2, 'arm': 'n128'}, {'index': 3, 'arm': 'n64'}]
RAW_BYTES = 3 * 2047 * 154880 * 4
FIXED = {'schema': 'glm53-p8.index-order-integration-plan.v1', 'capture_image': IMAGE,
         'order': ORDER, 'window_id': WINDOW, 'rows_per_window': 2047, 'raw_capture_bytes': RAW_BYTES,
         'topology': {'tp': 4, 'ep': False, 'dcp': 1}, 'kv_cache_dtype': 'nvfp4_ds_mla',
         'max_num_seqs': 1, 'arms': base.ARMS, 'thermal_start_max_c': 75, 'thermal_abort_c': 90,
         'ready_timeout_seconds': 2400, 'request_timeout_seconds': 900,
         'device_plan_sha256': DEVICE_PLAN_SHA, 'device_execution_sha256': DEVICE_ROOT_SHA,
         'trace_plan_sha256': TRACE_PLAN_SHA, 'trace_execution_sha256': TRACE_ROOT_SHA,
         'kernel_original_sha256': ORIGINAL, 'kernel_emitted_sha256': EMITTED,
         'teacher_logits_opened': False, 'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
         'observer_enabled': False, 'speed_measurement_valid': False, 'allocation_restart': False,
         'independent_processes': 3, 'experimental_unit': 'fresh serving process; one opened canary, correlated token rows',
         'hypothesis': 'Deterministic short-pool placement removes the previously localized serving non-repeatability.',
         'rivals': ['Unobserved attention or other model state remains non-repeatable.',
                    'N64 or fused-scratch arithmetic differs even after N128 repeatability is restored.'],
         'repeat_gate': 'All2047 raw rows bitwise exact across corrected N128 repeats before N64 starts.',
         'n64_gate': 'All2047 raw rows bitwise exact against both corrected N128 captures.',
         'marker_claim': 'Source-qualified successful public call, possibly startup; not standalone short-branch or M1 proof. Existing42-layer M1/runtime and causal-capture receipts separately validate serving.',
         'retry_policy': 'no retry or resume; preserve each failure and stop; require explicit amended plan',
         'next_stage': 'No automatic full panel or KLD. A separately sealed32-window protocol is required.',
         'isa_cost': 'P8 E4M3 mxf8f6f4 issues twice the MMA instructions of NVFP4; this is not a speed measurement.'}


def slot_name(entry):
    if entry not in ORDER:
        raise ValueError('undeclared integration stage')
    return f'repeat-{entry["index"]:02d}-{entry["arm"]}'


def prefix(entry):
    return f'{PREFIX}-repeat-{entry["index"]:02d}'


def authenticate_sources(repo, path, digest, expected_count):
    if path != path.resolve() or pilot.sha(path) != digest or path.with_suffix('.sha256').read_text().split()[0] != digest:
        raise ValueError('historical plan seal differs')
    plan = json.loads(path.read_text())
    if len(plan['source_sha256']) != expected_count:
        raise ValueError('historical source inventory differs')
    for name, expected in plan['source_sha256'].items():
        source = repo / name
        if source != source.resolve() or not source.is_relative_to(repo) or pilot.sha(source) != expected:
            raise ValueError('historical frozen source drift')
    return plan


def historical_inputs():
    trace = authenticate_sources(TRACE_REPO, TRACE_PLAN, TRACE_PLAN_SHA, 39)
    if (trace['output'] != str(TRACE_ROOT) or pilot.sha(TRACE_ROOT / 'execution.json') != TRACE_ROOT_SHA
            or pilot.sha(TRACE_ROOT / 'comparison.json') != TRACE_COMPARE_SHA):
        raise ValueError('prior completed trace identity differs')
    trace_execution = json.loads((TRACE_ROOT / 'execution.json').read_text())
    if (trace_execution['exit_code'] != 0 or trace_execution['comparison_sha256'] != TRACE_COMPARE_SHA
            or trace_execution['plan_sha256'] != TRACE_PLAN_SHA
            or trace_execution['final_identity_audit'] != {'ok': True}
            or trace_execution['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
            or [entry['index'] for entry in trace_execution['repeats']] != [1, 2]
            or trace_execution['teacher_logits_opened'] is not False or trace_execution['protected_roles_opened'] != []
            or trace_execution['restoration'] != {'backend': True, 'timer': False, 'errors': []}):
        raise ValueError('prior trace terminal contract differs')
    for entry in trace_execution['repeats']:
        stage_path = TRACE_ROOT / f'repeat-{entry["index"]:02d}-n128/execution.json'
        if pilot.sha(stage_path) != entry['execution_sha256'] or json.loads(stage_path.read_text())['exit_code'] != 0:
            raise ValueError('prior trace stage receipt differs')
    prior = authenticate_sources(DEVICE_REPO, DEVICE_PLAN, DEVICE_PLAN_SHA, 9)
    if prior['output'] != str(DEVICE_ROOT) or pilot.sha(DEVICE_ROOT / 'execution.json') != DEVICE_ROOT_SHA:
        raise ValueError('five-process device gate identity differs')
    execution = json.loads((DEVICE_ROOT / 'execution.json').read_text())
    if (execution['exit_code'] != 0 or execution.get('kernel_gate_pass') is not True
            or execution['final_identity_audit'] is not True
            or execution['plan_sha256'] != DEVICE_PLAN_SHA
            or execution['restoration'] != {'backend': True, 'timer': False, 'safe': True, 'errors': []}
            or [entry['index'] for entry in execution['repeats']] != list(range(1, 6))):
        raise ValueError('five fresh successful device processes required')
    signatures, ids = [], []
    for entry in execution['repeats']:
        directory = DEVICE_ROOT / f'repeat-{entry["index"]:02d}'
        path = directory / 'result.json'
        if pilot.sha(path) != entry['result_sha256']:
            raise ValueError('device result identity differs')
        result = json.loads(path.read_text())
        signatures.append(device.validate_result(result, prior))
        if (signatures[-1] != entry['determinism_sha256']
                or result['source_transformations']['emitted_sha256'] != EMITTED
                or result['source_transformations']['original_sha256'] != ORIGINAL):
            raise ValueError('device determinism or emitted source identity differs')
        ids.append((directory / 'container-id.txt').read_text().strip())
        state = json.loads((directory / 'container-state.json').read_text())
        if state['Running'] is not False or state['ExitCode'] != 0:
            raise ValueError('prior device container not successfully stopped')
    if len(set(signatures)) != 1 or len(set(ids)) != 5 or any(not re.fullmatch('[a-f0-9]{64}', value) for value in ids):
        raise ValueError('device five-process identity or determinism differs')
    if pilot.sha(REPO / 'runtime_patch/p8_index_order/patches.py') != prior['source_sha256']['runtime_patch/p8_index_order/patches.py']:
        raise ValueError('device-qualified kernel transformation changed')
    # This established helper validates original v3/cold/image/weights and reads
    # only role metadata, the one already-opened token file, and carrier config.
    window, stats, old = lineage.historical_inputs()
    if window['id'] != WINDOW or trace['windows'] != [window]:
        raise ValueError('already-opened canary identity differs')
    return window, stats, old


def source_files():
    return base.SOURCE_FILES | lineage.source_files() | set(device.SOURCE_FILES) | {
        'glm53_nvfp4/p8_index_order_integration.py', 'tests/test_p8_index_order_integration.py',
        'runtime_patch/p8_index_order_receipt.py', 'tests/test_p8_index_order_receipt.py',
        'scripts/preflight_p8_index_order_serving_import.py'}


def import_receipt(path):
    if path != path.resolve() or not path.is_file():
        raise ValueError('canonical integration import preflight required')
    value = json.loads(path.read_text())
    expected = {'schema': 'glm53-p8.index-order-serving-import.v1', 'status': 'passed',
                'image_id': IMAGE, 'gpu_used': False, 'model_loaded': False, 'teacher_logits_opened': False,
                'module_source_exact': True, 'cute_class_source_isolated': True,
                'compile_cache_identity_isolated': True, 'receipt_wrapper_installed': True,
                'wrapped_source_globals_exact': True, 'worker_breakable_graph_mode': True,
                'serving_rank_apis_available': True, 'marker_emitted': False,
                'actual_distributed_call_tested': False}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('integration import preflight failed or differs')
    names = {'runtime_patch/sitecustomize.py', 'runtime_patch/p8_index_order/__init__.py',
             'runtime_patch/p8_index_order/patches.py', 'runtime_patch/p8_index_order_receipt.py',
             'scripts/preflight_p8_index_order_serving_import.py',
             'scripts/preflight_p8_index_order_import.py', 'scripts/preflight_p8_index_order_device.py'}
    if value.get('source_sha256') != {name: pilot.sha(REPO / name) for name in names}:
        raise ValueError('integration import source inventory differs')
    source = value.get('source_transformation', {})
    if (source.get('original_sha256') != ORIGINAL or source.get('emitted_sha256') != EMITTED
            or source.get('inspect_source_sha256') != EMITTED or source.get('cache_suffix') != '_p8logicalshortv1'):
        raise ValueError('integration import kernel identity differs')


def make_plan(path, output, preflight):
    if (path != path.resolve() or output != output.resolve() or output != ROOT / 'index-order-integration-v1'
            or path.exists() or path.with_suffix('.sha256').exists() or output.exists()):
        raise ValueError('fresh canonical plan, seal and output required')
    window, stats, old = historical_inputs()
    import_receipt(preflight)
    plan = {**copy.deepcopy(FIXED), 'created_at': pilot.now(), 'output': str(output), 'windows': [window],
            'input_stats': stats, 'image_receipt': old['image_receipt'],
            'image_receipt_sha256': old['image_receipt_sha256'],
            'import_preflight': str(preflight), 'import_preflight_sha256': pilot.sha(preflight),
            'capture_artifact_owner': {'uid': os.getuid(), 'gid': os.getgid()},
            'source_sha256': {name: pilot.sha(REPO / name) for name in sorted(source_files())}}
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), pilot.sha(path) + '  ' + path.name + '\n')
    return plan


def verify_identities(plan):
    if any(plan.get(k) != v for k, v in FIXED.items()) or set(plan['source_sha256']) != source_files():
        raise ValueError('fixed integration protocol or source inventory differs')
    if Path(plan['output']) != (ROOT / 'index-order-integration-v1').resolve():
        raise ValueError('integration output scope differs')
    for name, expected in plan['source_sha256'].items():
        source = REPO / name
        if source != source.resolve() or not source.is_relative_to(REPO) or pilot.sha(source) != expected:
            raise ValueError('integration source drift')
    preflight = Path(plan['import_preflight'])
    if pilot.sha(preflight) != plan['import_preflight_sha256']:
        raise ValueError('integration import receipt drift')
    import_receipt(preflight)
    window, stats, old = historical_inputs()
    if (plan['windows'] != [window] or plan['input_stats'] != stats
            or plan['image_receipt'] != old['image_receipt'] or plan['image_receipt_sha256'] != old['image_receipt_sha256']
            or plan['capture_artifact_owner'] != {'uid': os.getuid(), 'gid': os.getgid()}):
        raise ValueError('integration payload or owner drift')


def authenticate(path):
    if path != path.resolve() or pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('integration plan seal differs')
    plan = json.loads(path.read_text())
    verify_identities(plan)
    return plan


def order_runtime_audit(log):
    marker = 'GLM53_P8_INDEX_ORDER_RECEIPT '
    matches = [json.loads(line.split(marker, 1)[1]) for line in log.splitlines() if marker in line]
    expected = {'schema': 'glm53-p8.index-order-serving-receipt.v1',
                'module': 'b12x.attention.nsa_indexer.fused_indexer',
                'mode': 'logical-short-v1', 'tp_world_size': 4,
                'original_sha256': ORIGINAL, 'emitted_sha256': EMITTED, 'inspect_source_sha256': EMITTED,
                'cache_suffix': '_p8logicalshortv1', 'num_heads': 32, 'topk': 512, 'output_physical_slots': False}
    if (len(matches) != 4 or {v.get('tp_rank') for v in matches} != set(range(4))
            or any(any(v.get(k) != value for k, value in expected.items()) for v in matches)
            or any(type(v.get(k)) is not int for v in matches for k in ('tp_rank', 'global_rank', 'tp_world_size'))
            or any(v.get('global_rank') != v['tp_rank'] or type(v.get('pid')) is not int or v['pid'] <= 0 for v in matches)
            or len({v['pid'] for v in matches}) != 4):
        raise ValueError('four exact corrected-indexer worker receipts required')
    if 'GLM53_P8_INDEX_TRACE_COMPLETE' in log:
        raise ValueError('observer must not run in integration closure')
    return {'index_order_ranks': list(range(4)), 'index_order_worker_receipts': matches,
            'kernel_emitted_sha256': EMITTED, 'observer_enabled': False,
            'marker_evidence': 'source-qualified successful public API call; not standalone direct-branch/M1 proof'}


@contextmanager
def stage_adapter(plan, entry):
    slot_name(entry)
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities', 'clone_argv', 'runtime_audit')
    original = {name: getattr(base, name) for name in names}
    def clone(container, image, stage, out, windows, owner):
        env = [v.split('=', 1)[0] for v in container['Config']['Env']]
        if any(k.startswith(('GLM53_P8_INDEX_TRACE', 'GLM53_P8_INDEX_ORDER')) for k in env):
            raise ValueError('prior recipe unexpectedly enables observer or index correction')
        if any(bind.split(':')[1] == '/p8-index-traces' for bind in container['HostConfig']['Binds']):
            raise ValueError('observer mount forbidden')
        args = original['clone_argv'](container, image, stage, out, windows, owner)
        return [*args[:-3], '--env', 'GLM53_P8_INDEX_ORDER=logical-short-v1',
                '--env', 'GLM53_P8_INDEX_ORDER_RECEIPT=1', '--env', 'GLM53_P8_INDEX_TRACE=', *args[-3:]]
    def runtime(log, arm, completed_windows=None):
        return {**original['runtime_audit'](log, arm, completed_windows), **order_runtime_audit(log)}
    try:
        base.PREFIX = prefix(entry)
        base.stage_windows = lambda p, stage: p['windows'] if stage == 'canary' else (_ for _ in ()).throw(ValueError('full panel forbidden'))
        base.verify_stage_identities = verify_identities
        base.clone_argv, base.runtime_audit = clone, runtime
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def compare_pair(plan, left, right):
    tokens = np.load(plan['windows'][0]['token_path'], allow_pickle=False)
    arrays, metas = [], []
    for entry in (left, right):
        array, meta = protocol.load_capture(Path(plan['output']) / slot_name(entry) / 'captures', WINDOW, tokens)
        arrays.append(array); metas.append(meta)
    for key in ('original_logit_dtype', 'original_logit_width', 'sequence_token_ids_sha256'):
        if metas[0][key] != metas[1][key]:
            raise ValueError('paired original dtype, vocabulary or forced causal sequence differs')
    result = protocol.exact_logits(*arrays, chunk_rows=8)
    full_rows_exact(result)
    return {'left': slot_name(left), 'right': slot_name(right), 'window_id': WINDOW,
            **result, 'teacher_logits_opened': False, 'kld_measured': False}


def full_rows_exact(result):
    if result.get('rows') != 2047 or result.get('vocabulary') != 154880 or type(result.get('exact')) is not bool:
        raise ValueError('exact comparison must cover every2047rows and154880vocabulary entries')
    return result['exact']


def restoration_safety(digest):
    safety = {'ok': False, 'errors': [], 'containers': []}
    try:
        expected = {f'/{prefix(entry)}-canary-{entry["arm"]}': entry for entry in ORDER}
        ids = pilot.command(['docker', 'ps', '-a', '--no-trunc', '--filter', f'name=^/{PREFIX}-', '--format', '{{.ID}}']).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise ValueError('invalid owned container id')
            value = json.loads(pilot.command(['docker', 'inspect', cid]).stdout)[0]
            entry = expected.get(value['Name'])
            if (entry is None or value['Id'] != cid or value['Image'] != IMAGE
                    or value['Config'].get('Labels', {}).get(cold.LABEL) != f'{digest}:canary-{entry["arm"]}'):
                raise ValueError('integration container ownership uncertain')
            safety['containers'].append({'id': cid, 'name': value['Name'], 'running': value['State']['Running']})
            if value['State']['Running']:
                raise ValueError('integration container remains live')
        safety['ok'] = True
    except Exception as error:
        safety['errors'].append(type(error).__name__)
    return safety


def prelaunch(plan):
    for unit in ('glm53-p8-index-order-device-v2.service', 'glm53-p8-index-trace-v2.service'):
        state = pilot.command(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result']).stdout
        if set(state.splitlines()) != {'ActiveState=inactive', 'MainPID=0', 'Result=success'}:
            raise ValueError('prior gate must be terminal successful')
    if shutil.disk_usage(ROOT).free < RAW_BYTES + 20 * 2**30:
        raise ValueError('three full captures plus20GiB headroom required')
    if pilot.command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip() != IMAGE:
        raise ValueError('exact capture image missing')
    receipt = base.verify_image_receipt(Path(plan['image_receipt']), IMAGE)
    installed = {**receipt['image_source_sha256'], **lineage.UNMODIFIED_SOURCES}
    text = pilot.command(['docker', 'run', '--rm', '--network=none', '--runtime=runc', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
                          '--entrypoint', 'sha256sum', IMAGE, *installed]).stdout
    if {s.split(maxsplit=1)[1].strip(): s.split()[0] for s in text.splitlines()} != installed:
        raise ValueError('installed image source identity differs')
    path = cold.PILOT / 'container-final.private.json'
    if pilot.sha(path) != cold.PINNED[path]:
        raise ValueError('historical serving recipe differs')
    recipe = json.loads(path.read_text()); cold.validate_recipe(recipe, 'p8')
    return recipe


def run(path):
    plan = authenticate(path)
    output, digest = Path(plan['output']), pilot.sha(path)
    if output.exists() or pilot.command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip():
        raise ValueError('fresh output and clean sealed checkout required')
    recipe = prelaunch(plan)
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {'backend': pilot.active('klc-backend.service', True), 'timer': pilot.active('klc-model-stack.timer')}
        record = {'schema': 'glm53-p8.index-order-integration-execution.v1', 'plan_sha256': digest,
                  'started_at': pilot.now(), 'prior': prior, 'exit_code': 1, 'stages': [],
                  'teacher_logits_opened': False, 'protected_roles_opened': [], 'observer_enabled': False,
                  'speed_measurement_valid': False, 'allocation_restart': False, 'full_panel_authorized': False}
        handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f'integration interrupted by signal {signum}')
            for sig in base.INTERRUPT_SIGNALS:
                handlers[sig] = signal.signal(sig, interrupted)
            if prior['timer']:
                pilot.command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                pilot.command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            hardware = cold.inventory(); pilot.save(output / 'hardware-inventory.json', hardware)
            ids = set()
            for entry in ORDER:
                directory = output / slot_name(entry)
                with stage_adapter(plan, entry):
                    stage = base.capture_stage(plan, {'stage': 'canary', 'arm': entry['arm']}, recipe, directory, digest, hardware)
                record['stages'].append({**entry, 'execution_sha256': pilot.sha(directory / 'execution.json')})
                if stage['exit_code'] != 0:
                    raise RuntimeError('capture stage failed; no retry')
                if stage['container_id'] in ids:
                    raise ValueError('capture processes were not fresh')
                ids.add(stage['container_id'])
                if entry['index'] == 2:
                    repeated = compare_pair(plan, ORDER[0], ORDER[1])
                    pilot.save(output / 'n128-repeat-exact.json', repeated)
                    record['n128_repeat_sha256'] = pilot.sha(output / 'n128-repeat-exact.json')
                    if not full_rows_exact(repeated):
                        raise RuntimeError('corrected N128 repeatability failed; N64 not started')
                if entry['index'] == 3:
                    comparisons = [compare_pair(plan, reference, ORDER[2]) for reference in ORDER[:2]]
                    all_exact = all(full_rows_exact(pair) for pair in comparisons)
                    pilot.save(output / 'n64-exact.json', {'pairs': comparisons, 'all_exact': all_exact})
                    record['n64_comparison_sha256'] = pilot.sha(output / 'n64-exact.json')
                    if not all_exact:
                        raise RuntimeError('corrected N64 exact closure failed')
            record.update(exit_code=0, canary_exact_gate_passed=True, full_model_kld_measured=False)
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
                if record['restoration']['errors'] or any(record['restoration'][k] != v for k, v in prior.items()):
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
        raise RuntimeError('integration canary gate failed; preserved without retry')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--import-preflight', type=Path, required=True)
    execute = commands.add_parser('run'); execute.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.plan, args.output, args.import_preflight) if args.command == 'plan' else run(args.plan)
    print(json.dumps(result))
