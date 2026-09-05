"""Two same-N128 diagnostic captures, using a scoped legacy lifecycle adapter.

No teacher loads, GPU calls or service actions occur on import or planning.
The adapter mutates only this launcher's imported base module, restores every
attribute in finally, and never edits the frozen base or capture helper files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import time

import numpy as np

from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol
from .p8_weight_identity import fingerprint

pilot, cold = base.pilot, base.cold
REPO = Path(__file__).resolve().parents[1]
ROOT = base.ROOT
PREFIX = 'glm53-p8-index-trace-v1'
OLD_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-capture-v3')
OLD_PLAN = OLD_REPO / 'experiments/p8-forced-m1-v2-v3.json'
OLD_PLAN_SHA = '7b675117a4a2c8a5abe87056dae6a3b6bcd1db2d44d8d0b5c45ad0f33b219085'
OLD_OUTPUT = ROOT / 'forced-m1-v2-v3'
OLD_EXEC_SHA = '4a1c85cc91f0f89a858ad4509d5cb76b05e5c73fe607de01518313b0c8f7e63c'
OLD_GATE_SHA = '6d6a16679528bbe6890bfb4a1df1b88d7f4a37607e1902a5eae1995310e98bb2'
IMAGE = 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9'
WINDOW = 'conditional-fit-0056'
MODEL_CONFIG = Path('/media/brandonmusic/klcstore/bmxfp4-glm53/codec-v2/p8-h128/identity-mcg-layer3-v1/candidate-bf16-pseudoquant/config.json')
MODEL_CONFIG_SHA = '16c46a1e0684671791c833662a81e6d2150fcd5e82216d2427cc8ea97e21ca7c'
UNMODIFIED_SOURCES = {'/opt/infernal-invocation/b12x/b12x/attention/nsa_indexer/fused_indexer.py':
                      '69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af'}
REPEATS = (1, 2)
RAW_BYTES = 2 * protocol.ROWS * protocol.VOCAB_LIMIT * 4
FIXED = {'schema': 'glm53-p8.index-trace-plan.v1', 'capture_image': IMAGE,
         'model_config': str(MODEL_CONFIG), 'model_config_sha256': MODEL_CONFIG_SHA,
         'unmodified_image_sources': UNMODIFIED_SOURCES,
         'repeats': list(REPEATS), 'window_id': WINDOW, 'trace_rows': list(range(255, 264)),
         'rows_per_window': 2047, 'raw_capture_bytes': RAW_BYTES,
         'topology': {'tp': 4, 'ep': False, 'dcp': 1}, 'kv_cache_dtype': 'nvfp4_ds_mla',
         'fc1_tile_n': 128, 'fused_scratch_zero': False, 'max_num_seqs': 1,
         'thermal_start_max_c': 75, 'thermal_abort_c': 90,
         'ready_timeout_seconds': 2400, 'request_timeout_seconds': 900,
         'teacher_logits_opened': False, 'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
         'speed_measurement_valid': False, 'allocation_restart': False,
         'retry_policy': 'no retry or resume; preserve failed attempt',
         'decision': 'complete two same-N128 captures and validate traces; logit differences do not stop repeat two',
         'hypothesis': 'The repeated N128 logit divergence near row259 can be localized to sparse index selection or attention ordering under unchanged inputs.',
         'rivals': ['Index query or index-weight inputs differ before pool selection.',
                    'Valid pooled-index cache bytes or valid MLA cache bytes differ.',
                    'Selected logical token sets differ.',
                    'The selected set is unchanged but ordering differs.',
                    'Attention output differs despite identical query, valid caches and exact selection order.',
                    'The bounded observed seams do not expose the divergence or the observer changes scheduling.'],
         'mechanism_rule': 'Compare identical positions/layers: query and weights, full valid index cache, full valid MLA cache, selected set, selection ordering, then attention output. Changes upstream prevent attributing downstream differences uniquely.',
         'null_rule': 'No observed difference in two runs is inconclusive, not determinism or qualification.',
         'observer_limit': '143 observer copy kernels per token across11layers may strongly perturb CTA arrival and suppress or change the mechanism. This is one-way positive diagnostic evidence only; a null is not falsification and never authorizes automatic reruns.',
         'evidence_level': 'diagnostic, not numerical or speed qualification'}


def repeat_name(index):
    if index not in REPEATS:
        raise ValueError('undeclared repeat')
    return f'repeat-{index:02d}-n128'


def prefix(index):
    return f'{PREFIX}-repeat-{index:02d}'


def verify_model_config():
    if MODEL_CONFIG != MODEL_CONFIG.resolve() or pilot.sha(MODEL_CONFIG) != MODEL_CONFIG_SHA:
        raise ValueError('pinned carrier model config differs')
    text = json.loads(MODEL_CONFIG.read_text())['text_config']
    layers = text['layer_types']
    if (text['num_hidden_layers'] != 45 or len(layers) != 45
            or set(layers) != {'linear_attention', 'deepseek_sparse_attention'}
            or [index for index, kind in enumerate(layers) if kind == 'deepseek_sparse_attention'] != list(range(3, 44, 4))):
        raise ValueError('nested sparse-layer model geometry differs')


def historical_inputs():
    """Authenticate metadata and source only: deliberately no teacher loader."""
    verify_model_config()
    if pilot.sha(OLD_PLAN) != OLD_PLAN_SHA or OLD_PLAN.with_suffix('.sha256').read_text().split()[0] != OLD_PLAN_SHA:
        raise ValueError('old v3 plan seal differs')
    old = json.loads(OLD_PLAN.read_text())
    if (Path(old['output']) != OLD_OUTPUT or old['capture_image'] != IMAGE
            or pilot.sha(OLD_OUTPUT / 'execution.json') != OLD_EXEC_SHA
            or pilot.sha(OLD_OUTPUT / 'canary-exact.json') != OLD_GATE_SHA):
        raise ValueError('old v3 failed-gate receipts differ')
    execution = json.loads((OLD_OUTPUT / 'execution.json').read_text())
    gate = json.loads((OLD_OUTPUT / 'canary-exact.json').read_text())
    if (execution['exit_code'] != 1 or gate['all_exact'] is not False
            or execution['restoration'] != {'backend': True, 'timer': False, 'errors': []}
            or execution['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
            or execution['protected_roles_opened'] != []):
        raise ValueError('historical gate/restoration status differs')
    for relative, digest in old['source_sha256'].items():
        p = OLD_REPO / relative
        if p != p.resolve() or not p.is_relative_to(OLD_REPO) or pilot.sha(p) != digest:
            raise ValueError('frozen v3 source differs')
    roles = Path(old['roles'])
    if pilot.sha(roles) != protocol.ROLES_SHA256:
        raise ValueError('role metadata hash differs')
    rows = protocol.validate_role_payload(json.loads(roles.read_text()))
    matches = [w for w in rows if w['id'] == WINDOW]
    if len(matches) != 1 or rows[0]['id'] != WINDOW:
        raise ValueError('already-opened canary role differs')
    window = copy.deepcopy(matches[0])
    token_path = Path(window['token_path'])
    if token_path != token_path.resolve() or pilot.sha(token_path) != window['input_sha256']:
        raise ValueError('single approved token file differs')
    protocol.full_window_contract(np.load(token_path, allow_pickle=False))
    image_receipt = Path(old['image_receipt'])
    if pilot.sha(image_receipt) != old['image_receipt_sha256']:
        raise ValueError('original instrumentation image receipt differs')
    base.verify_image_receipt(image_receipt, IMAGE)
    cold_path = Path(old['cold_plan'])
    if pilot.sha(cold_path) != old['cold_plan_sha256']:
        raise ValueError('prior cold plan differs')
    cold_plan = json.loads(cold_path.read_text())
    cold.weight_receipt(Path(cold_plan['weight_audit']), cold_plan['weight_audit_sha256'])
    return window, {str(p): fingerprint(p) for p in (roles, token_path, MODEL_CONFIG)}, old


def source_files():
    return base.SOURCE_FILES | {'glm53_nvfp4/p8_index_trace_launcher.py',
        'tests/test_p8_index_trace_launcher.py',
        'tests/test_p8_index_trace_observer.py', 'scripts/preflight_p8_index_trace.py',
        'glm53_nvfp4/p8_index_trace_analysis.py', 'tests/test_p8_index_trace_analysis.py',
        *(f'runtime_patch/p8_index_trace/{name}' for name in ('__init__.py', 'observer.py', 'patches.py'))}


def core_transformations(value):
    spec = importlib.util.spec_from_file_location('p8_index_trace_pinned_patches', REPO / 'runtime_patch/p8_index_trace/patches.py')
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)
    if not isinstance(value, dict) or set(value) != set(patches.ORIGINAL_SHA):
        raise ValueError('exact three source transformation modules required')
    core = {}
    for name, original in patches.ORIGINAL_SHA.items():
        entry = value[name]
        if (entry.get('module') != name or entry.get('original_sha256') != original
                or not re.fullmatch('[a-f0-9]{64}', entry.get('emitted_sha256', ''))
                or entry['emitted_sha256'] == original):
            raise ValueError('source transformation original/emitted hash invalid')
        core[name] = {key: entry[key] for key in ('module', 'original_sha256', 'emitted_sha256')}
    return core


def preflight_receipt(path):
    if path != path.resolve() or not path.is_file():
        raise ValueError('canonical source preflight receipt required')
    value = json.loads(path.read_text())
    required = {'schema': 'glm53-p8.index-trace-preflight.v1', 'status': 'passed', 'device': 'cpu',
                'cpu_interpreter': True, 'cuda_graph_replay_tested': False, 'image_id': IMAGE,
                'model_loaded': False, 'teacher_logits_opened': False, 'speed_measurement_valid': False}
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError('exact-image CPU source preflight differs')
    if value.get('unmodified_sources') != UNMODIFIED_SOURCES:
        raise ValueError('unmodified B12X fused-indexer source identity missing or changed')
    return core_transformations(value['source_transformations'])


def make_plan(path, output, source_preflight):
    if (path != path.resolve() or output != output.resolve() or output.parent != ROOT
            or path.exists() or path.with_suffix('.sha256').exists() or output.exists()):
        raise ValueError('fresh canonical plan, seal and campaign output required')
    window, stats, old = historical_inputs()
    transformations = preflight_receipt(source_preflight)
    plan = {**copy.deepcopy(FIXED), 'created_at': pilot.now(), 'output': str(output), 'windows': [window],
            'input_stats': stats, 'old_plan_sha256': OLD_PLAN_SHA, 'old_execution_sha256': OLD_EXEC_SHA,
            'image_receipt': old['image_receipt'], 'image_receipt_sha256': old['image_receipt_sha256'],
            'source_preflight': str(source_preflight), 'source_preflight_sha256': pilot.sha(source_preflight),
            'source_transformations': transformations,
            'source_preflight_script_sha256': pilot.sha(REPO / 'scripts/preflight_p8_index_trace.py'),
            'capture_artifact_owner': {'uid': os.getuid(), 'gid': os.getgid()},
            'source_sha256': {name: pilot.sha(REPO / name) for name in sorted(source_files())}}
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), pilot.sha(path) + '  ' + path.name + '\n')
    return plan


def verify_identities(plan):
    if any(plan.get(k) != value for k, value in FIXED.items()) or not source_files() <= plan['source_sha256'].keys():
        raise ValueError('index trace fixed protocol differs')
    if Path(plan['output']) != Path(plan['output']).resolve() or Path(plan['output']).parent != ROOT:
        raise ValueError('index trace output scope differs')
    for name, digest in plan['source_sha256'].items():
        p = REPO / name
        if p != p.resolve() or not p.is_relative_to(REPO) or pilot.sha(p) != digest:
            raise ValueError('index trace source drift')
    preflight = Path(plan['source_preflight'])
    if (pilot.sha(preflight) != plan['source_preflight_sha256']
            or pilot.sha(REPO / 'scripts/preflight_p8_index_trace.py') != plan['source_preflight_script_sha256']
            or preflight_receipt(preflight) != plan['source_transformations']):
        raise ValueError('source preflight or emitted-source identity drift')
    window, stats, old = historical_inputs()
    if (plan['windows'] != [window] or plan['input_stats'] != stats
            or plan['image_receipt'] != old['image_receipt'] or plan['image_receipt_sha256'] != old['image_receipt_sha256']
            or plan['old_plan_sha256'] != OLD_PLAN_SHA or plan['old_execution_sha256'] != OLD_EXEC_SHA
            or plan['capture_artifact_owner'] != {'uid': os.getuid(), 'gid': os.getgid()}):
        raise ValueError('index trace identity or role drift')


def authenticate(path):
    if path != path.resolve() or pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('index trace seal differs')
    plan = json.loads(path.read_text())
    verify_identities(plan)
    return plan


TRACE_FILES = {f'rank-{rank}{suffix}' for rank in range(4) for suffix in ('.json', '-buffers.npz')}
LAYERS = list(range(3, 44, 4))
TRACE_LABELS = {'pool_ids', 'pool_lengths', 'pool_table', 'index_query', 'index_weights', 'pool_cache',
                'attention_lengths', 'attention_query', 'logical_tokens', 'physical_slots',
                'attention_table', 'attention_cache', 'attention_output'}
BASE_ARRAYS = {f'layer{layer:03d}__{label}' for layer in LAYERS for label in TRACE_LABELS}
ALL_ARRAYS = {key + suffix for key in BASE_ARRAYS for suffix in ('', '__counts', '__errors')}


def trace_manifests(root, transformations):
    expected_transformations = core_transformations(transformations)
    if root != root.resolve() or not root.is_dir() or {p.name for p in root.iterdir()} != TRACE_FILES:
        raise ValueError('exact eight-file index trace inventory required')
    result = []
    for rank in range(4):
        meta, raw = root / f'rank-{rank}.json', root / f'rank-{rank}-buffers.npz'
        if any(p != p.resolve() or not stat.S_ISREG(p.lstat().st_mode) for p in (meta, raw)):
            raise ValueError('index trace refuses symlink or nonregular file')
        value = json.loads(meta.read_text())
        required = {'schema': 'glm53-p8.index-trace.v1', 'window_id': WINDOW, 'tp_rank': rank,
            'positions': list(range(255, 264)), 'layers': LAYERS, 'raw_file': raw.name,
            'capture_complete': True, 'speed_measurement_valid': False,
            'protected_roles_opened': [], 'counts_valid': True}
        if (any(value.get(key) != expected for key, expected in required.items())
                or not isinstance(value.get('arrays'), dict) or set(value['arrays']) != ALL_ARRAYS
                or not value.get('source_transformations') or pilot.sha(raw) != value.get('raw_sha256')):
            raise ValueError('index trace manifest identity/completeness differs')
        if core_transformations(value['source_transformations']) != expected_transformations:
            raise ValueError('index trace original/emitted source transformations differ from preflight')
        with np.load(raw, allow_pickle=False) as arrays:
            if set(arrays.files) != set(value['arrays']):
                raise ValueError('index trace array inventory differs')
            for key, spec in value['arrays'].items():
                array = arrays[key]
                if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
                    raise ValueError('index trace array geometry/dtype differs')
            for key in BASE_ARRAYS:
                data, counts, errors = (arrays[key + suffix] for suffix in ('', '__counts', '__errors'))
                if (data.ndim != 2 or data.shape[0] != 9 or data.dtype != np.uint8
                        or counts.shape != (9,) or errors.shape != (9,)
                        or counts.dtype != np.int32 or errors.dtype != np.int32
                        or not np.all(counts == 1) or not np.all(errors == 0)
                        or value['arrays'][key].get('recorded_in_cuda_graph') is not True):
                    raise ValueError('index trace per-row counts/errors or graphed evidence invalid')
        result.append({'tp_rank': rank, 'manifest_sha256': pilot.sha(meta),
                       'raw_sha256': value['raw_sha256'], 'source_transformations': value['source_transformations']})
    return result


def normalize_traces(root, cid, image, name, owner, target, transformations):
    # Capture HTTP can complete slightly before the other ranks finish their
    # files. Wait boundedly; no retry of an inference request is performed.
    deadline = time.monotonic() + 30
    while {p.name for p in root.iterdir()} != TRACE_FILES:
        if time.monotonic() >= deadline:
            raise RuntimeError('four-rank index trace publication timeout')
        time.sleep(.25)
    c = json.loads(pilot.command(['docker', 'inspect', cid]).stdout)[0]
    if (c['Id'] != cid or not re.fullmatch('[a-f0-9]{64}', cid) or c['Image'] != image
            or c['Name'] != '/' + name or c['Config'].get('Labels', {}).get(cold.LABEL) != owner
            or c['State']['Running'] is not True):
        raise ValueError('trace ownership container identity differs')
    before = {}
    for filename in sorted(TRACE_FILES):
        p = root / filename
        if p != p.resolve() or not stat.S_ISREG(p.lstat().st_mode):
            raise ValueError('trace normalization refuses symlink/nonregular file')
        before[filename] = fingerprint(p)
    pilot.command(['docker', 'exec', '--user', '0:0', cid, 'chown', '-h', f'{target["uid"]}:{target["gid"]}', '--',
                   *('/p8-index-traces/' + name for name in sorted(TRACE_FILES))])
    for filename, expected in before.items():
        p = root / filename
        if fingerprint(p) != expected or (p.stat().st_uid, p.stat().st_gid) != (target['uid'], target['gid']):
            raise ValueError('trace file identity or owner changed unexpectedly')
    return {'container_id': cid, 'operation': 'chown -h eight explicit trace files only',
            'target': target, 'files': before, 'manifests': trace_manifests(root, transformations)}


@contextmanager
def stage_adapter(plan, index):
    """Only valid in this single-threaded launcher process; always restored."""
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities', 'clone_argv',
             'normalize_capture_ownership', 'runtime_audit')
    original = {name: getattr(base, name) for name in names}
    def clone(container, image, entry, out, windows, owner):
        if any(v.split('=', 1)[0].startswith('GLM53_P8_INDEX_TRACE') for v in container['Config']['Env']):
            raise ValueError('source recipe unexpectedly enables index trace')
        if any(bind.split(':')[1] == '/p8-index-traces' for bind in container['HostConfig']['Binds']):
            raise ValueError('index trace mount collision')
        argv = original['clone_argv'](container, image, entry, out, windows, owner)
        (out / 'index-traces').mkdir(mode=0o700)
        return [*argv[:-3], '--env', 'GLM53_P8_INDEX_TRACE=1',
                '--env', 'GLM53_P8_INDEX_TRACE_ROOT=/p8-index-traces',
                '--volume', f'{out / "index-traces"}:/p8-index-traces:rw', *argv[-3:]]
    def normalize(root, windows, cid, image, name, owner, target, **kwargs):
        result = original['normalize_capture_ownership'](root, windows, cid, image, name, owner, target, **kwargs)
        trace = normalize_traces(root.parent / 'index-traces', cid, image, name, owner, target,
                                 plan['source_transformations'])
        pilot.save(root.parent / 'trace-ownership.json', trace)
        return result
    def runtime(log, arm, completed_windows=None):
        proof = original['runtime_audit'](log, arm, completed_windows)
        if completed_windows is not None:
            matches = re.findall(r'GLM53_P8_INDEX_TRACE_COMPLETE tp_rank=(\d+) window=(conditional-fit-\d{4}) rows=(\d+) layers=(\d+)', log)
            if len(matches) != 4 or set(matches) != {(str(r), WINDOW, '9', '11') for r in range(4)}:
                raise ValueError('four-rank trace completion markers missing or duplicated')
            proof['index_trace_complete_ranks'] = list(range(4))
        return proof
    try:
        base.PREFIX = prefix(index)
        base.stage_windows = lambda p, stage: p['windows'] if stage == 'canary' else (_ for _ in ()).throw(ValueError('no full-panel stage'))
        base.verify_stage_identities = verify_identities
        base.clone_argv, base.normalize_capture_ownership, base.runtime_audit = clone, normalize, runtime
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def restoration_safety(plan_hash):
    result = {'ok': False, 'containers': [], 'errors': []}
    try:
        expected = {f'/{prefix(index)}-canary-n128' for index in REPEATS}
        ids = pilot.command(['docker', 'ps', '-a', '--no-trunc', '--filter', f'name=^/{PREFIX}-', '--format', '{{.ID}}']).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise ValueError('invalid trace container id')
            c = json.loads(pilot.command(['docker', 'inspect', cid]).stdout)[0]
            if (c['Id'] != cid or c['Name'] not in expected or c['Image'] != IMAGE
                    or c['Config'].get('Labels', {}).get(cold.LABEL) != plan_hash + ':canary-n128'):
                raise ValueError('trace container ownership uncertain')
            result['containers'].append({'id': cid, 'name': c['Name'], 'running': c['State']['Running']})
            if c['State']['Running']:
                raise ValueError('trace container still running')
        result['ok'] = True
    except Exception as error:
        result['errors'].append(type(error).__name__)
    return result


def compare_repeats(plan):
    root = Path(plan['output'])
    tokens = np.load(plan['windows'][0]['token_path'], allow_pickle=False)
    arrays, metadata = [], []
    for index in REPEATS:
        slot = root / repeat_name(index)
        array, meta = protocol.load_capture(slot / 'captures', WINDOW, tokens)
        arrays.append(array)
        metadata.append(meta)
        trace_manifests(slot / 'index-traces', plan['source_transformations'])
    for key in ('original_logit_dtype', 'original_logit_width', 'sequence_token_ids_sha256'):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError('same-N128 logit representation or causal sequence differs')
    exact = protocol.exact_logits(*arrays, chunk_rows=8)
    del arrays
    from . import p8_index_trace_analysis
    analysis = p8_index_trace_analysis.compare(root, transformations=plan['source_transformations'])
    return {'schema': 'glm53-p8.index-trace-comparison.v1', 'same_n128_logits': exact,
        'index_trace_analysis': analysis, 'teacher_logits_opened': False, 'kld_measured': False,
        'qualification_pass': False, 'speed_measurement_valid': False, 'allocation_restart': False,
        'interpretation': 'Full2047 same-N128 logits show whether divergence recurred under instrumentation; matching logits in two runs remain inconclusive, not falsification or proof of observer causality.'}


def run(path):
    plan = authenticate(path)
    output, digest = Path(plan['output']), pilot.sha(path)
    if output.exists() or pilot.command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip():
        raise ValueError('fresh output and clean sealed checkout required')
    state = pilot.command(['systemctl', '--user', 'show', 'glm53-p8-forced-m1-v2-v3.service', '-p', 'ActiveState', '-p', 'Result']).stdout
    if set(state.strip().splitlines()) != {'ActiveState=failed', 'Result=exit-code'}:
        raise ValueError('original v3 unit must be terminal failed')
    if shutil.disk_usage(ROOT).free < RAW_BYTES + 20 * 2**30:
        raise ValueError('insufficient capture and trace disk headroom')
    if pilot.command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip() != IMAGE:
        raise ValueError('exact capture image unavailable')
    receipt = base.verify_image_receipt(Path(plan['image_receipt']), IMAGE)
    installed = {**receipt['image_source_sha256'], **UNMODIFIED_SOURCES}
    text = pilot.command(['docker', 'run', '--rm', '--network=none', '--runtime=runc', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
                          '--entrypoint', 'sha256sum', IMAGE, *installed]).stdout
    if {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in text.splitlines()} != installed:
        raise ValueError('installed capture source identity differs')
    recipe_path = cold.PILOT / 'container-final.private.json'
    if pilot.sha(recipe_path) != cold.PINNED[recipe_path]:
        raise ValueError('historical serving recipe changed')
    recipe = json.loads(recipe_path.read_text())
    cold.validate_recipe(recipe, 'p8')
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {'backend': pilot.active('klc-backend.service', True), 'timer': pilot.active('klc-model-stack.timer')}
        record = {'schema': 'glm53-p8.index-trace-execution.v1', 'plan_sha256': digest, 'started_at': pilot.now(),
                  'prior': prior, 'exit_code': 1, 'repeats': [], 'teacher_logits_opened': False,
                  'protected_roles_opened': [], 'allocation_restart': False, 'speed_measurement_valid': False,
                  'adapter': 'scoped base.capture_stage adapter; both repeats canary N128, no full panel or quality stopping gate'}
        handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f'index trace interrupted by signal {signum}')
            for signum in base.INTERRUPT_SIGNALS:
                handlers[signum] = signal.signal(signum, interrupted)
            if prior['timer']:
                pilot.command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                pilot.command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            hardware = cold.inventory()
            pilot.save(output / 'hardware-inventory.json', hardware)
            ids = set()
            for index in REPEATS:
                out = output / repeat_name(index)
                with stage_adapter(plan, index):
                    stage = base.capture_stage(plan, {'stage': 'canary', 'arm': 'n128'}, recipe, out, digest, hardware)
                record['repeats'].append({'index': index, 'execution_sha256': pilot.sha(out / 'execution.json')})
                if stage['exit_code'] != 0:
                    raise RuntimeError('diagnostic capture failed; no retry')
                if stage['container_id'] in ids:
                    raise ValueError('repeats did not use fresh containers')
                ids.add(stage['container_id'])
            comparison = compare_repeats(plan)
            pilot.save(output / 'comparison.json', comparison)
            record.update(exit_code=0, diagnostic_complete=True, qualification_pass=False,
                          comparison_sha256=pilot.sha(output / 'comparison.json'))
        except BaseException as error:
            record['error_type'] = type(error).__name__
            pilot.private_save(output / 'error.private.txt', str(error))
        finally:
            for signum in handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                verify_identities(plan)
                record['final_identity_audit'] = {'ok': True}
            except Exception as error:
                record['final_identity_audit'] = {'ok': False, 'error_type': type(error).__name__}
                record['exit_code'] = 1
            record['restoration_safety'] = restoration_safety(digest)
            record['restoration'] = cold.restore_if_safe(prior, record['restoration_safety'])
            if record['restoration']['errors'] or any(record['restoration'][key] != value for key, value in prior.items()):
                record['exit_code'] = 1
            record['finished_at'] = pilot.now()
            pilot.save(output / 'execution.json', record)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    if record['exit_code']:
        raise RuntimeError('index trace failed; evidence preserved')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--source-preflight', type=Path, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.plan, args.output, args.source_preflight) if args.command == 'plan' else run(args.plan)))
