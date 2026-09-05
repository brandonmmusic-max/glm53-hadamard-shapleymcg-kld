"""Sealed V2 forced-M1 capture lifecycle; no GPU work on import or planning."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import socket
import stat
import sys
import threading
import time
from urllib.request import Request, urlopen

import numpy as np

from . import p8_decode_protocol as protocol
from . import p8_fc1_cold_compare as cold
from . import p8_fc1_integration as pilot
from .p8_weight_identity import fingerprint
from scripts import build_p8_decode_capture_image as builder

REPO = Path(__file__).resolve().parents[1]
ROOT = cold.ROOT
PREFIX = 'glm53-p8-forced-m1-v2-v2'
PRIOR_PLAN_SHA = 'e3332469bc2ed9aec86eabb86b8dc57f76123542e1de309b9136893e4014725a'
PRIOR_RUNTIME_MOUNT = '/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/runtime_patch:/runtime-patch:ro'
PRIOR_REPO = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1')
COLD_PLAN = PRIOR_REPO / 'experiments/p8-fc1-cold-comparison-v1.json'
COLD_PLAN_SHA = '77fd98c7af0c16a44101beb006a2fc26f0036e227ef1a89316e613ca26da1ec8'
PORT = 8023
INTERRUPT_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
RAW_CAPTURE_BYTES = 2 * 33 * protocol.ROWS * protocol.VOCAB_LIMIT * 4
COLD_UNIT = 'glm53-p8-fc1-cold-comparison-v1.service'
ARMS = {'n128': {'tile_n': 128, 'fused_scratch_zero': False},
        'n64': {'tile_n': 64, 'fused_scratch_zero': True}}
ORDER = [{'stage': stage, 'arm': arm} for stage in ('canary', 'full') for arm in ARMS]
SOURCE_FILES = cold.SOURCES | {
    'glm53_nvfp4/p8_decode_launcher.py', 'tests/test_p8_decode_launcher.py',
    'glm53_nvfp4/p8_decode_protocol.py', 'glm53_nvfp4/role_eval.py',
    'glm53_nvfp4/p8_decode_analysis.py', 'tests/test_p8_decode_analysis.py',
    'tests/test_p8_decode_analysis_receipts.py',
    'glm53_nvfp4/paired_role_analysis.py',
    'scripts/build_p8_decode_capture_image.py',
    'tests/test_build_p8_decode_capture_image.py', 'tests/test_p8_decode_capture_v2.py',
    *(f'runtime_patch/p8_decode_capture/{name}' for name in builder.SOURCE_NAMES),
}
FIXED = {'schema': 'glm53-p8.forced-m1-v2-plan.v2', 'port': PORT, 'order': ORDER, 'arms': ARMS,
         'amends_plan_sha256': PRIOR_PLAN_SHA,
         'amendment': 'startup-only lexical warmup scope; same numerical decision and conditional-fit role',
         'warmup_gate': 'four rank-tagged lexical scope closures; one registration and two samples per rank',
         'topology': {'tp': 4, 'ep': False, 'dcp': 1}, 'kv_cache_dtype': 'nvfp4_ds_mla',
         'max_num_seqs': 1, 'rows_per_window': 2047, 'full_windows': 32,
         'windows_per_domain': 8, 'canary_windows': 1, 'canary_selection': 'first role-manifest window',
         'canary_gate': 'bitwise exact full2047rows before either full-stage server starts',
         'cold_comparison_required': 'complete ten-run protocol; speed win is not a prerequisite',
         'thermal_start_max_c': 75, 'thermal_abort_c': 90,
         'request_timeout_seconds': 900, 'ready_timeout_seconds': 2400,
         'speed_measurement_valid': False, 'allocation_restart': False,
         'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
         'retry_policy': 'no retry or resume; preserve failure and require explicit amendment',
         'canary_is_domain_general_evidence': False}


def slot_name(entry):
    return f'{entry["stage"]}-{entry["arm"]}'


def model_name(arm):
    return f'{PREFIX}-{arm}'


def verify_image_receipt(path, image):
    value = json.loads(path.read_text())
    if (not re.fullmatch(r'sha256:[a-f0-9]{64}', image)
            or value.get('schema') != 'glm53-p8.decode-capture-image.v2' or value.get('status') != 'complete'
            or value.get('image_id') != image or value.get('parent_image_id') != cold.IMAGES['p8']
            or value.get('gpu_used') is not False or value.get('speed_measurement_valid') is not False
            or value.get('builder_sha256') != pilot.sha(REPO / 'scripts/build_p8_decode_capture_image.py')
            or set(value['source_sha256']) != set(builder.SOURCE_NAMES)):
        raise ValueError('dedicated capture image receipt differs')
    for name in builder.SOURCE_NAMES:
        if pilot.sha(builder.CONTEXT / name) != value['source_sha256'][name]:
            raise ValueError('capture package/build source differs')
    installed = value['image_source_sha256']
    if (any(installed[path] != builder.SAMPLER_PATCHED_SHA for path in builder.SAMPLERS)
            or any(installed[path] != builder.WARMUP_PATCHED_SHA for path in builder.WARMUPS)
            or installed[builder.INHERITED_FC2] != builder.INHERITED_FC2_SHA):
        raise ValueError('sampler patch or unchanged FC2 identity differs')
    for name in builder.SOURCE_NAMES:
        if name.endswith('.py') and installed[builder.PACKAGE + name] != value['source_sha256'][name]:
            raise ValueError('installed capture helper differs from mounted source')
    return value


def cold_prerequisite(plan_path):
    state = pilot.command(['systemctl', '--user', 'show', COLD_UNIT, '-p', 'ActiveState', '-p', 'Result']).stdout
    if set(state.strip().splitlines()) != {'ActiveState=inactive', 'Result=success'}:
        raise ValueError('cold comparison must be observably terminal success')
    if plan_path != COLD_PLAN or pilot.sha(plan_path) != COLD_PLAN_SHA:
        raise ValueError('original cold plan path or identity differs')
    declared = json.loads(plan_path.read_text())
    root = Path(declared['output'])
    # The historical verifier authenticates absolute prerequisite paths. Run
    # its unchanged, hash-verified source from the original worktree rather
    # than rewriting its plan to fit this new capture checkout.
    for relative, expected in declared['source_sha256'].items():
        source = PRIOR_REPO / relative
        if source != source.resolve() or not source.is_relative_to(PRIOR_REPO) or pilot.sha(source) != expected:
            raise ValueError('historical cold verifier source differs')
    replay = (
        'import sys,json; from pathlib import Path; '
        f'sys.path.insert(0, {str(PRIOR_REPO)!r}); '
        'from glm53_nvfp4.p8_fc1_cold_compare import analyze; '
        f'p=Path({str(plan_path)!r}); '
        'print(json.dumps(analyze(Path(json.loads(p.read_text())["output"]),p)))'
    )
    analysis = json.loads(pilot.command([sys.executable, '-I', '-B', '-c', replay]).stdout)
    if pilot.sha(plan_path) != COLD_PLAN_SHA:
        raise ValueError('historical cold plan changed during replay')
    if any(pilot.sha(PRIOR_REPO / relative) != expected for relative, expected in declared['source_sha256'].items()):
        raise ValueError('historical cold verifier source changed during replay')
    if analysis != json.loads((root / 'analysis.json').read_text()):
        raise ValueError('terminal cold comparison analysis does not replay')
    execution = json.loads((root / 'execution.json').read_text())
    # The strict cold analyzer checks all ten starts, metrics and restoration.
    if execution['exit_code'] != 0 or execution['completed_slots'] != 10:
        raise ValueError('a partial/failed cold protocol requires an amendment')
    return root, analysis


def approved_inputs(roles, teacher):
    windows = protocol.load_role_inputs(roles, teacher, verify_teacher_bytes=True)
    paths = {Path(roles)}
    for window in windows:
        paths.add(Path(window['token_path']))
        paths.add(Path(teacher) / window['teacher_path'])
    return windows, {str(path): fingerprint(path) for path in sorted(paths)}


def verify_input_stats(plan):
    for path, expected in plan['input_stats'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('approved role/token/teacher metadata changed after byte verification')


def verify_stage_identities(plan):
    """Cheap boundary checks for sources and mounted payloads, not teacher rehashing."""
    for relative, expected in plan['source_sha256'].items():
        source = REPO / relative
        if source != source.resolve() or not source.is_relative_to(REPO) or pilot.sha(source) != expected:
            raise ValueError('forced-M1 source identity changed between stages')
    cold_path = Path(plan['cold_plan'])
    if pilot.sha(cold_path) != plan['cold_plan_sha256']:
        raise ValueError('cold plan identity changed between stages')
    cold_plan = json.loads(cold_path.read_text())
    cold.weight_receipt(Path(cold_plan['weight_audit']), cold_plan['weight_audit_sha256'])
    verify_input_stats(plan)


def make_plan(path, output, image, image_receipt, roles, teacher_root, cold_plan):
    if any(p != p.resolve() for p in (path, output, image_receipt, roles, teacher_root, cold_plan)):
        raise ValueError('canonical planning paths required')
    if path.exists() or path.with_suffix('.sha256').exists() or output.exists() or output.parent != ROOT:
        raise ValueError('fresh plan/seal/output required')
    verify_image_receipt(image_receipt, image)
    cold_root, cold_analysis = cold_prerequisite(cold_plan)
    windows, stats = approved_inputs(roles, teacher_root)
    plan = {**copy.deepcopy(FIXED), 'created_at': pilot.now(), 'output': str(output), 'capture_image': image,
            'image_receipt': str(image_receipt), 'image_receipt_sha256': pilot.sha(image_receipt),
            'roles': str(roles), 'roles_sha256': protocol.ROLES_SHA256, 'teacher_root': str(teacher_root),
            'windows': windows, 'input_stats': stats, 'canary_window_id': windows[0]['id'],
            'cold_plan': str(cold_plan), 'cold_plan_sha256': pilot.sha(cold_plan),
            'cold_execution_sha256': pilot.sha(cold_root / 'execution.json'),
            'cold_analysis_sha256': pilot.sha(cold_root / 'analysis.json'),
            'cold_speed_gate_pass_observed': cold_analysis['speed_gate_pass'],
            'source_sha256': {relative: pilot.sha(REPO / relative) for relative in sorted(SOURCE_FILES)},
            'raw_capture_bytes': RAW_CAPTURE_BYTES,
            'capture_artifact_owner': {'uid': os.getuid(), 'gid': os.getgid()},
            'limits': ['Canary is one-window smoke, not domain-general evidence.',
                       'Full stage retains32 conditional-fit windows, eight per domain,2047 causalrows each.',
                       'One-token prefill is row0; true decode is rows1..2046. No protected roles are opened.',
                       'V2 pre-mask capture is outside the model CUDA graph; its timings are not speed measurements.',
                       'Full exact-logit comparison uses existing protocol; teacher KLD scoring is a separate analysis step.',
                       'No correctness or throughput result in this launcher authorizes allocation.']}
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), pilot.sha(path) + '  ' + path.name + '\n')
    return plan


def authenticate(path):
    plan = json.loads(path.read_text())
    if pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('forced-M1 plan seal differs')
    if any(plan.get(k) != v for k, v in FIXED.items()) or not SOURCE_FILES <= plan['source_sha256'].keys():
        raise ValueError('forced-M1 protocol differs')
    if plan.get('raw_capture_bytes') != RAW_CAPTURE_BYTES or Path(plan['output']).parent != ROOT:
        raise ValueError('forced-M1 capture size or output parent differs')
    for relative, expected in plan['source_sha256'].items():
        source = REPO / relative
        if source != source.resolve() or not source.is_relative_to(REPO) or pilot.sha(source) != expected:
            raise ValueError('forced-M1 source identity differs')
    for key in ('image_receipt', 'cold_plan', 'roles', 'teacher_root', 'output'):
        p = Path(plan[key])
        if p != p.resolve():
            raise ValueError('noncanonical prerequisite path')
    if (pilot.sha(Path(plan['image_receipt'])) != plan['image_receipt_sha256']
            or pilot.sha(Path(plan['cold_plan'])) != plan['cold_plan_sha256']
            or plan['roles_sha256'] != protocol.ROLES_SHA256
            or pilot.sha(Path(plan['roles'])) != protocol.ROLES_SHA256):
        raise ValueError('image/cold/role prerequisite identity differs')
    image_receipt = verify_image_receipt(Path(plan['image_receipt']), plan['capture_image'])
    cold_root, _ = cold_prerequisite(Path(plan['cold_plan']))
    if (pilot.sha(cold_root / 'execution.json') != plan['cold_execution_sha256']
            or pilot.sha(cold_root / 'analysis.json') != plan['cold_analysis_sha256']):
        raise ValueError('cold terminal receipt changed')
    windows, stats = approved_inputs(Path(plan['roles']), Path(plan['teacher_root']))
    if windows != plan['windows'] or stats != plan['input_stats'] or windows[0]['id'] != plan['canary_window_id']:
        raise ValueError('full approved role/teacher/token inventory changed')
    if plan['capture_artifact_owner'] != {'uid': os.getuid(), 'gid': os.getgid()}:
        raise ValueError('capture artifact reader identity differs from plan')
    return plan, image_receipt


def stage_windows(plan, stage):
    if stage not in ('canary', 'full'):
        raise ValueError('undeclared capture stage')
    return plan['windows'][:1] if stage == 'canary' else plan['windows']


def clone_argv(container, image, entry, out, windows, owner):
    cold.validate_recipe(container, 'p8')
    arm = ARMS[entry['arm']]
    config, host = container['Config'], container['HostConfig']
    tokens = shlex.split(config['Cmd'][1])
    for option, value in {'--port': str(PORT), '--served-model-name': model_name(entry['arm']),
                          '--max-num-seqs': '1'}.items():
        if tokens.count(option) != 1:
            raise ValueError('missing or duplicate explicit serving option')
        tokens[tokens.index(option) + 1] = value
    if '--logits-processors' in tokens:
        raise ValueError('V1/custom logits processor must not be configured')
    overrides = {'GLM53_P8_FC1_TILE_N': str(arm['tile_n']),
                 'GLM53_P8_FUSED_SCRATCH': '1' if arm['fused_scratch_zero'] else '',
                 'VLLM_USE_V2_MODEL_RUNNER': '1', 'GLM53_P8_DECODE_CAPTURE_V2': '1',
                 'GLM53_P8_DECODE_CAPTURE_ROOT': '/p8-captures',
                 'GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS': ','.join(w['id'] for w in windows),
                 'GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS': '2047'}
    if any(value.split('=', 1)[0].startswith('GLM53_P8_DECODE_CAPTURE') for value in config['Env']):
        raise ValueError('source recipe unexpectedly enables a capture hook')
    env = [value for value in config['Env'] if value.split('=', 1)[0] not in overrides]
    env.extend(f'{key}={value}' for key, value in overrides.items())
    argv = ['docker', 'create', '--name', f'{PREFIX}-{slot_name(entry)}', '--cidfile', str(out / 'container.cid'),
            '--label', f'{cold.LABEL}={owner}', '--network', 'host', '--ipc', 'host',
            '--shm-size', str(host['ShmSize']), '--gpus', 'all', '--runtime', host['Runtime'],
            '--restart', 'no', '--security-opt', 'label=disable', '--workdir', '/', '--entrypoint', '/bin/bash']
    for value in env:
        argv += ['--env', value]
    runtime_mounts = [bind for bind in host['Binds'] if bind.split(':')[1] == '/runtime-patch']
    if runtime_mounts != [PRIOR_RUNTIME_MOUNT]:
        raise ValueError('expected exactly one authenticated prior runtime mount')
    for bind in host['Binds']:
        if bind.split(':')[1] == '/p8-captures':
            raise ValueError('capture mount collision')
        # PYTHONPATH prefers this mount to site-packages. Bind the sealed new
        # worktree, otherwise the old helper masks the patched image package.
        replacement = f'{REPO / "runtime_patch"}:/runtime-patch:ro' if bind == PRIOR_RUNTIME_MOUNT else bind
        argv += ['--volume', replacement]
    return [*argv, '--volume', f'{out / "captures"}:/p8-captures:rw', image, '-lc', shlex.join(tokens)]


def runtime_audit(log, arm, completed_windows=None):
    configuration = ARMS[arm]
    proof = pilot.verify_runtime(log, configuration['tile_n'], configuration['fused_scratch_zero'])
    for marker in ('Using V2 Model Runner', 'tensor_parallel_size=4', 'decode_context_parallel_size=1',
                   'speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla'):
        if marker not in log:
            raise ValueError('V2 topology/graph serving receipt missing')
    if "'enable_expert_parallel': True" in log:
        raise ValueError('unexpected expert-parallel capture')
    ready = re.findall(r'GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=(\d+) max_num_reqs=(\d+) real_vocab=(\d+) expected_outputs=(\d+)', log)
    if len(ready) != 4 or set(ready) != {(str(rank), '1', str(protocol.VOCAB_LIMIT), '2047') for rank in range(4)}:
        raise ValueError('exactly four expected V2READY receipts required')
    closed = re.findall(r'GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=(\d+) registrations=(\d+) samples=(\d+)', log)
    if len(closed) != 4 or set(closed) != {(str(rank), '1', '2') for rank in range(4)}:
        raise ValueError('exactly four completed startup warmup scopes required before evaluation')
    result = {**proof, 'v2_ready_ranks': list(range(4)), 'warmup_closed_ranks': list(range(4)),
              'warmup_registrations_per_rank': 1, 'warmup_samples_per_rank': 2,
              'arm': arm, **configuration}
    if completed_windows is not None:
        completed = re.findall(r'GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)', log)
        if len(completed) != len(completed_windows) or set(completed) != {(w['id'], '2047', '0') for w in completed_windows}:
            raise ValueError('V2 completion log inventory differs from exact requested windows')
        result['complete_window_ids'] = [w['id'] for w in completed_windows]
    return result


def request_capture(request, tick, timeout):
    """Keep the lifecycle thread sampling thermals during a blocking HTTP call."""
    replies = queue.Queue(maxsize=1)
    def worker():
        try:
            payload = json.dumps(request).encode()
            query = Request(f'http://127.0.0.1:{PORT}/v1/completions', payload, {'Content-Type': 'application/json'})
            with urlopen(query, timeout=timeout) as response:
                replies.put((True, json.load(response)))
        except BaseException as error:
            replies.put((False, error))
    threading.Thread(target=worker, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        tick()
        try:
            ok, value = replies.get(timeout=min(2, max(.01, deadline - time.monotonic())))
        except queue.Empty:
            if time.monotonic() >= deadline:
                raise RuntimeError('forced capture request timeout')
            continue
        if not ok:
            raise value
        return value


CAPTURE_SUFFIXES = ('capture.json', 'logits.f32', 'capture.inprogress.json',
                    'capture.failed.json', 'capture.json.partial', 'logits.f32.partial')


def normalize_capture_ownership(root, windows, cid, image, name, owner, target, command=pilot.command):
    """Change ownership only, on explicit regular artifacts in this fresh root."""
    if root != root.resolve() or not root.is_dir():
        raise ValueError('capture ownership root is not canonical')
    container = json.loads(command(['docker', 'inspect', cid]).stdout)[0]
    if (container['Id'] != cid or not re.fullmatch('[a-f0-9]{64}', cid)
            or container['Name'] != '/' + name or container['Image'] != image
            or container['Config'].get('Labels', {}).get(cold.LABEL) != owner
            or not container['State']['Running']):
        raise ValueError('live capture container ownership did not authenticate')
    rows, paths = [], []
    for window in windows:
        if not re.fullmatch(r'conditional-fit-\d{4}', window['id']):
            raise ValueError('nonapproved capture window name')
        for suffix in CAPTURE_SUFFIXES:
            path = root / f'{window["id"]}.{suffix}'
            try:
                before = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(before.st_mode) or path.resolve() != path:
                raise ValueError('capture normalization refuses symlink/nonregular file')
            rows.append({'window': window['id'], 'path': str(path),
                         'container_path': '/p8-captures/' + path.name,
                         'prior_stat': {'uid': before.st_uid, 'gid': before.st_gid,
                                        'mode': stat.S_IMODE(before.st_mode), 'bytes': before.st_size,
                                        'inode': before.st_ino, 'device': before.st_dev, 'mtime_ns': before.st_mtime_ns}})
            paths.append('/p8-captures/' + path.name)
    if paths:
        command(['docker', 'exec', '--user', '0:0', cid, 'chown', '-h', f'{target["uid"]}:{target["gid"]}', '--', *paths])
    for row in rows:
        after = Path(row['path']).lstat()
        before = row['prior_stat']
        if (not stat.S_ISREG(after.st_mode) or after.st_uid != target['uid'] or after.st_gid != target['gid']
                or (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns)
                != (before['inode'], before['device'], before['bytes'], before['mtime_ns'])):
            raise ValueError('capture ownership or content metadata changed unexpectedly')
        row['normalized_uid'], row['normalized_gid'] = after.st_uid, after.st_gid
    return {'container_id': cid, 'image': image, 'owner_label': owner,
            'target_uid': target['uid'], 'target_gid': target['gid'], 'artifacts': rows,
            'operation': 'chown -h explicit regular capture files; no bytes modified'}


def capture_stage(plan, entry, container, out, plan_hash, hardware):
    windows = stage_windows(plan, entry['stage'])
    out.mkdir(mode=0o700)
    (out / 'captures').mkdir(mode=0o700)
    (out / 'requests').mkdir(mode=0o700)
    name, owner = f'{PREFIX}-{slot_name(entry)}', f'{plan_hash}:{slot_name(entry)}'
    record = {**entry, 'schema': 'glm53-p8.forced-m1-v2-stage.v1', 'started_at': pilot.now(),
              'plan_sha256': plan_hash, 'capture_image': plan['capture_image'], 'exit_code': 1,
              'windows': [], 'ownership_normalizations': [], 'protected_roles_opened': [], 'allocation_restart': False}
    cid = None
    try:
        verify_stage_identities(plan)
        if pilot.command(['docker', 'ps', '-a', '--filter', f'name=^/{name}$', '--format', '{{.ID}}']).stdout.strip():
            raise ValueError('capture container name already exists')
        with socket.socket() as port:
            port.bind(('127.0.0.1', PORT))
        deadline = time.monotonic() + 1800
        with (out / 'cooldown.jsonl').open('x') as cooldown:
            while True:
                temperatures = pilot.temperatures()
                cooldown.write(json.dumps({'at': pilot.now(), 'temperatures': temperatures}) + '\n'); cooldown.flush()
                if max(temperatures) <= 75:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('capture startup cooling timeout')
                time.sleep(10)
        if cold.inventory() != hardware or pilot.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip():
            raise RuntimeError('GPU inventory changed or another workload is active')
        pilot.save(out / 'nvidia-before.xml', pilot.command(['nvidia-smi', '-q', '-x']).stdout)
        argv = clone_argv(container, plan['capture_image'], entry, out, windows, owner)
        pilot.private_save(out / 'launch-spec.private.json', {'argv': argv})
        verify_stage_identities(plan)
        cid = pilot.command(argv).stdout.strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise RuntimeError('invalid capture container ID')
        record['container_id'] = cid
        pilot.command(['docker', 'start', cid])
        pilot.private_save(out / 'container.private.json', pilot.command(['docker', 'inspect', cid]).stdout)
        with (out / 'thermal.jsonl').open('x') as thermal:
            def tick():
                values = pilot.temperatures()
                thermal.write(json.dumps({'at': pilot.now(), 'temperatures': values}) + '\n'); thermal.flush()
                if max(values) >= 90:
                    raise RuntimeError('capture thermal abort at or above90C')
                if not json.loads(pilot.command(['docker', 'inspect', '-f', '{{json .State}}', cid]).stdout)['Running']:
                    raise RuntimeError('capture server exited')
            deadline = time.monotonic() + plan['ready_timeout_seconds']
            while time.monotonic() < deadline:
                tick()
                try:
                    with urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if [m['id'] for m in models['data']] == [model_name(entry['arm'])]:
                        pilot.save(out / 'models.json', models)
                        break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('V2 capture readiness timeout')
            logs = pilot.command(['docker', 'logs', cid])
            ready_log = logs.stdout + logs.stderr
            pilot.private_save(out / 'server-ready.private.log', ready_log)
            pilot.save(out / 'runtime-ready-audit.json', runtime_audit(ready_log, entry['arm']))
            for window in windows:
                token_path = Path(window['token_path'])
                if pilot.sha(token_path) != window['input_sha256']:
                    raise ValueError('approved request token file changed')
                tokens = np.load(token_path, allow_pickle=False)
                request = protocol.completion_request(tokens, window['id'], model_name(entry['arm']))
                pilot.private_save(out / 'requests' / (window['id'] + '.request.json'), request)
                response = request_capture(request, tick, plan['request_timeout_seconds'])
                pilot.private_save(out / 'requests' / (window['id'] + '.response.json'), response)
                response_audit = protocol.verify_response(response, request)
                record['ownership_normalizations'].append(normalize_capture_ownership(
                    out / 'captures', [window], cid, plan['capture_image'], name, owner, plan['capture_artifact_owner']))
                array, metadata = protocol.load_capture(out / 'captures', window['id'], tokens)
                del array
                record['windows'].append({'id': window['id'], 'domain': window['domain'],
                    'response_audit': response_audit, 'capture_sha256': pilot.sha(out / 'captures' / (window['id'] + '.capture.json')),
                    'raw_sha256': metadata['raw_sha256'], 'rows': metadata['rows_completed']})
                tick()
        record['exit_code'] = 0
    except BaseException as error:
        record['error_type'] = type(error).__name__
        pilot.private_save(out / 'error.private.txt', str(error))
    finally:
        handlers = {s: signal.signal(s, signal.SIG_IGN) for s in INTERRUPT_SIGNALS}
        if cid is not None and record['exit_code'] != 0:
            try:
                record['ownership_normalizations'].append(normalize_capture_ownership(
                    out / 'captures', windows, cid, plan['capture_image'], name, owner, plan['capture_artifact_owner']))
            except Exception as error:
                record['partial_ownership_normalization_error_type'] = type(error).__name__
        try:
            ok, errors = cold.cleanup_owned(out, name, plan['capture_image'], owner)
            record['cleanup'] = {'ok': ok, 'errors': errors}
            if not ok:
                record['exit_code'] = 1
        except BaseException as error:
            record['cleanup'] = {'ok': False, 'error_type': type(error).__name__}
            record['exit_code'] = 1
        try:
            verify_stage_identities(plan)
            if cold.inventory() != hardware:
                raise RuntimeError('capture GPU identity/power changed')
            pilot.save(out / 'nvidia-after.xml', pilot.command(['nvidia-smi', '-q', '-x']).stdout)
            if record['exit_code'] == 0:
                final = runtime_audit((out / 'server-final.private.log').read_text(), entry['arm'], windows)
                pilot.save(out / 'runtime-final-audit.json', final)
        except Exception as error:
            record['final_audit_error_type'] = type(error).__name__
            record['exit_code'] = 1
        record['finished_at'] = pilot.now()
        record['files'], record['unreadable_artifacts'] = {}, []
        for p in out.rglob('*'):
            if not p.is_file():
                continue
            try:
                record['files'][str(p.relative_to(out))] = {'bytes': p.stat().st_size, 'sha256': pilot.sha(p)}
            except Exception as error:
                record['unreadable_artifacts'].append({'path': str(p.relative_to(out)), 'error_type': type(error).__name__})
                record['exit_code'] = 1
        pilot.save(out / 'execution.json', record)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return record


def compare_stage(plan, root, stage):
    rows = []
    for window in stage_windows(plan, stage):
        tokens = np.load(window['token_path'], allow_pickle=False)
        a, reference_metadata = protocol.load_capture(root / f'{stage}-n128/captures', window['id'], tokens)
        b, candidate_metadata = protocol.load_capture(root / f'{stage}-n64/captures', window['id'], tokens)
        for key in ('original_logit_dtype', 'original_logit_width'):
            if reference_metadata[key] != candidate_metadata[key]:
                raise ValueError('paired capture native logit representation differs')
        rows.append({'window_id': window['id'], 'domain': window['domain'],
                     'original_logit_dtype': reference_metadata['original_logit_dtype'],
                     'original_logit_width': reference_metadata['original_logit_width'],
                     **protocol.exact_logits(a, b)})
        del a, b
    return {'schema': 'glm53-p8.forced-m1-v2-stage-exact.v1', 'stage': stage,
            'all_exact': all(row['exact'] for row in rows), 'windows': rows,
            'canary_is_domain_general_evidence': False, 'allocation_restart': False}


def restoration_safety(plan_hash, image, command=pilot.command):
    record = {'ok': False, 'containers': [], 'errors': []}
    try:
        expected = {f'/{PREFIX}-{slot_name(e)}': e for e in ORDER}
        ids = command(['docker', 'ps', '-a', '--no-trunc', '--filter', f'name=^/{PREFIX}-', '--format', '{{.ID}}']).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise ValueError('unresolved container identity')
            c = json.loads(command(['docker', 'inspect', cid]).stdout)[0]
            entry = expected.get(c['Name'])
            if (entry is None or c['Id'] != cid or c['Image'] != image
                    or c['Config'].get('Labels', {}).get(cold.LABEL) != f'{plan_hash}:{slot_name(entry)}'):
                raise ValueError('capture ownership uncertain')
            record['containers'].append({'id': cid, 'name': c['Name'], 'running': c['State']['Running']})
            if c['State']['Running']:
                raise RuntimeError('capture container still live')
        record['ok'] = True
    except Exception as error:
        record['errors'].append(type(error).__name__)
    return record


def run(path):
    plan, image_receipt = authenticate(path)
    output, image = Path(plan['output']), plan['capture_image']
    if output.exists() or output.parent != ROOT or output != output.resolve():
        raise ValueError('fresh canonical capture output required')
    if pilot.command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip():
        raise ValueError('clean checkout required')
    state = pilot.command(['systemctl', '--user', 'show', COLD_UNIT, '-p', 'ActiveState', '-p', 'Result']).stdout
    if set(state.strip().splitlines()) != {'ActiveState=inactive', 'Result=success'}:
        raise ValueError('prior cold comparison unit is not terminal success')
    if shutil.disk_usage(ROOT).free < plan['raw_capture_bytes'] + 20 * 2**30:
        raise ValueError('insufficient disk for all66 full-window captures plus20GiB margin')
    if pilot.command(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']).stdout.strip() != image:
        raise ValueError('dedicated immutable capture image missing')
    installed = image_receipt['image_source_sha256']
    text = pilot.command(['docker', 'run', '--rm', '--network=none', '--runtime', 'runc', '--entrypoint', 'sha256sum', image, *installed]).stdout
    observed = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in text.splitlines()}
    if observed != installed:
        raise ValueError('installed capture image sources differ from receipt')
    container = json.loads((cold.PILOT / 'container-final.private.json').read_text())
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {'backend': pilot.active('klc-backend.service', True), 'timer': pilot.active('klc-model-stack.timer')}
        record = {'schema': 'glm53-p8.forced-m1-v2-execution.v1', 'plan_sha256': pilot.sha(path),
                  'started_at': pilot.now(), 'prior': prior, 'exit_code': 1, 'stages': [],
                  'protected_roles_opened': [], 'opened_roles': ['conditional-fit'],
                  'allocation_restart': False, 'speed_measurement_valid': False,
                  'capture_image': image, 'source_commit': pilot.command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).stdout.strip()}
        old_signals = {}
        try:
            def interrupt(signum, frame):
                raise RuntimeError(f'forced-M1 capture interrupted by signal{signum}')
            for s in INTERRUPT_SIGNALS:
                old_signals[s] = signal.signal(s, interrupt)
            if prior['timer']:
                pilot.command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                pilot.command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            hardware = cold.inventory()
            pilot.save(output / 'hardware-inventory.json', hardware)
            for stage in ('canary', 'full'):
                for arm in ARMS:
                    entry = {'stage': stage, 'arm': arm}
                    receipt = capture_stage(plan, entry, container, output / slot_name(entry), pilot.sha(path), hardware)
                    record['stages'].append({'slot': slot_name(entry), 'execution_sha256': pilot.sha(output / slot_name(entry) / 'execution.json')})
                    if receipt['exit_code'] != 0:
                        raise RuntimeError('capture stage failed; no retry or advance')
                closure = compare_stage(plan, output, stage)
                pilot.save(output / f'{stage}-exact.json', closure)
                if stage == 'canary' and not closure['all_exact']:
                    raise RuntimeError('canary failed bitwise closure; full stage forbidden')
                if stage == 'full':
                    record['full_exact'] = closure['all_exact']
            record['capture_protocol_complete'] = True
            record['exit_code'] = 0
        except BaseException as error:
            record['error_type'] = type(error).__name__
            pilot.private_save(output / 'error.private.txt', str(error))
        finally:
            for s in old_signals:
                signal.signal(s, signal.SIG_IGN)
            try:
                verify_stage_identities(plan)
                record['final_identity_audit'] = {'ok': True}
            except Exception as error:
                record['final_identity_audit'] = {'ok': False, 'error_type': type(error).__name__}
                record['exit_code'] = 1
            record['restoration_safety'] = restoration_safety(pilot.sha(path), image)
            record['restoration'] = cold.restore_if_safe(prior, record['restoration_safety'])
            if record['restoration']['errors'] or any(record['restoration'][key] != value for key, value in prior.items()):
                record['exit_code'] = 1
            record['finished_at'] = pilot.now()
            pilot.save(output / 'execution.json', record)
            for s, handler in old_signals.items():
                signal.signal(s, handler)
    if record['exit_code']:
        raise RuntimeError('forced-M1 protocol failed; partial evidence preserved')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    for name in ('plan', 'output', 'image-receipt', 'roles', 'teacher-root', 'cold-plan'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--image', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = make_plan(args.plan, args.output, args.image, args.image_receipt, args.roles, args.teacher_root, args.cold_plan)
    else:
        result = run(args.plan)
    print(json.dumps(result, sort_keys=True))
