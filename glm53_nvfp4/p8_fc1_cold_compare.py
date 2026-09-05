"""Sealed five-new-process-per-arm P8/EXL3 speed-only comparison.

The plan command reads artifacts and writes a new plan/seal; only the explicit
run command can launch serving. Importing performs no external operations.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.request import urlopen

from . import p8_fc1_integration as pilot
from . import p8_weight_identity as weights

REPO = Path(__file__).resolve().parents[1]
ROOT = pilot.ROOT
PILOT = ROOT / 'fc1-integrated-v1'
PILOT_PLAN = REPO / 'experiments/p8-fc1-integrated-v1.json'
EXL3_ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1/speed-v2a3')
PORT = 8022
PREFIX = 'glm53-p8-fc1-cold-v1'
LABEL = 'org.klc.p8-fc1-cold-plan-slot'
IMAGES = {'p8': 'sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141',
          'exl3': 'sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf'}
TOPOLOGY = {'p8': {'tp': 4, 'ep': False, 'dcp': 1},
            'exl3': {'tp': 4, 'ep': True, 'dcp': 4}}
ORDER = [{'round': r, 'arm': arm} for r in range(1, 6)
         for arm in (('exl3', 'p8') if r % 2 else ('p8', 'exl3'))]
PINNED = {
    PILOT_PLAN: '298e2eca8d5ba71c7834a016302eaf3e47e84553cf76fd1737ce9bd9e3fe99d6',
    PILOT / 'execution.json': '361bffa4664b89617b3d6cd5becc81888873cf7a6def6e9fc5902c8e8ae4fb2c',
    PILOT / 'container-final.private.json': 'c7f5832af2244926ebf0f2d481efbe7461bee85afb18cc4d85dd3769d44ffdf8',
    EXL3_ROOT / 'round-01-exl3/container.json': '5c0e4de4b4494be7a4a65ad349c62ca54c97e4d65cef08311ec6821cc3d634d9',
    EXL3_ROOT / 'round-01-exl3/receipt.json': '10350534707eb505f8cfeaf04033095d282bcec231af202842c7af111f81dcba',
    EXL3_ROOT / 'execution.json': 'a21fb65279564519ac674731713c83dd60d8a980c88b944177b7e2d9dbb674fe',
    EXL3_ROOT / 'analysis.json': '133c8da65f3b73a2965a4151f077db5021c7cc1296bdf4d9b1142d9fa97dfca1',
    Path('/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw/model.safetensors.index.json'):
        '2f64d21c67c90bbafeb36c4e9b2f06f54063ed439e9f7cf95962d425a1d8515d',
    Path('/home/brandonmusic/KLC_SANDBOXES/glm-5.3-flash-exl3-4bpw-release/results/materialization-receipt.json'):
        'afe588284702c0676b7af5df48bc0e0568bb42b1821808ab71c6dfa1f0c48b61',
}
SOURCES = pilot.REQUIRED_SOURCES | {'glm53_nvfp4/p8_fc1_cold_compare.py', 'tests/test_p8_fc1_cold_compare.py',
                                    'glm53_nvfp4/p8_weight_identity.py'}
FIXED = {'schema': 'glm53-p8-fc1-cold-comparison-plan.v1', 'port': PORT,
         'cold_runs_per_arm': 5, 'order': ORDER, 'images': IMAGES, 'topology': TOPOLOGY,
         'thermal_start_max_c': 75, 'thermal_abort_c': 90, 'contexts': [32768],
         'concurrency': 1, 'duration_seconds': 20, 'prefill_duration_seconds': 20,
         'decode_warmup_seconds': 3, 'max_tokens': 4096,
         'primary_decode': 'openai_continuous_usage',
         'primary_prefill': 'uncontaminated_prometheus_server_validation',
         'compiled_caches': 'reuse exact existing warm paths; never clear',
         'experimental_unit': 'independently started server container and worker processes',
         'decision': 'both P8 medians strictly greater than corresponding EXL3 medians',
         'failure_policy': 'stop whole attempt; preserve all partial runs; no retry or resume',
         'numerical_closure_required_separately': True, 'allocation_restart': False,
         'protected_roles_opened': [], 'speed_only': True}


def model_name(arm):
    return f'{PREFIX}-{arm}'


def slot_name(entry):
    return f'round-{entry["round"]:02d}-{entry["arm"]}'


def recipes():
    """Authenticate recipe identities without exposing environment values."""
    for path, expected in PINNED.items():
        if pilot.sha(path) != expected:
            raise ValueError(f'pinned prerequisite differs: {path.name}')
    pilot_plan, _ = pilot.authenticate(PILOT_PLAN)
    execution = json.loads((PILOT / 'execution.json').read_text())
    if (execution['exit_code'] != 0 or execution['image'] != IMAGES['p8']
            or execution['tile_n'] != 64 or execution['fused_scratch_zero'] is not True
            or not execution['cleanup']['ok'] or execution['restoration']['errors']
            or any(execution['restoration'][key] != value for key, value in execution['prior'].items())):
        raise ValueError('pilot did not pass and restore services')
    for name in ('result.json', 'summary.json', 'runtime-audit.json', 'server-final.log'):
        if pilot.sha(PILOT / name) != execution['files'][name]['sha256']:
            raise ValueError('pilot output identity differs')
    if pilot.summarize(json.loads((PILOT / 'result.json').read_text())) != json.loads((PILOT / 'summary.json').read_text()):
        raise ValueError('pilot summary does not replay')
    if pilot.verify_runtime((PILOT / 'server-final.log').read_text(), 64, True) != json.loads((PILOT / 'runtime-audit.json').read_text()):
        raise ValueError('pilot runtime audit does not replay')
    receipt = json.loads((EXL3_ROOT / 'round-01-exl3/receipt.json').read_text())
    if receipt['arm'] != 'exl3' or receipt['round'] != 1 or receipt['protected_roles_opened'] != []:
        raise ValueError('historical EXL3 receipt differs')
    for name in ('container.json', 'server.log', 'benchmark.json'):
        if pilot.sha(EXL3_ROOT / 'round-01-exl3' / name) != receipt['files'][name]['sha256']:
            raise ValueError('historical EXL3 artifact differs')
    result = {'p8': json.loads((PILOT / 'container-final.private.json').read_text()),
              'exl3': json.loads((EXL3_ROOT / 'round-01-exl3/container.json').read_text())[0]}
    for arm, container in result.items():
        validate_recipe(container, arm)
    return result, pilot_plan


def validate_recipe(container, arm):
    host, config = container['HostConfig'], container['Config']
    if (container['Image'] != IMAGES[arm] or config['Entrypoint'] != ['/bin/bash']
            or config['WorkingDir'] != '/' or len(config['Cmd']) != 2 or config['Cmd'][0] != '-lc'
            or any(c in config['Cmd'][1] for c in '$`\n\r;|&<>()')
            or host['NetworkMode'] != 'host' or host['IpcMode'] != 'host' or host['Privileged']
            or host['SecurityOpt'] != ['label=disable']):
        raise ValueError('recipe executable or host regime differs')
    tokens = shlex.split(config['Cmd'][1])
    if tokens[:6] != ['exec', '/opt/venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', '/model']:
        raise ValueError('unexpected executable')
    for option, value in {'--tensor-parallel-size': '4', '--decode-context-parallel-size': str(TOPOLOGY[arm]['dcp']),
                          '--attention-backend': 'B12X_MLA_SPARSE', '--kv-cache-dtype': 'nvfp4_ds_mla',
                          '--max-num-seqs': '1', '--max-num-batched-tokens': '2048',
                          '--quantization': 'modelopt' if arm == 'p8' else 'exl3'}.items():
        if tokens.count(option) != 1 or tokens[tokens.index(option) + 1] != value:
            raise ValueError('recipe topology/backend differs')
    if ('--enable-expert-parallel' in tokens) != TOPOLOGY[arm]['ep'] or any('speculat' in token or token == '--enforce-eager' for token in tokens):
        raise ValueError('recipe EP/speculation/graph regime differs')
    env = dict(value.split('=', 1) for value in config['Env'])
    if len(env) != len(config['Env']) or env.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('recipe environment/GPU inventory differs')
    if arm == 'p8' and any(env.get(k) != v for k, v in {
        'GLM53_P8_NATIVE': '1', 'GLM53_P8_SMALL_M': '1',
        'GLM53_P8_FC1_TILE_N': '64', 'GLM53_P8_FUSED_SCRATCH': '1'}.items()):
        raise ValueError('recipe narrow/fused dispatch differs')


def weight_receipt(path, expected=None):
    if path != path.resolve() or (expected is not None and pilot.sha(path) != expected):
        raise ValueError('fresh payload receipt identity differs')
    receipt = json.loads(path.read_text())
    if (receipt['protected_roles_opened'] != []
            or receipt['manifests'] != {str(weights.P8_MANIFEST): weights.P8_SHA, str(weights.EXL3_RECEIPT): weights.EXL3_SHA}
            or any(row['sha256'] != row['expected_sha256'] for row in receipt['files'])
            or len({row['path'] for row in receipt['files']}) != 288
            or {arm: sum(row['arm'] == arm for row in receipt['files']) for arm in ('p8', 'exl3')} != {'p8': 168, 'exl3': 120}):
        raise ValueError('fresh payload receipt completeness differs')
    weights.verify_stats(receipt)
    return receipt


def make_plan(path, output, weight_audit):
    if path.exists() or path.with_suffix('.sha256').exists() or output.exists():
        raise ValueError('fresh plan, seal, and output paths required')
    if output != output.resolve() or output.parent != ROOT or path != path.resolve():
        raise ValueError('canonical paths required')
    source, _ = recipes()
    weight_receipt(weight_audit)
    plan = {**copy.deepcopy(FIXED), 'created_at': pilot.now(), 'output': str(output),
            'prerequisite_sha256': {str(p): value for p, value in PINNED.items()},
            'source_sha256': {relative: pilot.sha(REPO / relative) for relative in sorted(SOURCES)},
            'benchmark_sha256': pilot.sha(pilot.BENCH),
            'weight_audit': str(weight_audit), 'weight_audit_sha256': pilot.sha(weight_audit),
            'cache_mounts': {arm: [bind for bind in c['HostConfig']['Binds'] if bind.split(':')[1] == '/cache']
                             for arm, c in source.items()},
            'limits': ['Different images and TP4/no-EP/DCP1 versus TP4/EP4/DCP4 product topologies; not kernel causality.',
                       'Five cold processes per arm; requests and thermal samples are subsamples.',
                       'No pilot or historical run is included in the ten-run primary comparison.',
                       'Fresh bytes cover168P8 routed sidecars and120EXL3shards; P8carrier uses prior provenance, not fresh rehash.',
                       'Speed qualification cannot establish numerical closure or authorize allocation.']}
    pilot.save(path, plan)
    pilot.save(path.with_suffix('.sha256'), pilot.sha(path) + '  ' + path.name + '\n')
    return plan


def authenticate(path):
    plan = json.loads(path.read_text())
    if pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('comparison seal differs')
    if any(plan.get(k) != v for k, v in FIXED.items()):
        raise ValueError('comparison protocol differs')
    if plan['prerequisite_sha256'] != {str(p): value for p, value in PINNED.items()}:
        raise ValueError('prerequisite inventory differs')
    if not SOURCES <= plan['source_sha256'].keys():
        raise ValueError('missing source identities')
    for relative, expected in plan['source_sha256'].items():
        source = REPO / relative
        if source != source.resolve() or not source.is_relative_to(REPO) or pilot.sha(source) != expected:
            raise ValueError('source identity differs')
    if pilot.sha(pilot.BENCH) != plan['benchmark_sha256']:
        raise ValueError('benchmark source differs')
    source, _ = recipes()
    weight_receipt(Path(plan['weight_audit']), plan['weight_audit_sha256'])
    caches = {arm: [bind for bind in c['HostConfig']['Binds'] if bind.split(':')[1] == '/cache']
              for arm, c in source.items()}
    if caches != plan['cache_mounts'] or any(len(values) != 1 for values in caches.values()):
        raise ValueError('warm cache identity differs')
    return plan, source


def clone_argv(container, arm, entry, out, owner):
    validate_recipe(container, arm)
    tokens = shlex.split(container['Config']['Cmd'][1])
    tokens[tokens.index('--port') + 1] = str(PORT)
    tokens[tokens.index('--served-model-name') + 1] = model_name(arm)
    host = container['HostConfig']
    argv = ['docker', 'create', '--name', f'{PREFIX}-{slot_name(entry)}', '--cidfile', str(out / 'container.cid'),
            '--label', f'{LABEL}={owner}', '--network', 'host', '--ipc', 'host', '--shm-size', str(host['ShmSize']),
            '--gpus', 'all', '--runtime', host['Runtime'], '--restart', 'no', '--security-opt', 'label=disable',
            '--workdir', '/', '--entrypoint', '/bin/bash']
    for value in container['Config']['Env']:
        argv += ['--env', value]
    for bind in host['Binds']:
        argv += ['--volume', bind]
    return [*argv, IMAGES[arm], '-lc', shlex.join(tokens)]


def benchmark_argv(out, arm):
    argv = pilot.benchmark_argv(out)
    argv[argv.index('--port') + 1] = str(PORT)
    argv[argv.index('--model') + 1] = model_name(arm)
    return argv


def summarize(result, arm):
    if result['metadata']['model'] != model_name(arm) or result['metadata']['server'] != f'127.0.0.1:{PORT}':
        raise ValueError('benchmark endpoint identity differs')
    normalized = copy.deepcopy(result)
    normalized['metadata'].update(model=pilot.NAME, server=f'127.0.0.1:{pilot.PORT}')
    common = pilot.summarize(normalized)
    prefill = result['prefill']['32768']
    server = prefill.get('server_validation', {})
    if (server.get('method') != 'prometheus' or server.get('invalid_reason') != ''
            or server.get('token_source') != 'kv_computed' or server.get('cached_tokens') != 0
            or type(server.get('samples')) is not int or server['samples'] <= 0
            or server['samples'] != prefill['samples']
            or any(not 32768 * .95 <= server.get(key, 0) <= 32768 * 1.05
                   for key in ('prompt_tokens', 'request_prompt_tokens'))
            or not server['prompt_tokens'] == server['request_prompt_tokens'] == prefill['prompt_tokens']):
        raise ValueError('uncontaminated server prefill validation missing')
    tps, seconds = float(server.get('tok_per_sec', 0)), float(server.get('prefill_seconds', 0))
    if not all(math.isfinite(v) and v > 0 for v in (tps, seconds)):
        raise ValueError('invalid primary prefill metric')
    return {'arm': arm, 'decode_tokens_per_second': common['decode_c1_tokens_per_second'],
            'prefill_server_tokens_per_second': tps, 'prefill_client_tokens_per_second': prefill['client_tok_per_sec'],
            'decode_source': 'openai_continuous_usage', 'prefill_source': 'prometheus/kv_computed',
            'allocation_restart': False, 'speed_only': True}


def runtime_audit(log, arm):
    for required in ('tensor_parallel_size=4', f'decode_context_parallel_size={TOPOLOGY[arm]["dcp"]}',
                     'speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla'):
        if required not in log:
            raise ValueError('runtime topology/speculation/KV receipt missing')
    if ("'enable_expert_parallel': True" in log) != TOPOLOGY[arm]['ep']:
        raise ValueError('runtime expert parallel receipt differs')
    if arm == 'p8':
        proof = pilot.verify_runtime(log, 64, True)
    else:
        for required in ('EXL3 full-expert EP runtime planned', 'quantization=exl3',
                         'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts'):
            if required not in log:
                raise ValueError('EXL3 backend receipt missing')
        proof = {'graph': pilot.verify_graphs(log)}
    return {'arm': arm, 'topology': TOPOLOGY[arm], **proof}


def cleanup_owned(out, name, image, owner, run=pilot.command):
    errors = []
    found = run(['docker', 'inspect', name], check=False)
    if found.returncode:
        listing = run(['docker', 'ps', '-a', '--filter', f'name=^/{name}$', '--format', '{{.ID}}'])
        return not listing.stdout.strip(), ['container not inspectable'] if listing.stdout.strip() else []
    container = json.loads(found.stdout)[0]
    cid = container['Id']
    if (not re.fullmatch('[a-f0-9]{64}', cid) or container['Name'] != '/' + name
            or container['Image'] != image or container['Config'].get('Labels', {}).get(LABEL) != owner):
        return False, ['container ownership did not authenticate']
    for action in (['stop', '--time', '30'], ['kill']):
        if not container['State']['Running']:
            break
        try:
            run(['docker', *action, cid], check=False, timeout=45)
            container = json.loads(run(['docker', 'inspect', cid]).stdout)[0]
        except Exception as error:
            errors.append(type(error).__name__)
    for filename, read in (
        ('server-final.private.log', lambda: run(['docker', 'logs', cid], check=False)),
        ('container-final.private.json', lambda: container),
    ):
        try:
            value = read()
            pilot.private_save(out / filename, value.stdout + value.stderr if filename.endswith('.log') else value)
        except Exception as error:
            errors.append(filename + ':' + type(error).__name__)
    if container['State']['Running']:
        return False, [*errors, 'owned container still running']
    try:
        run(['docker', 'rm', cid])
    except Exception as error:
        errors.append('remove:' + type(error).__name__)
    return not errors, errors


def inventory():
    text = pilot.command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,power.limit', '--format=csv,noheader']).stdout
    lines = text.strip().splitlines()
    if len(lines) != 4 or [line.split(',')[0].strip() for line in lines] != ['0', '1', '2', '3']:
        raise ValueError('physical GPU inventory differs')
    return lines


def restoration_safety(plan_hash, run=pilot.command):
    """Fail closed if any comparison container is live or cannot be identified."""
    record = {'ok': False, 'containers': [], 'errors': []}
    try:
        listing = run(['docker', 'ps', '-a', '--no-trunc', '--filter', f'name=^/{PREFIX}-round-', '--format', '{{.ID}}'])
        expected = {f'/{PREFIX}-{slot_name(e)}': e for e in ORDER}
        for cid in listing.stdout.splitlines():
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise ValueError('invalid container identity in restoration check')
            container = json.loads(run(['docker', 'inspect', cid]).stdout)[0]
            entry = expected.get(container['Name'])
            if (entry is None or container['Id'] != cid or container['Image'] != IMAGES[entry['arm']]
                    or container['Config'].get('Labels', {}).get(LABEL) != f'{plan_hash}:{slot_name(entry)}'):
                raise ValueError('comparison container ownership uncertain')
            record['containers'].append({'id': cid, 'name': container['Name'], 'running': container['State']['Running']})
            if container['State']['Running']:
                raise RuntimeError('comparison container still running')
        record['ok'] = True
    except Exception as error:
        record['errors'].append(type(error).__name__)
    return record


def restore_if_safe(prior, safety, restore=pilot.restore_services, read_active=pilot.active):
    if safety['ok']:
        return restore(prior)
    result = {'withheld': True, 'reason': 'comparison container live or ownership/state uncertain',
              'errors': [{'operation': 'restore', 'error_type': 'UnsafeToRestartProduction'}]}
    # A timer can restart serving too. Neither unit is started in this case,
    # but their observed states are still recorded independently.
    for key, unit, user in (('backend', 'klc-backend.service', True), ('timer', 'klc-model-stack.timer', False)):
        try:
            result[key] = read_active(unit, user)
        except Exception as error:
            result[key] = None
            result['errors'].append({'unit': unit, 'error_type': type(error).__name__})
    return result


def file_inventory(out):
    return {path.name: {'sha256': pilot.sha(path), 'bytes': path.stat().st_size}
            for path in out.iterdir() if path.is_file() and path.name != 'execution.json'}


def run_slot(entry, container, out, plan_hash, expected_inventory):
    arm = entry['arm']
    name, owner = f'{PREFIX}-{slot_name(entry)}', f'{plan_hash}:{slot_name(entry)}'
    out.mkdir(mode=0o700)
    record = {**entry, 'schema': 'glm53-p8-fc1-cold-run.v1', 'started_at': pilot.now(),
              'plan_sha256': plan_hash, 'image': IMAGES[arm], 'topology': TOPOLOGY[arm],
              'exit_code': 1, 'protected_roles_opened': [], 'allocation_restart': False}
    process, cid = None, None
    try:
        if pilot.command(['docker', 'ps', '-a', '--filter', f'name=^/{name}$', '--format', '{{.ID}}']).stdout.strip():
            raise ValueError('cold container name already exists')
        with socket.socket() as port:
            port.bind(('127.0.0.1', PORT))
        with (out / 'cooldown.jsonl').open('x') as cooldown:
            deadline = time.monotonic() + 1800
            while True:
                values = pilot.temperatures()
                cooldown.write(json.dumps({'at': pilot.now(), 'temperatures': values}) + '\n'); cooldown.flush()
                if max(values) <= 75:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('startup cooling timeout')
                time.sleep(10)
        if inventory() != expected_inventory:
            raise RuntimeError('GPU identity or power limits changed')
        if pilot.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip():
            raise RuntimeError('another GPU compute process is active')
        pilot.save(out / 'nvidia-before.xml', pilot.command(['nvidia-smi', '-q', '-x']).stdout)
        launch = clone_argv(container, arm, entry, out, owner)
        pilot.private_save(out / 'launch-spec.private.json', {'argv': launch})
        cid = pilot.command(launch).stdout.strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise RuntimeError('invalid cold container id')
        record['container_id'] = cid
        pilot.command(['docker', 'start', cid])
        pilot.private_save(out / 'container.private.json', pilot.command(['docker', 'inspect', cid]).stdout)
        with (out / 'thermal.jsonl').open('x') as thermal:
            def tick():
                values = pilot.temperatures()
                thermal.write(json.dumps({'at': pilot.now(), 'temperatures': values}) + '\n'); thermal.flush()
                if max(values) >= 90:
                    raise RuntimeError('thermal abort at or above 90 C')
                state = json.loads(pilot.command(['docker', 'inspect', '-f', '{{json .State}}', cid]).stdout)
                if not state['Running']:
                    raise RuntimeError('cold server exited')

            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                tick()
                try:
                    with urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if [m['id'] for m in models['data']] == [model_name(arm)]:
                        pilot.save(out / 'models.json', models)
                        break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('readiness timeout')
            logs = pilot.command(['docker', 'logs', cid])
            text = logs.stdout + logs.stderr
            pilot.private_save(out / 'server-ready.private.log', text)
            pilot.save(out / 'runtime-audit.json', runtime_audit(text, arm))
            with (out / 'benchmark.private.log').open('x') as log:
                process = subprocess.Popen(benchmark_argv(out, arm), stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 900
                while process.poll() is None:
                    tick()
                    if time.monotonic() >= deadline:
                        raise RuntimeError('benchmark timeout')
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError('benchmark failed')
            tick()
        pilot.save(out / 'summary.json', summarize(json.loads((out / 'result.json').read_text()), arm))
        record['exit_code'] = 0
    except BaseException as error:
        record['error_type'] = type(error).__name__
        pilot.private_save(out / 'error.private.txt', str(error))
    finally:
        cleanup_signals = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGHUP)}
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate(); process.wait(timeout=10)
                except Exception:
                    process.kill(); process.wait(timeout=10)
        except BaseException as error:
            record['benchmark_cleanup_error_type'] = type(error).__name__
            record['exit_code'] = 1
        try:
            ok, errors = cleanup_owned(out, name, IMAGES[arm], owner)
            record['cleanup'] = {'ok': ok, 'errors': errors}
            if not ok:
                record['exit_code'] = 1
        except BaseException as error:
            record['cleanup'] = {'ok': False, 'error_type': type(error).__name__}
            record['exit_code'] = 1
        try:
            pilot.save(out / 'nvidia-after.xml', pilot.command(['nvidia-smi', '-q', '-x']).stdout)
            if inventory() != expected_inventory:
                raise RuntimeError('GPU identity/power changed')
            if record['exit_code'] == 0:
                final = runtime_audit((out / 'server-final.private.log').read_text(), arm)
                if final != json.loads((out / 'runtime-audit.json').read_text()):
                    raise RuntimeError('final runtime receipts differ')
        except Exception as error:
            record['final_audit_error_type'] = type(error).__name__
            record['exit_code'] = 1
        record['finished_at'] = pilot.now()
        try:
            record['files'] = file_inventory(out)
        except Exception as error:
            record['file_inventory_error_type'] = type(error).__name__
            record['exit_code'] = 1
        pilot.save(out / 'execution.json', record)
        for signum, handler in cleanup_signals.items():
            signal.signal(signum, handler)
    return record


def analyze(out, plan_path):
    plan, _ = authenticate(plan_path)
    execution = json.loads((out / 'execution.json').read_text())
    if (execution['exit_code'] != 0 or execution['completed_slots'] != 10
            or execution['plan_sha256'] != pilot.sha(plan_path) or execution['restoration']['errors']
            or not execution['restoration_safety']['ok']
            or execution['protected_roles_opened'] != [] or execution['allocation_restart'] is not False
            or [r['slot'] for r in execution['runs']] != [slot_name(e) for e in ORDER]
            or any(execution['restoration'][key] != value for key, value in execution['prior'].items())):
        raise ValueError('all ten runs and restoration must complete')
    rows, ids, previous_end = [], set(), None
    for index, entry in enumerate(ORDER):
        slot = out / slot_name(entry)
        if pilot.sha(slot / 'execution.json') != execution['runs'][index]['execution_sha256']:
            raise ValueError('cold-run execution identity differs')
        receipt = json.loads((slot / 'execution.json').read_text())
        if (receipt['round'] != entry['round'] or receipt['arm'] != entry['arm'] or receipt['exit_code'] != 0
                or not receipt['cleanup']['ok'] or receipt['plan_sha256'] != pilot.sha(plan_path)
                or receipt['image'] != IMAGES[entry['arm']] or receipt['topology'] != TOPOLOGY[entry['arm']]
                or receipt['protected_roles_opened'] != [] or receipt['container_id'] in ids
                or not re.fullmatch('[a-f0-9]{64}', receipt['container_id'])):
            raise ValueError('cold-run identity/completion differs')
        begin, end = (datetime.fromisoformat(receipt[key]) for key in ('started_at', 'finished_at'))
        if end <= begin or (previous_end is not None and begin < previous_end):
            raise ValueError('cold-process order or overlap differs')
        previous_end = end
        ids.add(receipt['container_id'])
        for name, expected in receipt['files'].items():
            path = slot / name
            if path.name != name or {'sha256': pilot.sha(path), 'bytes': path.stat().st_size} != expected:
                raise ValueError('cold-run artifact hash differs')
        summary = summarize(json.loads((slot / 'result.json').read_text()), entry['arm'])
        if summary != json.loads((slot / 'summary.json').read_text()):
            raise ValueError('cold-run summary differs')
        proof = runtime_audit((slot / 'server-final.private.log').read_text(), entry['arm'])
        if proof != json.loads((slot / 'runtime-audit.json').read_text()):
            raise ValueError('cold-run runtime proof differs')
        cooling = [json.loads(line)['temperatures'] for line in (slot / 'cooldown.jsonl').read_text().splitlines()]
        thermal = [json.loads(line)['temperatures'] for line in (slot / 'thermal.jsonl').read_text().splitlines()]
        if (not cooling or not thermal or len(cooling[-1]) != 4 or max(cooling[-1]) > 75
                or any(len(values) != 4 or not all(math.isfinite(v) and v < 90 for v in values) for values in thermal)):
            raise ValueError('cold-run thermal/start gate failed')
        final_container = json.loads((slot / 'container-final.private.json').read_text())
        if (final_container['Id'] != receipt['container_id'] or final_container['Image'] != IMAGES[entry['arm']]
                or final_container['State']['Running'] is not False
                or final_container['Config']['Labels'].get(LABEL) != f'{pilot.sha(plan_path)}:{slot_name(entry)}'):
            raise ValueError('final owned container state differs')
        created = datetime.fromisoformat(final_container['Created'].replace('Z', '+00:00'))
        started = datetime.fromisoformat(final_container['State']['StartedAt'].replace('Z', '+00:00'))
        stopped = datetime.fromisoformat(final_container['State']['FinishedAt'].replace('Z', '+00:00'))
        if not begin <= created <= started < stopped <= end:
            raise ValueError('container was not freshly created/started/stopped inside this run')
        rows.append({**entry, **summary, 'execution_sha256': pilot.sha(slot / 'execution.json')})
    medians = {arm: {key: statistics.median(row[key] for row in rows if row['arm'] == arm)
                     for key in ('decode_tokens_per_second', 'prefill_server_tokens_per_second')}
               for arm in ('p8', 'exl3')}
    wins = {key: medians['p8'][key] > medians['exl3'][key] for key in medians['p8']}
    return {'schema': 'glm53-p8-fc1-cold-comparison-analysis.v1', 'plan_sha256': pilot.sha(plan_path),
            'rows': rows, 'medians': medians, 'primary_wins': wins, 'speed_gate_pass': all(wins.values()),
            'allocation_restart': False, 'numerical_closure_required_separately': True,
            'independent_process_runs_per_arm': 5, 'topology': TOPOLOGY, 'limits': plan['limits']}


def run(plan_path):
    plan, source = authenticate(plan_path)
    out = Path(plan['output'])
    if out != out.resolve() or out.parent != ROOT or out.exists():
        raise ValueError('fresh canonical output required')
    if pilot.command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout.strip():
        raise ValueError('clean checkout required')
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise ValueError('insufficient evidence disk space')
    for arm, image in IMAGES.items():
        if pilot.command(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']).stdout.strip() != image:
            raise ValueError('immutable image not installed')
        cache = Path(plan['cache_mounts'][arm][0].split(':')[0])
        if not cache.is_dir() or not any(cache.iterdir()):
            raise ValueError('expected existing warm compiled cache missing')
    with open('/run/lock/klc/model-stack.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(mode=0o700)
        prior = {'backend': pilot.active('klc-backend.service', True), 'timer': pilot.active('klc-model-stack.timer')}
        record = {'schema': 'glm53-p8-fc1-cold-comparison-execution.v1', 'plan_sha256': pilot.sha(plan_path),
                  'started_at': pilot.now(), 'prior': prior, 'exit_code': 1, 'completed_slots': 0, 'runs': [],
                  'protected_roles_opened': [], 'allocation_restart': False,
                  'source_commit': pilot.command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).stdout.strip()}
        old_signals = {}
        try:
            def interrupt(signum, frame):
                raise RuntimeError(f'comparison interrupted by signal {signum}')

            for signum in (signal.SIGTERM, signal.SIGHUP):
                old_signals[signum] = signal.signal(signum, interrupt)
            if prior['timer']:
                pilot.command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
            if prior['backend']:
                pilot.command(['systemctl', '--user', 'stop', 'klc-backend.service'])
            hardware = inventory()
            pilot.save(out / 'hardware-inventory.json', hardware)
            payload_receipt = weight_receipt(Path(plan['weight_audit']), plan['weight_audit_sha256'])
            pilot.save(out / 'weight-audit-prerequisite.json', payload_receipt)
            for entry in ORDER:
                weights.verify_stats(payload_receipt)
                receipt = run_slot(entry, source[entry['arm']], out / slot_name(entry), pilot.sha(plan_path), hardware)
                record['runs'].append({'slot': slot_name(entry), 'execution_sha256': pilot.sha(out / slot_name(entry) / 'execution.json')})
                if receipt['exit_code'] != 0:
                    raise RuntimeError('cold run failed; no retry or advance')
                weights.verify_stats(payload_receipt)
                record['completed_slots'] += 1
            record['exit_code'] = 0
        except BaseException as error:
            record['error_type'] = type(error).__name__
            pilot.private_save(out / 'error.private.txt', str(error))
        finally:
            for signum in old_signals:
                signal.signal(signum, signal.SIG_IGN)
            record['restoration_safety'] = restoration_safety(pilot.sha(plan_path))
            record['restoration'] = restore_if_safe(prior, record['restoration_safety'])
            if record['restoration']['errors'] or any(record['restoration'][key] != value for key, value in prior.items()):
                record['exit_code'] = 1
            record['finished_at'] = pilot.now()
            pilot.save(out / 'execution.json', record)
            for signum, handler in old_signals.items():
                signal.signal(signum, handler)
    if record['exit_code']:
        raise RuntimeError('comparison failed; partial evidence preserved')
    analysis = analyze(out, plan_path)
    pilot.save(out / 'analysis.json', analysis)
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--weight-audit', type=Path, required=True)
    for mode in ('run', 'analyze'):
        command = commands.add_parser(mode)
        command.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        payload = make_plan(args.plan, args.output, args.weight_audit)
    elif args.command == 'run':
        payload = run(args.plan)
    else:
        payload = analyze(Path(json.loads(args.plan.read_text())['output']), args.plan)
    print(json.dumps(payload, sort_keys=True))
