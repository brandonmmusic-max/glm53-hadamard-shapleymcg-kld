"""Run a receipt-pinned synthetic trace of the integrated P8 small-M path."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from .p8_profile_launch import REQUIRED, BOOLEAN_FLAGS, VALUE_FLAGS, PREFIX, PROFILER

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/integrated-v1')
IMAGE = 'sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed'
RUNTIME_COMMIT = '02418d30e245bfae6586cb34b46bafd7c9e6a98f'
SOURCE_SHA = 'aa0d79bebc7cebd42b998e89fd4c45e85968832d267a1495c058c7bbb00ceba3'
NAME = 'glm53-p8-smallm-profile-v1'
PORT = 8020
NSYS = '/usr/local/cuda/bin/nsys'


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def command(args, *, check=True, timeout=60):
    return subprocess.run(args, check=check, timeout=timeout, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def save(path: Path, value):
    with path.open('x') as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def build_argv(container: dict, out: Path) -> list[str]:
    if container['Image'] != IMAGE or container['Config']['Entrypoint'] != ['/bin/bash']:
        raise ValueError('source image/entrypoint differs')
    raw = container['Config']['Cmd']
    if len(raw) != 2 or raw[0] != '-lc' or any(c in raw[1] for c in '$`\n\r;|&<>()'):
        raise ValueError('unexpected source command')
    tokens = shlex.split(raw[1])
    if tokens[:6] != ['exec', *PREFIX]:
        raise ValueError('unexpected source serving executable')
    serving = tokens[1:]
    required = {**REQUIRED, '--served-model-name': 'glm53-p8-smallm-integrated-v1', '--port': '8019'}
    options, indices = {}, {}
    i = len(PREFIX)
    while i < len(serving):
        opt = serving[i]
        if opt in options or opt not in VALUE_FLAGS | BOOLEAN_FLAGS:
            raise ValueError('unexpected source serving option')
        indices[opt] = i
        options[opt] = True if opt in BOOLEAN_FLAGS else serving[i+1]
        i += 1 if opt in BOOLEAN_FLAGS else 2
    if any(options.get(k) != v for k, v in required.items()) or not BOOLEAN_FLAGS <= options.keys():
        raise ValueError('source serving regime differs')
    serving[indices['--port']+1] = str(PORT)
    serving[indices['--served-model-name']+1] = NAME
    serving += ['--profiler-config', json.dumps(PROFILER, separators=(',', ':'))]
    host = container['HostConfig']
    if host['IpcMode'] != 'host' or host['NetworkMode'] != 'host' or host['Privileged']:
        raise ValueError('unexpected source host configuration')
    env = container['Config']['Env']
    if 'GLM53_P8_SMALL_M=1' not in env or 'GLM53_P8_NATIVE=1' not in env:
        raise ValueError('source small-M activation missing')
    argv = ['docker', 'run', '--detach', '--name', NAME, '--cidfile', str(out/'container.cid'),
            '--network', 'host', '--ipc', 'host', '--shm-size', str(host['ShmSize']),
            '--gpus', 'all', '--runtime', host['Runtime'], '--security-opt', 'label=disable',
            '--entrypoint', NSYS]
    for value in env:
        argv += ['--env', value]
    for value in host['Binds']:
        if value.split(':')[1] == '/profile':
            raise ValueError('profile mount collision')
        argv += ['--volume', value]
    return [*argv, '--volume', f'{out}:/profile:rw', IMAGE,
            'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
            '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop', '--output=/profile/smallm-c1', *serving]


def active(unit, user=False):
    result = command(['systemctl', *(['--user'] if user else []), 'is-active', unit], check=False)
    if result.stdout.strip() not in {'active', 'inactive', 'failed'}:
        raise RuntimeError(f'ambiguous service state: {unit}')
    return result.stdout.strip() == 'active'


def temperatures():
    text = command(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits']).stdout
    values = [int(x) for x in text.splitlines()]
    if len(values) != 4:
        raise RuntimeError('temperature inventory differs')
    return values


def run(out: Path, plan: Path):
    declared = json.loads(plan.read_text())
    if (declared['image'] != IMAGE or declared['runtime_commit'] != RUNTIME_COMMIT
            or declared['source_container_sha256'] != SOURCE_SHA):
        raise ValueError('plan identities differ from runner')
    if sha(plan) != plan.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('plan seal differs')
    if not out.is_absolute() or out != out.resolve() or out.exists():
        raise ValueError('a fresh absolute output directory is required')
    if sha(SOURCE/'container.json') != SOURCE_SHA:
        raise ValueError('source container receipt changed')
    source_receipt = json.loads((SOURCE/'execution.json').read_text())
    if source_receipt['exit_code'] != 0 or source_receipt['files']['container.json']['sha256'] != SOURCE_SHA:
        raise ValueError('source execution receipt invalid')
    if json.loads((SOURCE/'analysis.json').read_text())['integration_status'] != 'pass':
        raise ValueError('integrated prerequisite missing')
    command(['git', '-C', str(REPO), 'diff', '--exit-code', RUNTIME_COMMIT, '--', 'runtime_patch'])
    if command(['git', '-C', str(REPO), 'status', '--porcelain']).stdout:
        raise ValueError('source checkout must be clean')
    image = command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip()
    if image != IMAGE or shutil.disk_usage(out.parent).free < 5 * 2**30:
        raise ValueError('image or free space preflight failed')
    if command(['docker', 'inspect', NAME], check=False).returncode == 0:
        raise ValueError('diagnostic container name already exists')
    with socket.socket() as port:
        port.bind(('127.0.0.1', PORT))
    if max(temperatures()) > 75:
        raise ValueError('start temperature exceeds 75 C')
    lock = open('/run/lock/klc/model-stack.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(parents=True, mode=0o700)
    argv = build_argv(json.loads((SOURCE/'container.json').read_text())[0], out)
    save(out/'launch-spec.private.json', {'argv': argv, 'plan_sha256': sha(plan), 'source_sha256': SOURCE_SHA})
    prior_backend = active('klc-backend.service', True)
    prior_timer = active('klc-model-stack.timer')
    record = {'schema': 'glm53-p8-smallm-profile-execution.v1', 'started_at': now(),
              'image': IMAGE, 'runtime_commit': RUNTIME_COMMIT, 'plan_sha256': sha(plan),
              'runner_sha256': sha(Path(__file__)),
              'runner_commit': command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).stdout.strip(),
              'source_execution_sha256': sha(SOURCE/'execution.json'),
              'prior_backend_active': prior_backend, 'prior_timer_active': prior_timer,
              'protected_roles_opened': [], 'ldlq': False, 'exit_code': 1}
    stop = threading.Event()
    thermal_errors = []
    cid = None
    watcher = None
    try:
        if prior_timer:
            command(['sudo', '-n', 'systemctl', 'stop', 'klc-model-stack.timer'])
        if prior_backend:
            command(['systemctl', '--user', 'stop', 'klc-backend.service'])
        compute = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip()
        if compute:
            raise RuntimeError('another compute process is active')
        save(out/'nvidia-before.xml', command(['nvidia-smi', '-q', '-x']).stdout)
        command(argv)
        cid = (out/'container.cid').read_text().strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise RuntimeError('invalid owned container id')
        save(out/'container.private.json', command(['docker', 'inspect', cid]).stdout)

        def watch():
            try:
                with (out/'thermal.jsonl').open('x') as handle:
                    while not stop.is_set():
                        temps = temperatures()
                        handle.write(json.dumps({'at': now(), 'temperatures': temps})+'\n')
                        handle.flush()
                        if max(temps) >= 90:
                            raise RuntimeError(f'thermal abort: {max(temps)} C')
                        stop.wait(2)
            except BaseException as error:
                thermal_errors.append(str(error))
                command(['docker', 'stop', '--time', '5', cid], check=False, timeout=30)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        deadline = time.monotonic()+2400
        while time.monotonic() < deadline:
            if thermal_errors:
                raise RuntimeError(thermal_errors[0])
            if command(['docker', 'inspect', '-f', '{{.State.Running}}', cid]).stdout.strip() != 'true':
                raise RuntimeError('profile server exited during startup')
            try:
                with urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=5) as response:
                    models = json.load(response)
                if [entry['id'] for entry in models['data']] == [NAME]:
                    save(out/'models.json', models)
                    break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('server readiness timeout')
        # Docker sends application stderr on its own stderr stream.
        logs = command(['docker', 'logs', cid])
        log = logs.stdout + logs.stderr
        save(out/'server-ready.log', log)
        pairs = {(int(a), int(b)) for a, b in re.findall(
            r'GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+).*small_m_scheduler=true', log)}
        if pairs != {(layer, rank) for layer in range(3,45) for rank in range(4)}:
            raise RuntimeError('small-M forward inventory differs')
        graph = command([sys.executable, '-m', 'glm53_nvfp4.audit_speed_graphs', '--log', str(out/'server-ready.log')])
        save(out/'graph-capture-audit.json', json.loads(graph.stdout))
        with (out/'client.log').open('x') as client_log:
            subprocess.run([sys.executable, '-m', 'glm53_nvfp4.p8_profile_client',
                            '--base-url', f'http://127.0.0.1:{PORT}', '--expected-model', NAME,
                            '--output-dir', str(out/'client'), '--prefill-iterations', '16',
                            '--profiler-delay-iterations', '33', '--profiler-max-iterations', '16'],
                           stdout=client_log, stderr=subprocess.STDOUT, check=True, timeout=900)
        if thermal_errors:
            raise RuntimeError(thermal_errors[0])
        record['exit_code'] = 0
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        stop.set()
        if watcher:
            watcher.join(timeout=65)
        cleanup_ok = True
        if cid is None and (out/'container.cid').exists():
            value = (out/'container.cid').read_text().strip()
            cid = value if re.fullmatch('[a-f0-9]{64}', value) else None
        if cid:
            try:
                command(['docker', 'stop', '--time', '120', cid], check=False, timeout=150)
                logs = command(['docker', 'logs', cid], check=False)
                save(out/'server-final.log', logs.stdout+logs.stderr)
                state = command(['docker', 'inspect', cid])
                save(out/'container-final.private.json', state.stdout)
                cleanup_ok = not json.loads(state.stdout)[0]['State']['Running']
                if cleanup_ok:
                    command(['docker', 'rm', cid])
            except BaseException as error:
                cleanup_ok = False
                record['cleanup_error'] = str(error)
        if cleanup_ok:
            if prior_backend:
                command(['systemctl', '--user', 'start', 'klc-backend.service'])
            if prior_timer:
                command(['sudo', '-n', 'systemctl', 'start', 'klc-model-stack.timer'])
        record['restored_backend_active'] = active('klc-backend.service', True)
        record['restored_timer_active'] = active('klc-model-stack.timer')
        if not cleanup_ok or record['restored_backend_active'] != prior_backend or record['restored_timer_active'] != prior_timer:
            record['exit_code'] = 1
        record['finished_at'] = now()
        save(out/'nvidia-after.xml', command(['nvidia-smi', '-q', '-x']).stdout)
        save(out/'execution.json', record)
        lock.close()
    if record['exit_code']:
        raise RuntimeError(record.get('error', record))
    # CPU-only export after restoration. Its failure does not strand services.
    with (out/'export.log').open('x') as handle:
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', NSYS,
                        '-v', f'{out}:/profile:rw', IMAGE, 'export', '--type', 'sqlite',
                        '--output', '/profile/smallm-c1.sqlite', '/profile/smallm-c1.nsys-rep'],
                       stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=180)
    print(json.dumps({'status':'captured-exported-analysis-pending', 'output':str(out)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.plan)
