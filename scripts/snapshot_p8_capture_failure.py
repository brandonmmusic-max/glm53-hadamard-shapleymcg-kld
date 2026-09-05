#!/usr/bin/env python3
"""Preserve the exact v1 startup failure without publishing private logs or env."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/forced-m1-v2-v1')
PLAN = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/experiments/p8-forced-m1-v2-v1.json')
DEST = REPO / 'evidence/opened/codec-v2/p8-forced-m1-v2-v1-failure'
PLAN_SHA = 'e3332469bc2ed9aec86eabb86b8dc57f76123542e1de309b9136893e4014725a'
EXECUTION_SHA = 'f4bd60d514db0f9ddd02009cb388500a38019eef6e05c14e5fb0db81d0aa0114'
IMAGE = 'sha256:071da1f9e9b24709e59a0959b97a8d524e82fca2fbf4de9a5dc2113a938c6b60'
UNIT = 'glm53-p8-forced-m1-v2-v1.service'
SLOT = 'canary-n128'
FILES = {'container-final.private.json', 'container.cid', 'container.private.json',
         'cooldown.jsonl', 'error.private.txt', 'launch-spec.private.json',
         'nvidia-after.xml', 'nvidia-before.xml', 'server-final.private.log', 'thermal.jsonl'}
FRAMES = (
    ('vllm/v1/worker/gpu_worker.py', 882, 'compile_or_warm_up_model'),
    ('vllm/v1/worker/gpu/warmup.py', 355, 'warmup_kernels'),
    ('vllm/v1/worker/gpu/model_runner.py', 1420, 'add_requests'),
    ('vllm/v1/worker/gpu/sample/sampler.py', 70, 'add_request'),
    ('p8_decode_capture/v2_hook.py', 109, 'add_request'),
)
ERROR = 'V2 decode capture requires one prompt token at request row zero'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def terminal_state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                   '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    state = dict(line.split('=', 1) for line in raw.strip().splitlines())
    if state != {'ActiveState': 'failed', 'Result': 'exit-code', 'ExecMainStatus': '1'}:
        raise ValueError('original capture unit is not terminal failed as recorded')
    return state


def nvidia_projection(data):
    root = ET.fromstring(data)
    fields = ('product_name', 'uuid', 'performance_state', 'temperature/gpu_temp',
              'utilization/gpu_util', 'fb_memory_usage/used', 'gpu_power_readings/power_draw',
              'gpu_power_readings/current_power_limit', 'clocks/graphics_clock', 'clocks/sm_clock')
    gpus = [{key: gpu.findtext(key) for key in fields} for gpu in root.findall('gpu')]
    if len(gpus) != 4:
        raise ValueError('four GPU records required')
    return {'driver_version': root.findtext('driver_version'), 'cuda_version': root.findtext('cuda_version'),
            'gpus': gpus, 'omitted': 'processes, serial numbers, environment and non-allowlisted fields'}


def snapshot(destination=DEST):
    destination = Path(destination)
    if not destination.is_absolute() or destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical output required; never overwrite')
    state = terminal_state()
    captured = {}
    def read(path):
        path = Path(path)
        if path != path.resolve() or not path.is_file():
            raise ValueError('input must be a canonical regular file')
        value = path.read_bytes()
        captured[path] = value
        return value
    plan_bytes = read(PLAN)
    if sha(plan_bytes) != PLAN_SHA or read(PLAN.with_suffix('.sha256')).decode().split()[0] != PLAN_SHA:
        raise ValueError('original v1 plan identity differs')
    plan = json.loads(plan_bytes)
    if Path(plan['output']) != SOURCE or plan['capture_image'] != IMAGE:
        raise ValueError('original plan target differs')
    execution_bytes = read(SOURCE / 'execution.json')
    if sha(execution_bytes) != EXECUTION_SHA:
        raise ValueError('original failure execution identity differs')
    execution = json.loads(execution_bytes)
    if (execution['exit_code'] != 1 or execution['plan_sha256'] != PLAN_SHA
            or execution['capture_image'] != IMAGE or execution['protected_roles_opened'] != []
            or execution['allocation_restart'] is not False or execution['speed_measurement_valid'] is not False
            or execution['final_identity_audit'] != {'ok': True}
            or execution['restoration_safety'] != {'ok': True, 'containers': [], 'errors': []}
            or execution['prior'] != {'backend': True, 'timer': False}
            or execution['restoration'] != {'backend': True, 'timer': False, 'errors': []}
            or len(execution['stages']) != 1 or execution['stages'][0]['slot'] != SLOT):
        raise ValueError('failure, scope or restoration receipt differs')
    slot = SOURCE / SLOT
    stage_bytes = read(slot / 'execution.json')
    if sha(stage_bytes) != execution['stages'][0]['execution_sha256']:
        raise ValueError('root to stage receipt chain differs')
    stage = json.loads(stage_bytes)
    if (stage['exit_code'] != 1 or stage['windows'] != [] or stage['plan_sha256'] != PLAN_SHA
            or stage['capture_image'] != IMAGE or stage['cleanup'] != {'ok': True, 'errors': []}
            or stage['protected_roles_opened'] != [] or stage['unreadable_artifacts'] != []
            or set(stage['files']) != FILES):
        raise ValueError('expected zero-window startup failure differs')
    for subdir in ('requests', 'captures'):
        directory = slot / subdir
        if directory != directory.resolve() or not directory.is_dir() or any(directory.iterdir()):
            raise ValueError('evaluation request or capture exists')
    if any((SOURCE / name).exists() for name in ('canary-n64', 'full-n128', 'full-n64', 'canary-exact.json', 'full-exact.json')):
        raise ValueError('unexpected evaluation stage or result exists')
    for name, receipt in stage['files'].items():
        data = read(slot / name)
        if {'bytes': len(data), 'sha256': sha(data)} != receipt:
            raise ValueError(f'raw artifact identity differs: {name}')
    container = json.loads(captured[slot / 'container-final.private.json'])
    if (container['Id'] != stage['container_id'] or container['Image'] != IMAGE
            or container['Name'] != '/glm53-p8-forced-m1-v2-v1-canary-n128'
            or container['State']['Running'] is not False or container['State']['ExitCode'] != 1):
        raise ValueError('failed container was not observably stopped')
    log = captured[slot / 'server-final.private.log'].decode()
    for path, line, function in FRAMES:
        if not re.search(re.escape(path) + r'", line ' + str(line) + ', in ' + re.escape(function), log):
            raise ValueError('canonical startup failure frame missing')
    if 'RuntimeError: ' + ERROR not in log or 'GLM53_P8_DECODE_CAPTURE_V2_COMPLETE ' in log:
        raise ValueError('canonical failure or zero-capture evidence differs')
    outputs = {'execution.json': execution_bytes, SLOT + '/execution.json': stage_bytes}
    def encoded(value):
        return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    for name in ('thermal.jsonl', 'cooldown.jsonl'):
        rows = [json.loads(line) for line in captured[slot / name].splitlines()]
        safe = []
        for row in rows:
            datetime.fromisoformat(row['at'])
            temperatures = row['temperatures']
            if len(temperatures) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 150 for v in temperatures):
                raise ValueError('invalid thermal row')
            safe.append(json.dumps({'at': row['at'], 'temperatures': temperatures}, sort_keys=True))
        outputs[SLOT + '/' + name] = ('\n'.join(safe) + '\n').encode()
    for name in ('nvidia-before.xml', 'nvidia-after.xml'):
        outputs[SLOT + '/' + name.replace('.xml', '.sanitized.json')] = encoded(nvidia_projection(captured[slot / name]))
    outputs['startup-failure.sanitized.json'] = encoded({
        'frames': [{'file': p, 'line': n, 'function': f} for p, n, f in FRAMES],
        'error_type': 'RuntimeError', 'error': ERROR,
        'transformation': 'fixed canonical frames and error verified present; no arbitrary log text copied'})
    outputs['report.json'] = encoded({
        'schema': 'glm53-p8.capture-startup-failure-snapshot.v1', 'status': 'failed-startup-before-evaluation',
        'plan_sha256': PLAN_SHA, 'capture_image': IMAGE, 'evaluation_requests': 0, 'captured_windows': 0,
        'captured_rows': 0, 'protected_roles_opened': [], 'restoration_verified_from_receipt': True,
        'numerical_closure_passed': False, 'kld_available': False, 'speed_measurement_valid': False,
        'allocation_restart': False, 'terminal_unit': state,
        'limits': ['Conditional-fit teacher inputs were verified during planning; zero evaluation captures does not mean the role was unopened.',
                   'This preserves a failed startup, not numerical or throughput evidence. Private raw logs and environment are referenced by hashes only.'],
        'reproduction': 'Run scripts/snapshot_p8_capture_failure.py with the pinned original plan and raw root still available.'})
    # Revalidate every byte and terminal state before creating any output.
    if terminal_state() != state or any(path.read_bytes() != value for path, value in captured.items()):
        raise ValueError('input or unit state changed during snapshot')
    manifest = {'schema': 'glm53-p8.failure-snapshot-manifest.v1',
                'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(SOURCE),
                'generator_sha256': sha(Path(__file__).read_bytes()),
                'inputs': {str(p): {'bytes': len(v), 'sha256': sha(v)} for p, v in captured.items()},
                'outputs': {p: {'bytes': len(v), 'sha256': sha(v)} for p, v in outputs.items()}}
    destination.mkdir(parents=True)
    for relative, value in outputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(value)
    with (destination / 'snapshot.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return {'status': 'failed-startup-preserved', 'output': str(destination), 'files': len(outputs) + 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
