#!/usr/bin/env python3
"""Preserve the pinned index-observer startup failure; never publish private bytes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import subprocess

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-trace-v1')
PLAN = REPO / 'experiments/p8-index-trace-v1-final.json'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-trace-v1-failure'
PLAN_SHA = 'a900b61096bd377a9430dea87127d2ebf6da6d4209f8b6cbe1b4f3ee7973b265'
ROOT_SHA = '43f91db06306435eaacf4e0969349983bc3ee2b5bb23f26673495b5c2a4d1024'
STAGE_SHA = '77b496198a9bf591ab999f656f8a8bf277f677d641cd997d4feb47e1de69d451'
IMAGE = 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9'
UNIT = 'glm53-p8-index-trace-v1.service'
SLOT = 'repeat-01-n128'
PREFIX = 'glm53-p8-index-trace-v1'
FILES = {'container-final.private.json', 'container.cid', 'container.private.json',
         'cooldown.jsonl', 'error.private.txt', 'launch-spec.private.json',
         'nvidia-after.xml', 'nvidia-before.xml', 'server-final.private.log', 'thermal.jsonl'}
ERRORS = ("NameError: name 'LayerNameType' is not defined",
          'ValueError: infer_schema(func): Unsupported type annotation LayerNameType.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def terminal_state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                  '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID'], text=True)
    state = dict(line.split('=', 1) for line in raw.strip().splitlines())
    if state != {'ActiveState': 'failed', 'Result': 'exit-code', 'MainPID': '0'}:
        raise ValueError('original index trace unit is not terminal failed')
    containers = subprocess.check_output(['docker', 'ps', '-a', '--filter',
        f'name=^{PREFIX}-', '--format', '{{.ID}}'], text=True).strip()
    if containers:
        raise ValueError('comparison-owned container remains')
    return {**state, 'remaining_owned_containers': []}


def snapshot(destination=DEST):
    destination = Path(destination)
    if not destination.is_absolute() or destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical output required; no overwrite')
    state = terminal_state()
    captured = {}

    def read(path):
        if path != path.resolve() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('canonical regular input required')
        data = path.read_bytes()
        captured[path] = data
        return data

    plan_bytes = read(PLAN)
    if sha(plan_bytes) != PLAN_SHA or read(PLAN.with_suffix('.sha256')).decode().split()[0] != PLAN_SHA:
        raise ValueError('plan identity differs')
    plan = json.loads(plan_bytes)
    if Path(plan['output']) != SOURCE or plan['capture_image'] != IMAGE:
        raise ValueError('plan target differs')
    for relative, digest in plan['source_sha256'].items():
        path = REPO / relative
        if not path.is_relative_to(REPO) or sha(read(path)) != digest:
            raise ValueError('sealed source identity differs')
    root_bytes = read(SOURCE / 'execution.json')
    stage_bytes = read(SOURCE / SLOT / 'execution.json')
    if sha(root_bytes) != ROOT_SHA or sha(stage_bytes) != STAGE_SHA:
        raise ValueError('pinned execution identity differs')
    root, stage = json.loads(root_bytes), json.loads(stage_bytes)
    if (root['exit_code'] != 1 or root['plan_sha256'] != PLAN_SHA
            or root['repeats'] != [{'index': 1, 'execution_sha256': STAGE_SHA}]
            or root['prior'] != {'backend': True, 'timer': False}
            or root['restoration'] != {'backend': True, 'timer': False, 'errors': []}
            or root['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
            or root['final_identity_audit'] != {'ok': True}
            or root['protected_roles_opened'] != [] or root['teacher_logits_opened'] is not False
            or root['speed_measurement_valid'] is not False or root['allocation_restart'] is not False):
        raise ValueError('failure or restoration receipt differs')
    if (stage['exit_code'] != 1 or stage['windows'] != [] or stage['plan_sha256'] != PLAN_SHA
            or stage['capture_image'] != IMAGE or stage['cleanup'] != {'ok': True, 'errors': []}
            or stage['protected_roles_opened'] != [] or stage['unreadable_artifacts'] != []
            or stage['arm'] != 'n128' or set(stage['files']) != FILES):
        raise ValueError('zero-request startup failure differs')
    slot = SOURCE / SLOT
    if {p.name for p in slot.iterdir()} != FILES | {'execution.json', 'requests', 'captures', 'index-traces'}:
        raise ValueError('unexpected stage inventory')
    for name in ('requests', 'captures', 'index-traces'):
        directory = slot / name
        if directory != directory.resolve() or not directory.is_dir() or any(directory.iterdir()):
            raise ValueError('evaluation or trace artifacts exist')
    if {p.name for p in SOURCE.iterdir()} != {SLOT, 'execution.json', 'error.private.txt', 'hardware-inventory.json'}:
        raise ValueError('unexpected later repeat or analysis')
    for name, receipt in stage['files'].items():
        data = read(slot / name)
        if receipt != {'bytes': len(data), 'sha256': sha(data)}:
            raise ValueError(f'raw artifact identity differs: {name}')
    for name in ('error.private.txt', 'hardware-inventory.json'):
        read(SOURCE / name)
    container = json.loads(captured[slot / 'container-final.private.json'])
    if (container['Id'] != stage['container_id'] or container['Image'] != IMAGE
            or container['Name'] != '/' + PREFIX + '-repeat-01-canary-n128'
            or container['State']['Running'] is not False or container['State']['ExitCode'] != 1):
        raise ValueError('failed owned container identity differs')
    log = captured[slot / 'server-final.private.log'].decode()
    if (any(error not in log for error in ERRORS) or 'torch/_library/infer_schema.py' not in log
            or any(marker in log for marker in ('GLM53_P8_INDEX_TRACE_COMPLETE ',
                                                'GLM53_P8_DECODE_CAPTURE_V2_COMPLETE '))):
        raise ValueError('canonical failure or zero-capture log proof differs')
    loader_path = REPO / 'runtime_patch/p8_index_trace/__init__.py'
    loader = captured.get(loader_path, read(loader_path)).decode()
    if ("from __future__ import annotations" not in loader
            or "compile(source, self.origin, 'exec')" not in loader):
        raise ValueError('failed loader compile semantics differ')
    outputs = {'execution.json': root_bytes, SLOT + '/execution.json': stage_bytes}
    for name in ('thermal.jsonl', 'cooldown.jsonl'):
        safe = []
        for line in captured[slot / name].splitlines():
            row = json.loads(line)
            datetime.fromisoformat(row['at'])
            values = row['temperatures']
            if len(values) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 150 for v in values):
                raise ValueError('invalid thermal row')
            safe.append(json.dumps({'at': row['at'], 'temperatures': values}, sort_keys=True))
        outputs[SLOT + '/' + name] = ('\n'.join(safe) + '\n').encode()
    outputs['startup-failure.sanitized.json'] = encoded({
        'verified_log_excerpts': list(ERRORS), 'frame': 'torch/_library/infer_schema.py',
        'source_evidence': {'file': 'runtime_patch/p8_index_trace/__init__.py',
                            'sha256': sha(captured[loader_path]),
                            'future_annotations': True, 'compile_dont_inherit_omitted': True},
        'root_cause_inference': 'The observer loader inherits future annotations into dynamically compiled upstream code; infer_schema then cannot resolve the string LayerNameType alias.',
        'sanitization': 'Only fixed verified strings; no arbitrary log, environment or launch text copied.'})
    outputs['report.json'] = encoded({
        'schema': 'glm53-p8.index-trace-startup-failure-snapshot.v1',
        'status': 'failed-startup-before-evaluation', 'capture_image': IMAGE,
        'plan_sha256': PLAN_SHA, 'stage_files_verified': 11, 'sealed_source_files_verified': len(plan['source_sha256']),
        'evaluation_requests': 0, 'captured_rows': 0, 'trace_rows': 0,
        'teacher_logits_opened': False, 'protected_roles_opened': [],
        'restoration_verified_from_receipt': True, 'terminal_unit': state,
        'kld_measured': False, 'numerical_closure_passed': False,
        'speed_measurement_valid': False, 'allocation_restart': False,
        'limits': ['GPU/NCCL worker initialization occurred; zero evaluation requests does not mean zero GPU initialization.',
                   'Startup failure provides no evidence for or against indexer nondeterminism.',
                   'Private container, launch, error and full logs are preserved by hashes only.']})
    if terminal_state() != state or any(path.read_bytes() != data for path, data in captured.items()):
        raise ValueError('source or terminal state changed during snapshot')
    manifest = {'schema': 'glm53-p8.index-trace-failure-manifest.v1',
                'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(SOURCE),
                'generator_sha256': sha(Path(__file__).read_bytes()),
                'inputs': {str(p): {'bytes': len(v), 'sha256': sha(v)} for p, v in captured.items()},
                'outputs': {p: {'bytes': len(v), 'sha256': sha(v)} for p, v in outputs.items()}}
    destination.mkdir(parents=True)
    for relative, data in {**outputs, 'snapshot.json': encoded(manifest)}.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return {'status': 'failed-startup-preserved', 'output': str(destination), 'files': len(outputs) + 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
