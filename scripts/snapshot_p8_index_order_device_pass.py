#!/usr/bin/env python3
"""Snapshot the pinned five-process synthetic v2 kernel gate, not model closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4 import p8_index_order_device_runner as runner

PLAN = REPO / 'experiments/p8-index-order-device-v2.json'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-order-device-v2-pass'
PLAN_SHA = 'fa2b503b9791fd9135240bee7a0b811084e98c2fe95f53bb12d2c53cc205db91'
ROOT_SHA = '251757fd24931fe2ca260fedd63ed1a3819aa5170194b0298dc95036c33390f5'
PREFIX = 'glm53-p8-index-order-device-v2'
FILES = {'container-id.txt', 'container-state.json', 'launch.json', 'probe.log', 'result.json', 'thermal.jsonl'}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def terminal():
    raw = subprocess.check_output(['systemctl', '--user', 'show', PREFIX + '.service',
                                  '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result'], text=True)
    state = dict(line.split('=', 1) for line in raw.splitlines())
    if state != {'MainPID': '0', 'ActiveState': 'inactive', 'Result': 'success'}:
        raise ValueError('unit is not terminal success')
    if subprocess.check_output(['docker', 'ps', '-a', '--filter', f'name=^{PREFIX}-',
                                '--format', '{{.ID}}'], text=True).strip():
        raise ValueError('owned container remains')
    return {**state, 'remaining_owned_containers': []}


def classify(root, results, plan):
    if (root['exit_code'] != 0 or root['kernel_gate_pass'] is not True or root['plan_sha256'] != PLAN_SHA
            or root['prior'] != {'backend': True, 'timer': False}
            or root['restoration'] != {'backend': True, 'timer': False, 'safe': True, 'errors': []}
            or root['final_identity_audit'] is not True or root['allocation_restart'] is not False
            or root['teacher_logits_opened'] is not False or root['protected_roles_opened'] != []
            or [row['index'] for row in root['repeats']] != [1, 2, 3, 4, 5] or len(results) != 5):
        raise ValueError('five-process terminal/restoration contract differs')
    signatures = []
    for row, result in zip(root['repeats'], results):
        value = runner.validate_result(result, plan)
        if value != row['determinism_sha256'] or len(result['cases']) != 380 or len(result['state_transitions']) != 80:
            raise ValueError('case or signature receipt differs')
        if result['qualification_pass'] is not False or result['kld_measured'] is not False:
            raise ValueError('synthetic scope differs')
        signatures.append(value)
    if len(set(signatures)) != 1:
        raise ValueError('fresh process determinism differs')
    return {'schema': 'glm53-p8.index-order-device-pass-snapshot.v1', 'status': 'synthetic-kernel-gate-passed',
            'five_fresh_processes': True, 'case_calls': 1900, 'state_transition_calls': 400,
            'total_check_calls': 2300, 'determinism_sha256': signatures[0],
            'model_loaded': False, 'teacher_logits_opened': False, 'protected_roles_opened': [],
            'kld_measured': False, 'speed_measurement_valid': False, 'allocation_restart': False,
            'full_model_numerical_closure': False, 'restoration_verified_from_receipt': True}


def snapshot():
    if DEST.exists() or DEST != DEST.resolve():
        raise ValueError('fresh canonical destination required')
    state = terminal()
    inputs = {}

    def read(path):
        if path != path.resolve() or not path.is_file():
            raise ValueError('canonical regular input required')
        data = path.read_bytes()
        inputs[path] = data
        return data

    plan_bytes = read(PLAN)
    if digest(plan_bytes) != PLAN_SHA or read(PLAN.with_suffix('.sha256')).decode().split()[0] != PLAN_SHA:
        raise ValueError('plan seal differs')
    plan = json.loads(plan_bytes)
    raw = runner.validate(plan)
    if len(plan['source_sha256']) != 9:
        raise ValueError('sealed source count differs')
    for name, value in plan['source_sha256'].items():
        if digest(read(REPO / name)) != value:
            raise ValueError('sealed source drift')
    read(Path(plan['import_preflight']))
    root_bytes = read(raw / 'execution.json')
    if digest(root_bytes) != ROOT_SHA:
        raise ValueError('root receipt differs')
    root = json.loads(root_bytes)
    if {p.name for p in raw.iterdir()} != {'execution.json', 'cache'} | {f'repeat-{i:02d}' for i in range(1, 6)}:
        raise ValueError('repeat inventory differs')
    outputs = {'plan.json': plan_bytes, 'execution.json': root_bytes}
    results, ids, temperatures, previous_finish = [], [], [], None
    for index, receipt in enumerate(root['repeats'], 1):
        slot = raw / f'repeat-{index:02d}'
        if {p.name for p in slot.iterdir()} != FILES:
            raise ValueError('probe artifact inventory differs')
        data = {name: read(slot / name) for name in FILES}
        if digest(data['result.json']) != receipt['result_sha256']:
            raise ValueError('probe hash differs from root receipt')
        results.append(json.loads(data['result.json']))
        ids.append(data['container-id.txt'].decode().strip())
        container = json.loads(data['container-state.json'])
        if (container['Running'] is not False or container['Pid'] != 0 or container['ExitCode'] != 0
                or container['StartedAt'] >= container['FinishedAt']
                or (previous_finish is not None and container['StartedAt'] <= previous_finish)):
            raise ValueError('independent sequential container execution differs')
        previous_finish = container['FinishedAt']
        if json.loads(data['launch.json']) != {'argv': runner.argv(plan, raw, index, PLAN_SHA)}:
            raise ValueError('launch differs from sealed synthetic-only recipe')
        thermal = [json.loads(line) for line in data['thermal.jsonl'].splitlines()]
        if not thermal or any(set(row) != {'at', 'temperatures'} or len(row['temperatures']) != 4
                              or any(type(t) not in (int, float) or not 0 <= t < 90 for t in row['temperatures'])
                              for row in thermal):
            raise ValueError('thermal contract differs')
        temperatures.append(max(max(row['temperatures']) for row in thermal))
        outputs.update({f'repeat-{index:02d}/' + name: value for name, value in data.items() if name != 'probe.log'})
    if len(set(ids)) != 5 or any(len(value) != 64 for value in ids):
        raise ValueError('distinct container IDs not proven')
    report = classify(root, results, plan)
    report.update({'sealed_sources_verified': 9, 'terminal_unit': state,
                   'per_process_max_sampled_temperature_c': temperatures,
                   'max_sampled_temperature_c': max(temperatures)})
    outputs['report.json'] = encoded(report)
    if terminal() != state or any(path.read_bytes() != value for path, value in inputs.items()):
        raise ValueError('terminal evidence changed')
    manifest = {'schema': 'glm53-p8.index-order-device-pass-manifest.v1', 'source_root': str(raw),
                'generator_sha256': digest(Path(__file__).read_bytes()),
                'inputs': {str(p): {'bytes': len(v), 'sha256': digest(v)} for p, v in inputs.items()},
                'outputs': {p: {'bytes': len(v), 'sha256': digest(v)} for p, v in outputs.items()},
                'log_policy': 'logs hash only; launch copied after exact frozen synthetic-only argv validation'}
    DEST.mkdir(parents=True)
    for name, content in {**outputs, 'snapshot.json': encoded(manifest)}.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    return report


if __name__ == '__main__':
    print(json.dumps(snapshot(), sort_keys=True))
