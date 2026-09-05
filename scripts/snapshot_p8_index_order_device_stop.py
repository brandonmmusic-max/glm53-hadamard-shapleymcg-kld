#!/usr/bin/env python3
"""Preserve the exact terminal v1 UUID-format stop without retroactive acceptance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4 import p8_index_order_device_runner as runner

PLAN = REPO / 'experiments/p8-index-order-device-v1.json'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-order-device-v1-stop'
PLAN_SHA = '5ed0f8c3268dbdc05d039335b08884f033c6642f2c7f3781fc502bc83873e021'
ROOT_SHA = '23b973a5225823fd228941ae72200e52c67b1148ebc0885058879e17bf7339d3'
RESULT_SHA = 'fe6f56c73dc1d90fff28cc41535b8b4dfd3c4d7c390f017e9804ac8489104574'
PREFIX = 'glm53-p8-index-order-device-v1'
FILES = {'container-id.txt', 'container-state.json', 'launch.json', 'probe.log', 'result.json', 'thermal.jsonl'}


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def terminal():
    text = subprocess.check_output(['systemctl', '--user', 'show', PREFIX + '.service',
                                   '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result'], text=True)
    state = dict(line.split('=', 1) for line in text.splitlines())
    if state != {'MainPID': '0', 'ActiveState': 'failed', 'Result': 'exit-code'}:
        raise ValueError('unit is not the pinned terminal failure')
    if subprocess.check_output(['docker', 'ps', '-a', '--filter', f'name=^{PREFIX}-',
                                '--format', '{{.ID}}'], text=True).strip():
        raise ValueError('owned container remains')
    return {**state, 'remaining_owned_containers': []}


def classify(result, plan, root):
    if (root['exit_code'] != 1 or root['repeats'] != [] or root['plan_sha256'] != PLAN_SHA
            or root['error'] != {'type': 'ValueError', 'message': 'actual probe GPU does not match authorized physical GPU0'}
            or root['restoration'] != {'backend': True, 'timer': False, 'errors': [], 'safe': True}
            or root['prior'] != {'backend': True, 'timer': False}
            or root['final_identity_audit'] is not True
            or root['teacher_logits_opened'] is not False or root['protected_roles_opened'] != []
            or root['allocation_restart'] is not False):
        raise ValueError('terminal stop or restoration contract differs')
    expected = plan['gpu_inventory'][0].split(',')[1].strip()
    actual = result['gpu']['uuid']
    if expected != 'GPU-' + actual:
        raise ValueError('not solely the pinned GPU UUID prefix mismatch')
    normalized = copy.deepcopy(result)
    normalized['gpu']['uuid'] = expected
    runner.validate_result(normalized, plan)
    if (len(result['cases']) != 380 or len(result['state_transitions']) != 80
            or result['qualification_pass'] is not False or result['kld_measured'] is not False):
        raise ValueError('probe scope differs')
    return {'status': 'host-gate-stopped-uuid-format', 'probe_processes_completed': 1,
            'host_accepted_repeats': 0, 'planned_independent_processes': 5,
            'case_calls': 380, 'state_transition_calls': 80, 'five_process_gate_passed': False,
            'raw_probe_status': result['status'], 'reported_gpu_uuid': actual,
            'authorized_gpu_uuid': expected, 'uuid_differs_only_by_GPU_prefix': True,
            'audit_only_normalized_uuid_inventory_replay': True,
            'raw_receipts_modified': False, 'model_loaded': False, 'teacher_logits_opened': False,
            'kld_measured': False, 'speed_measurement_valid': False, 'allocation_restart': False}


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
    if len(plan['source_sha256']) != 9:
        raise ValueError('sealed source count differs')
    for name, value in plan['source_sha256'].items():
        if digest(read(REPO / name)) != value:
            raise ValueError('sealed source drift')
    raw = Path(plan['output'])
    if raw != Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-order-device-v1'):
        raise ValueError('raw root differs')
    if {p.name for p in raw.iterdir()} != {'cache', 'execution.json', 'repeat-01'}:
        raise ValueError('unexpected later repeats')
    slot = raw / 'repeat-01'
    if {p.name for p in slot.iterdir()} != FILES:
        raise ValueError('unexpected probe artifact inventory')
    root_bytes = read(raw / 'execution.json')
    data = {name: read(slot / name) for name in FILES}
    if digest(root_bytes) != ROOT_SHA or digest(data['result.json']) != RESULT_SHA:
        raise ValueError('pinned raw receipts differ')
    root, result = json.loads(root_bytes), json.loads(data['result.json'])
    report = classify(result, plan, root)
    container = json.loads(data['container-state.json'])
    if container['Running'] is not False or container['Pid'] != 0 or container['ExitCode'] != 0:
        raise ValueError('probe container did not exit successfully')
    if json.loads(data['launch.json']) != {'argv': runner.argv(plan, raw, 1, PLAN_SHA)}:
        raise ValueError('launch differs from exact sealed synthetic-only argv')
    thermal = [json.loads(line) for line in data['thermal.jsonl'].splitlines()]
    if not thermal or any(set(row) != {'at', 'temperatures'} or len(row['temperatures']) != 4
                          or any(type(t) not in (int, float) or not 0 <= t < 90 for t in row['temperatures'])
                          for row in thermal):
        raise ValueError('thermal contract differs')
    report.update({'sealed_sources_verified': 9, 'terminal_unit': state,
                   'max_sampled_temperature_c': max(max(row['temperatures']) for row in thermal),
                   'restoration_verified_from_receipt': True})
    outputs = {'plan.json': plan_bytes, 'execution.json': root_bytes, 'report.json': encoded(report)}
    outputs.update({'repeat-01/' + name: content for name, content in data.items() if name != 'probe.log'})
    if terminal() != state or any(path.read_bytes() != content for path, content in inputs.items()):
        raise ValueError('evidence changed during snapshot')
    manifest = {'schema': 'glm53-p8.index-order-device-stop-snapshot.v1', 'source_root': str(raw),
                'generator_sha256': digest(Path(__file__).read_bytes()),
                'inputs': {str(p): {'bytes': len(v), 'sha256': digest(v)} for p, v in inputs.items()},
                'outputs': {p: {'bytes': len(v), 'sha256': digest(v)} for p, v in outputs.items()},
                'log_policy': 'probe.log hash only; launch copied only after exact frozen synthetic argv comparison'}
    DEST.mkdir(parents=True)
    for name, content in {**outputs, 'snapshot.json': encoded(manifest)}.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    return report


if __name__ == '__main__':
    print(json.dumps(snapshot(), sort_keys=True))
