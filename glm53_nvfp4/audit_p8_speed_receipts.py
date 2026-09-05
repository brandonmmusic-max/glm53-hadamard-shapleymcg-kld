"""Read-only receipt audit; does not change the frozen throughput estimator."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .analyze_p8_vs_exl3_speed import _extract
from .audit_speed_graphs import verify as verify_graphs


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config_fingerprint(container: dict) -> str:
    env = container['Config']['Env']
    binds = container['HostConfig']['Binds']
    # Docker inspect does not preserve ordering of these key-addressed lists.
    # Reject duplicate keys/destinations before normalizing that ordering.
    names = [value.split('=', 1)[0] for value in env]
    destinations = [value.split(':')[1] for value in binds]
    if len(set(names)) != len(names) or len(set(destinations)) != len(destinations):
        raise ValueError('ambiguous duplicate environment or mount entries')
    return fingerprint({'cmd': container['Config']['Cmd'], 'env': sorted(env),
                        'binds': sorted(binds),
                        'devices': container['HostConfig']['DeviceRequests']})


def audit_slot(slot: Path, plan_path: Path, arm: str, round_no: int) -> dict:
    plan = json.loads(plan_path.read_text())
    spec = plan['candidate' if arm == 'p8' else 'baseline']
    receipt = json.loads((slot / 'receipt.json').read_text())
    if (receipt.get('round'), receipt.get('arm')) != (round_no, arm):
        raise ValueError('receipt slot identity differs')
    if receipt.get('protected_roles_opened') != []:
        raise ValueError('protected-role receipt differs')
    required = ('benchmark.json', 'server.log', 'container.json',
                'nvidia-before.xml', 'nvidia-after.xml', plan_path.name)
    for name in required:
        record = receipt['files'][name]
        path = plan_path if name == plan_path.name else slot / name
        if record['sha256'] != sha(path) or record['bytes'] != path.stat().st_size:
            raise ValueError(f'receipt hash/size differs: {name}')
    text = (slot / 'server.log').read_text(errors='replace')
    verify_graphs(text)
    for marker in ('tensor_parallel_size=4',
                   f'decode_context_parallel_size={spec["topology"]["dcp"]}',
                   'speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla'):
        if marker not in text:
            raise ValueError(f'runtime marker missing: {marker}')
    ep = "'enable_expert_parallel': True" in text
    if ep != spec['topology']['ep']:
        raise ValueError('EP topology differs')
    if arm == 'p8':
        pairs = {(int(a), int(b)) for a, b in re.findall(
            r'GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)', text)}
        if pairs != {(layer, rank) for layer in range(3, 45) for rank in range(4)}:
            raise ValueError('native P8 dispatch inventory differs')
    else:
        for marker in ('EXL3 full-expert EP runtime planned', 'quantization=exl3',
                       'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts'):
            if marker not in text:
                raise ValueError(f'EXL3 backend marker missing: {marker}')
    extracted = _extract(slot / 'benchmark.json', spec['model_name'])
    hardware = extracted['hardware']
    measured_temp = float(hardware['temp_max_c'])
    if hardware['gpu_count'] != 4 or not math.isfinite(measured_temp) or not 0 < measured_temp < 94:
        raise ValueError('measured thermal/GPU-count gate differs')
    before = ET.parse(slot / 'nvidia-before.xml').getroot()
    temps = [float(g.findtext('temperature/gpu_temp', '').split()[0])
             for g in before.findall('gpu')]
    if len(temps) != 4 or any(not math.isfinite(t) or not 0 < t <= 75 for t in temps):
        raise ValueError('cold-start thermal gate differs')
    containers = json.loads((slot / 'container.json').read_text())
    if len(containers) != 1:
        raise ValueError('container inventory differs')
    container = containers[0]
    if container['Image'] != spec['image_id']:
        raise ValueError('image identity differs')
    started, completed = container['State']['StartedAt'], receipt['completed_at']
    if stamp(started) >= stamp(completed):
        raise ValueError('container start/completion chronology differs')
    return {
        'round': round_no, 'arm': arm, 'container_id': container['Id'],
        'started_at': started, 'completed_at': completed,
        'image_id': container['Image'], 'receipt_sha256': sha(slot / 'receipt.json'),
        'benchmark_sha256': extracted['sha256'], 'start_max_temp_c': max(temps),
        'measured_max_temp_c': measured_temp,
        # Bind configuration without publishing environment contents.
        'config_sha256': config_fingerprint(container),
    }


def audit(root: Path, plan_path: Path, allow_incomplete: bool = False) -> dict:
    plan = json.loads(plan_path.read_text())
    orders = plan['benchmark']['arm_order_by_round']
    if len(orders) != plan['benchmark']['cold_process_runs_per_arm']:
        raise ValueError('plan round inventory differs')
    rows, missing = [], []
    ids, configs = set(), {}
    for round_no, order in enumerate(orders, 1):
        if sorted(order) != ['exl3', 'p8']:
            raise ValueError('plan arm inventory differs')
        for arm in order:
            slot = root / f'round-{round_no:02d}-{arm}'
            if not (slot / 'receipt.json').exists():
                missing.append(slot.name)
                continue
            if missing:
                raise ValueError('completed slot appears after a missing predecessor')
            row = audit_slot(slot, plan_path, arm, round_no)
            if row['container_id'] in ids:
                raise ValueError('cold runs reused a container')
            ids.add(row['container_id'])
            if arm in configs and configs[arm] != row['config_sha256']:
                raise ValueError('within-arm runtime configuration changed')
            configs[arm] = row['config_sha256']
            if rows and stamp(row['started_at']) <= stamp(rows[-1]['completed_at']):
                raise ValueError('cold-run chronology overlaps or order differs')
            rows.append(row)
    if missing and not allow_incomplete:
        raise ValueError(f'incomplete speed series: {missing}')
    return {'schema': 'glm53-p8-speed-receipt-audit.v1',
            'status': 'incomplete' if missing else 'pass', 'runs': rows,
            'missing': missing, 'plan_sha256': sha(plan_path),
            'scope': 'Receipt integrity and declared execution conditions, not a new speed estimator or causal codec claim.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.plan, args.allow_incomplete)
    if args.output:
        with args.output.open('x') as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
