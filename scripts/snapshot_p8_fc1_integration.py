#!/usr/bin/env python3
"""Preserve one terminal integration attempt, excluding private launch details."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4.p8_fc1_integration import authenticate, summarize, verify_runtime

ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1')
SOURCE = ROOT / 'fc1-integrated-v1'
DEST = REPO / 'evidence/opened/codec-v2/p8-fc1-integrated-v1'
UNIT = 'glm53-p8-fc1-integrated-v1.service'
FILES = ('result.json', 'summary.json', 'runtime-audit.json', 'thermal.jsonl',
         'device-prerequisite.json', 'models.json', 'nvidia-before.xml', 'nvidia-after.xml')
SUCCESS_FILES = {'result.json', 'summary.json', 'runtime-audit.json', 'thermal.jsonl',
                 'device-prerequisite.json', 'models.json'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def terminal_state():
    result = subprocess.run(
        ['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'Result'],
        check=True, capture_output=True, text=True, timeout=30)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if state.get('ActiveState') not in ('inactive', 'failed') or not state.get('Result'):
        raise ValueError('integration unit is not observably terminal')
    return state


def sanitize_log(text):
    """Extract typed receipt fields, never preserve arbitrary source log text."""
    rows = []
    for line in text.splitlines():
        for match in re.finditer(r'\benforce_eager=(True|False)\b', line):
            rows.append(f'enforce_eager={match[1]}')
        if 'Breakable CUDA graph enabled' in line:
            rows.append('Breakable CUDA graph enabled')
        if re.search(r'Capturing (?:decode )?CUDA graphs \(FULL\):[^\r\n]*100%', line):
            rows.append('Capturing CUDA graphs (FULL): 100%')
        for match in re.finditer(r'\(Worker_TP(\d+)[^)]*\)[^\r\n]*Graph capturing finished', line):
            rows.append(f'(Worker_TP{int(match[1])}) Graph capturing finished')
        for match in re.finditer(
            r'GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\r\n]*small_m_scheduler=(true|false)', line
        ):
            rows.append(f'GLM53_P8_NATIVE_FORWARD layer={int(match[1])} rank={int(match[2])} '
                        f'small_m_scheduler={match[3]}')
        for match in re.finditer(
            r'GLM53_P8_M1_DISPATCH layer=(\d+) rank=(\d+) fc1_tile_n=(\d+) fused_scratch_zero=(true|false)', line
        ):
            rows.append(f'GLM53_P8_M1_DISPATCH layer={int(match[1])} rank={int(match[2])} '
                        f'fc1_tile_n={int(match[3])} fused_scratch_zero={match[4]}')
    return ('\n'.join(rows) + ('\n' if rows else '')).encode()


def snapshot(plan_path):
    if SOURCE != SOURCE.resolve() or SOURCE.parent != ROOT or DEST != DEST.resolve():
        raise ValueError('noncanonical snapshot paths')
    if DEST.exists():
        raise ValueError('snapshot destination already exists; no overwrite')
    state = terminal_state()
    plan, _ = authenticate(plan_path)
    execution_path = SOURCE / 'execution.json'
    if execution_path != execution_path.resolve():
        raise ValueError('execution receipt must not be a symlink')
    raw_execution = execution_path.read_bytes()
    execution = json.loads(raw_execution)
    expected = {
        'schema': 'glm53-p8-fc1-integration-execution.v1',
        'plan_sha256': digest(plan_path.read_bytes()), 'image': plan['candidate_image'],
        'tile_n': plan['tile_n'], 'fused_scratch_zero': plan['fused_scratch_zero'],
        'protected_roles_opened': [], 'allocation_restart': False,
        'single_run_diagnostic': True, 'five_cold_run_product_comparison': False,
    }
    if (any(execution.get(key) != value for key, value in expected.items())
            or type(execution.get('exit_code')) is not int
            or execution['exit_code'] not in (0, 1) or not execution.get('finished_at')):
        raise ValueError('terminal execution identity differs from sealed diagnostic')
    success = execution['exit_code'] == 0
    if success and (state != {'ActiveState': 'inactive', 'Result': 'success'}
                    or not execution.get('cleanup', {}).get('ok')
                    or execution['restoration']['errors']
                    or any(execution['restoration'][key] != value
                           for key, value in execution['prior'].items())):
        raise ValueError('successful receipt lacks successful cleanup/restoration/unit state')
    staged = {'execution.json': raw_execution}
    records = [{'path': 'execution.json', 'source': str(execution_path),
                'bytes': len(raw_execution), 'sha256': digest(raw_execution), 'transformation': 'none'}]

    def read_verified(name):
        path = SOURCE / name
        if path != path.resolve():
            raise ValueError('snapshot input must not be a symlink')
        data = path.read_bytes()
        receipt = execution['files'].get(name)
        if receipt != {'sha256': digest(data), 'bytes': len(data)}:
            raise ValueError(f'raw artifact identity differs: {name}')
        return data

    missing = []
    for name in FILES:
        if not (SOURCE / name).is_file():
            missing.append(name)
            continue
        data = read_verified(name)
        if name == 'result.json':
            # The benchmark collects some host environment prefixes. The prior
            # exact recipe had none; do not publish unforeseen environment data.
            if json.loads(data).get('startup_diagnostics', {}).get('env', {}):
                raise ValueError('benchmark environment requires separate privacy review')
        staged[name] = data
        records.append({'path': name, 'source': str(SOURCE / name), 'bytes': len(data),
                        'sha256': digest(data), 'transformation': 'none'})
    for name in ('server-ready.log', 'server-final.log'):
        if not (SOURCE / name).is_file():
            missing.append(name)
            continue
        data = read_verified(name)
        safe = sanitize_log(data.decode(errors='replace'))
        target = name.removesuffix('.log') + '.sanitized.log'
        staged[target] = safe
        records.append({'path': target, 'source': str(SOURCE / name),
                        'raw_bytes': len(data), 'raw_sha256': digest(data),
                        'bytes': len(safe), 'sha256': digest(safe),
                        'transformation': 'canonical graph and native/M1 dispatch receipt fields only'})
    if success:
        if SUCCESS_FILES.intersection(missing) or 'server-final.log' in missing:
            raise ValueError('successful execution is missing required evidence')
        if summarize(json.loads(staged['result.json'])) != json.loads(staged['summary.json']):
            raise ValueError('integration summary does not replay')
        proof = verify_runtime(staged['server-final.sanitized.log'].decode(),
                               plan['tile_n'], plan['fused_scratch_zero'])
        if proof != json.loads(staged['runtime-audit.json']):
            raise ValueError('final sanitized dispatch/graph proof does not replay')
    if terminal_state() != state or execution_path.read_bytes() != raw_execution:
        raise ValueError('terminal source changed during snapshot')
    manifest = {
        'schema': 'glm53-p8-fc1-integration-snapshot.v1', 'unit': UNIT,
        'terminal_state': state, 'execution_exit_code': execution['exit_code'],
        'status': 'terminal-success-diagnostic' if success else 'terminal-failure-preserved',
        'plan': str(plan_path), 'plan_sha256': expected['plan_sha256'],
        'files': records, 'missing_artifacts': missing, 'allocation_restart': False,
        'five_cold_run_product_comparison': False, 'protected_roles_opened': [],
        'note': ('One synthetic32K C1 and standalone-prefill diagnostic, not full-model quality '
                 'or five-cold-run qualification. Failures remain failures even if some measured '
                 'values exist. Private launch/container/env/error files and arbitrary logs are '
                 'excluded; sanitized logs retain only typed graph/dispatch proof fields.'),
    }
    staged['snapshot.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    # Validate everything before the first write. Never modify raw inputs and
    # never replace an existing snapshot (including a partial prior snapshot).
    DEST.mkdir(parents=True, exist_ok=False)
    for name, data in staged.items():
        with (DEST / name).open('xb') as handle:
            handle.write(data)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    result = snapshot(parser.parse_args().plan)
    print(json.dumps({'status': result['status'], 'files': len(result['files']), 'destination': str(DEST)}))
