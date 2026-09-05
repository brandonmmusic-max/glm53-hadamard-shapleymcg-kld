#!/usr/bin/env python3
"""Preserve terminal v2 index diagnostics; large arrays and private logs stay local."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4 import p8_index_trace_launcher as launch
from scripts.snapshot_p8_fc1_cold_comparison import sanitize_log as sanitize_base

SOURCE = launch.ROOT / 'index-trace-v2'
PLAN = REPO / 'experiments/p8-index-trace-v2.json'
PLAN_SHA = 'd8e814b5a25ea7d557d2efeb297914aad4c3d98d338e1ef816c68d7c1ecddd2d'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-trace-v2'
UNIT = 'glm53-p8-index-trace-v2.service'
PATTERNS = (
    r'GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=\d+ max_num_reqs=\d+ real_vocab=\d+ expected_outputs=\d+',
    r'GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=\d+ registrations=\d+ samples=\d+',
    r'GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=conditional-fit-\d{4} rows=\d+ tp_rank=\d+',
    r'GLM53_P8_INDEX_TRACE_COMPLETE tp_rank=\d+ window=conditional-fit-\d{4} rows=\d+ layers=\d+',
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def sanitize_log(text):
    rows = sanitize_base(text).decode().splitlines()
    if 'Using V2 Model Runner' in text:
        rows.append('Using V2 Model Runner')
    for pattern in PATTERNS:
        rows.extend(re.findall(pattern, text))
    return ('\n'.join(rows) + '\n').encode()


def thermal_projection(data):
    rows = []
    for line in data.splitlines():
        row = json.loads(line)
        datetime.fromisoformat(row['at'])
        values = row['temperatures']
        if len(values) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 150 for v in values):
            raise ValueError('invalid thermal row')
        rows.append({'at': row['at'], 'temperatures': values})
    return ('\n'.join(json.dumps(row, sort_keys=True) for row in rows) + '\n').encode()


def terminal_state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                  '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID'], text=True)
    state = dict(line.split('=', 1) for line in raw.strip().splitlines())
    if state != {'ActiveState': 'inactive', 'Result': 'success', 'MainPID': '0'}:
        raise ValueError('diagnostic is not terminal successful')
    if subprocess.check_output(['docker', 'ps', '-a', '--filter',
            'name=^glm53-p8-index-trace-v2-', '--format', '{{.ID}}'], text=True).strip():
        raise ValueError('owned diagnostic container remains')
    return {**state, 'remaining_owned_containers': []}


def snapshot(destination=DEST):
    destination = Path(destination)
    if destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical destination required; no overwrite')
    state = terminal_state()
    if launch.pilot.sha(PLAN) != PLAN_SHA:
        raise ValueError('pinned plan differs')
    plan = launch.authenticate(PLAN)
    if Path(plan['output']) != SOURCE:
        raise ValueError('plan output differs')
    observed, output = {}, {}

    def reference(path, expected=None):
        if path != path.resolve() or not path.is_file():
            raise ValueError('canonical regular source required')
        receipt = {'bytes': path.stat().st_size, 'sha256': launch.pilot.sha(path)}
        if expected is not None and receipt != expected:
            raise ValueError(f'raw receipt differs: {path.name}')
        observed[str(path)] = receipt
        return receipt

    def preserve(relative, path):
        reference(path)
        output[relative] = path.read_bytes()

    preserve('execution.json', SOURCE / 'execution.json')
    execution = json.loads(output['execution.json'])
    if (execution['exit_code'] != 0 or execution['plan_sha256'] != PLAN_SHA
            or execution['diagnostic_complete'] is not True or execution['qualification_pass'] is not False
            or execution['final_identity_audit'] != {'ok': True}
            or execution['prior'] != {'backend': True, 'timer': False}
            or execution['restoration'] != {'backend': True, 'timer': False, 'errors': []}
            or execution['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
            or [item['index'] for item in execution['repeats']] != [1, 2]
            or execution['teacher_logits_opened'] is not False or execution['protected_roles_opened'] != []):
        raise ValueError('terminal execution/restoration differs')
    reference(PLAN)
    reference(PLAN.with_suffix('.sha256'))
    for name in plan['source_sha256']:
        reference(REPO / name)
    replay = launch.compare_repeats(plan)
    comparison = SOURCE / 'comparison.json'
    if launch.pilot.sha(comparison) != execution['comparison_sha256'] or replay != json.loads(comparison.read_text()):
        raise ValueError('complete comparison replay differs')
    preserve('comparison.json', comparison)
    ids, previous_end, maxima = set(), execution['started_at'], []
    for index, receipt in enumerate(execution['repeats'], 1):
        name = launch.repeat_name(index)
        slot = SOURCE / name
        preserve(name + '/execution.json', slot / 'execution.json')
        stage = json.loads(output[name + '/execution.json'])
        if (launch.pilot.sha(slot / 'execution.json') != receipt['execution_sha256']
                or stage['exit_code'] != 0 or stage['cleanup'] != {'ok': True, 'errors': []}
                or stage['plan_sha256'] != PLAN_SHA or stage['capture_image'] != launch.IMAGE
                or stage['arm'] != 'n128' or stage['unreadable_artifacts'] != []
                or stage['protected_roles_opened'] != [] or len(stage['windows']) != 1
                or stage['windows'][0]['id'] != launch.WINDOW or stage['windows'][0]['rows'] != 2047
                or stage['container_id'] in ids
                or not previous_end <= stage['started_at'] < stage['finished_at'] <= execution['finished_at']):
            raise ValueError('stage hash/lifecycle/causal-window differs')
        ids.add(stage['container_id'])
        previous_end = stage['finished_at']
        for relative, expected in stage['files'].items():
            reference(slot / relative, expected)
        tokens = launch.np.load(plan['windows'][0]['token_path'], allow_pickle=False)
        request = json.loads((slot / f'requests/{launch.WINDOW}.request.json').read_text())
        response = json.loads((slot / f'requests/{launch.WINDOW}.response.json').read_text())
        if request != launch.protocol.completion_request(tokens, launch.WINDOW, request['model']):
            raise ValueError('request differs from exact forced-token protocol')
        if launch.protocol.verify_response(response, request) != stage['windows'][0]['response_audit']:
            raise ValueError('response audit differs')
        with launch.stage_adapter(plan, index):
            for phase in ('ready', 'final'):
                raw_log = (slot / f'server-{phase}.private.log').read_text()
                safe = sanitize_log(raw_log)
                windows = stage['windows'] if phase == 'final' else None
                proof = launch.base.runtime_audit(safe.decode(), 'n128', windows)
                if proof != json.loads((slot / f'runtime-{phase}-audit.json').read_text()):
                    raise ValueError('sanitized runtime audit differs')
                output[name + f'/server-{phase}.sanitized.log'] = safe
                preserve(name + f'/runtime-{phase}-audit.json', slot / f'runtime-{phase}-audit.json')
        for filename in ('thermal.jsonl', 'cooldown.jsonl'):
            output[name + '/' + filename] = thermal_projection((slot / filename).read_bytes())
        rows = [json.loads(line) for line in output[name + '/thermal.jsonl'].splitlines()]
        maxima.append(max(max(row['temperatures']) for row in rows))
        preserve(name + '/capture-metadata.json', slot / f'captures/{launch.WINDOW}.capture.json')
        for rank in range(4):
            preserve(name + f'/index-traces/rank-{rank}.json', slot / f'index-traces/rank-{rank}.json')
    output['report.json'] = encoded({'schema': 'glm53-p8.index-trace-snapshot.v2',
        'status': 'terminal-diagnostic-replayed', 'plan_sha256': PLAN_SHA, 'terminal_state': state,
        'sealed_sources_verified': len(plan['source_sha256']), 'independent_processes': 2,
        'captured_rows_per_process': 2047, 'trace_positions': list(range(255, 264)),
        'maximum_gpu_temperature_c_per_repeat': maxima, 'comparison_replayed': True,
        'qualification_pass': False, 'kld_measured': False, 'speed_measurement_valid': False,
        'allocation_restart': False, 'teacher_logits_opened': False, 'protected_roles_opened': [],
        'limits': ['Two fresh processes; four matching trace rows are correlated subsamples, not four independent trials.',
                   'Observer kernels perturb scheduling; no uninstrumented causal intervention was tested.',
                   'Logits, trace NPZ, requests/responses and private logs remain hash-referenced at original local paths.']})
    if terminal_state() != state:
        raise ValueError('terminal state changed')
    launch.verify_identities(plan)
    for path, expected in list(observed.items()):
        reference(Path(path), expected)
    manifest = {'schema': 'glm53-p8.index-trace-snapshot-manifest.v2',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(SOURCE),
        'generator_sha256': launch.pilot.sha(Path(__file__)), 'inputs': observed,
        'outputs': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in output.items()}}
    destination.mkdir(parents=True)
    for name, data in {**output, 'snapshot.json': encoded(manifest)}.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return {'output': str(destination), 'files': len(output) + 1, 'status': 'preserved'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
