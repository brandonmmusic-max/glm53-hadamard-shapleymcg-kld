#!/usr/bin/env python3
"""Snapshot a terminal five-cold comparison without publishing private dumps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from glm53_nvfp4 import p8_fc1_cold_compare as cold
from scripts.snapshot_p8_fc1_integration import sanitize_log as sanitize_graph_dispatch

SOURCE = cold.ROOT / 'fc1-cold-comparison-v1'
DEST = REPO / 'evidence/opened/codec-v2/p8-fc1-cold-comparison-v1'
UNIT = 'glm53-p8-fc1-cold-comparison-v1.service'
SLOT_FILES = ('result.json', 'summary.json', 'runtime-audit.json', 'cooldown.jsonl', 'thermal.jsonl',
              'models.json', 'nvidia-before.xml', 'nvidia-after.xml')
LOGS = ('server-ready.private.log', 'server-final.private.log')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def terminal_state():
    result = subprocess.run(['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'Result'],
                            check=True, capture_output=True, text=True, timeout=30)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if state.get('ActiveState') not in ('inactive', 'failed') or not state.get('Result'):
        raise ValueError('comparison is not observably terminal')
    return state


def sanitize_log(text):
    """Canonical typed graph/dispatch/topology fields; no source lines copied."""
    rows = sanitize_graph_dispatch(text).decode().splitlines()
    for line in text.splitlines():
        for field in ('tensor_parallel_size', 'decode_context_parallel_size'):
            rows.extend(f'{field}={int(match[1])}' for match in re.finditer(r'\b' + field + r'=(\d+)\b', line))
        for match in re.finditer(r"'enable_expert_parallel': (True|False)\b", line):
            rows.append(f"'enable_expert_parallel': {match[1]}")
        for field in ('speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla', 'quantization=exl3'):
            if re.search(r'\b' + re.escape(field) + r'\b', line):
                rows.append(field)
        for marker in ('EXL3 full-expert EP runtime planned',
                       'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts'):
            if marker in line:
                rows.append(marker)
    return ('\n'.join(rows) + ('\n' if rows else '')).encode()


def replay_runtime(text, arm):
    # Exact integer checks complement the producer's substring-based config checks.
    if (set(re.findall(r'^tensor_parallel_size=(\d+)$', text, re.M)) != {'4'}
            or set(re.findall(r'^decode_context_parallel_size=(\d+)$', text, re.M)) != {str(cold.TOPOLOGY[arm]['dcp'])}):
        raise ValueError('sanitized topology integer inventory differs')
    return cold.runtime_audit(text, arm)


def verify_plan_for_snapshot(path):
    """Authenticate immutable design/code, not current mutable weight stats.

    A metadata-drift failure must remain preservable. A successful comparison
    additionally replays the producer's full analyzer and prerequisite checks.
    """
    if cold.pilot.sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('comparison plan seal differs')
    plan = json.loads(path.read_text())
    if (any(plan.get(key) != value for key, value in cold.FIXED.items())
            or plan['prerequisite_sha256'] != {str(p): value for p, value in cold.PINNED.items()}
            or not cold.SOURCES <= plan['source_sha256'].keys()):
        raise ValueError('immutable comparison design differs')
    for relative, expected in plan['source_sha256'].items():
        source = cold.REPO / relative
        if source != source.resolve() or not source.is_relative_to(cold.REPO) or cold.pilot.sha(source) != expected:
            raise ValueError('frozen comparison source differs')
    if cold.pilot.sha(cold.pilot.BENCH) != plan['benchmark_sha256']:
        raise ValueError('frozen benchmark source differs')
    return plan


def snapshot(plan_path):
    if (SOURCE != SOURCE.resolve() or SOURCE.parent != cold.ROOT or DEST != DEST.resolve()
            or plan_path != plan_path.resolve()):
        raise ValueError('canonical source/destination/plan paths required')
    if DEST.exists():
        raise ValueError('snapshot exists; no overwrite or resume')
    state = terminal_state()
    plan = verify_plan_for_snapshot(plan_path)
    if Path(plan['output']) != SOURCE:
        raise ValueError('plan output differs from this snapshot target')
    plan_hash = cold.pilot.sha(plan_path)
    observed, staged, files, omitted, missing = {}, {}, [], [], []

    def read(path, expected=None):
        if path != path.resolve() or not path.is_file():
            raise ValueError('nonlocal, symlink, or missing raw artifact')
        data = path.read_bytes()
        actual = {'bytes': len(data), 'sha256': sha(data)}
        if expected is not None and actual != expected:
            raise ValueError(f'raw receipt hash differs: {path.name}')
        observed[path] = actual
        return data

    def preserve(relative, path, data, transform='none'):
        staged[relative] = data
        row = {'path': relative, 'source': str(path), 'bytes': len(data), 'sha256': sha(data), 'transformation': transform}
        if transform != 'none':
            row.update(raw_bytes=observed[path]['bytes'], raw_sha256=observed[path]['sha256'])
        files.append(row)

    def exclude(relative, path, data, reason):
        omitted.append({'path': relative, 'source': str(path), 'bytes': len(data), 'sha256': sha(data), 'reason': reason})

    raw_execution = read(SOURCE / 'execution.json')
    execution = json.loads(raw_execution)
    if (execution.get('schema') != 'glm53-p8-fc1-cold-comparison-execution.v1'
            or execution.get('plan_sha256') != plan_hash or execution.get('protected_roles_opened') != []
            or execution.get('allocation_restart') is not False
            or type(execution.get('exit_code')) is not int or execution['exit_code'] not in (0, 1)
            or type(execution.get('completed_slots')) is not int or not 0 <= execution['completed_slots'] <= 10
            or not execution.get('finished_at')):
        raise ValueError('terminal execution is not the sealed comparison')
    preserve('execution.json', SOURCE / 'execution.json', raw_execution)
    terminal_success = (execution['exit_code'] == 0 and state == {'ActiveState': 'inactive', 'Result': 'success'})
    # Keep the original basename referenced by the checksum sidecar.
    preserve(plan_path.name, plan_path, read(plan_path))
    preserve(plan_path.with_suffix('.sha256').name, plan_path.with_suffix('.sha256'), read(plan_path.with_suffix('.sha256')))
    attempts = execution['runs']
    ordered = [cold.slot_name(entry) for entry in cold.ORDER]
    if ([row['slot'] for row in attempts] != ordered[:len(attempts)] or len(attempts) > 10
            or not execution['completed_slots'] <= len(attempts) <= execution['completed_slots'] + 1):
        raise ValueError('terminal attempted-slot prefix differs from sealed order')
    directories = {path.name for path in SOURCE.iterdir() if path.is_dir()}
    if not directories <= set(ordered) or directories != set(ordered[:len(directories)]):
        raise ValueError('raw slot directories are not an ordered attempt prefix')
    if not len(attempts) <= len(directories) <= min(10, len(attempts) + 1):
        raise ValueError('unreceipted slots exceed the one interrupted-attempt allowance')

    for name in ('hardware-inventory.json', 'weight-audit-prerequisite.json', 'analysis.json'):
        path = SOURCE / name
        if not path.is_file():
            missing.append(name)
            continue
        data = read(path)
        if name == 'weight-audit-prerequisite.json':
            # Producer reserializes the immutable source receipt; compare content.
            original = read(Path(plan['weight_audit']))
            if sha(original) != plan['weight_audit_sha256'] or json.loads(data) != json.loads(original):
                raise ValueError('payload prerequisite differs from sealed fresh hash audit')
        if name == 'analysis.json' and not terminal_success:
            preserve('raw-unverified-analysis.json', path, data,
                     'unchanged bytes renamed: UNVERIFIED; excluded from all qualification claims')
        else:
            preserve(name, path, data)
    if (SOURCE / 'error.private.txt').is_file():
        path = SOURCE / 'error.private.txt'
        exclude('error.private.txt', path, read(path), 'private exception text omitted')

    rows = []
    for index, entry in enumerate(cold.ORDER):
        name = cold.slot_name(entry)
        if name not in directories:
            continue
        slot = SOURCE / name
        receipt_path = slot / 'execution.json'
        receipt = None
        if receipt_path.is_file():
            data = read(receipt_path)
            if index < len(attempts) and sha(data) != attempts[index]['execution_sha256']:
                raise ValueError('slot execution differs from top-level receipt')
            receipt = json.loads(data)
            if (receipt.get('schema') != 'glm53-p8-fc1-cold-run.v1' or receipt.get('plan_sha256') != plan_hash
                    or receipt.get('round') != entry['round'] or receipt.get('arm') != entry['arm']
                    or receipt.get('image') != cold.IMAGES[entry['arm']] or receipt.get('topology') != cold.TOPOLOGY[entry['arm']]
                    or receipt.get('protected_roles_opened') != [] or receipt.get('allocation_restart') is not False
                    or receipt.get('exit_code') not in (0, 1) or not receipt.get('finished_at')):
                raise ValueError('slot execution identity differs')
            preserve(f'{name}/execution.json', receipt_path, data)
            if index < execution['completed_slots'] and receipt['exit_code'] != 0:
                raise ValueError('completed-slot count contradicts its execution receipt')
        elif index < len(attempts):
            raise ValueError('root-bound slot execution is missing')
        else:
            missing.append(f'{name}/execution.json')
        inventory = receipt.get('files', {}) if receipt else {}
        # Hash all inventory items, but publish only the explicit safe whitelist.
        raw_files = {}
        for path in slot.iterdir():
            if not path.is_file() or path.name == 'execution.json':
                continue
            if path.name in inventory:
                data = read(path, inventory[path.name])
            else:
                data = read(path)
                if receipt and receipt['exit_code'] == 0:
                    raise ValueError('successful slot has an unreceipted raw file')
            raw_files[path.name] = data
        if not inventory.keys() <= raw_files.keys():
            raise ValueError('receipted slot artifact is missing')
        for filename in (*SLOT_FILES, *LOGS):
            if filename not in raw_files:
                missing.append(f'{name}/{filename}')
        for filename, data in raw_files.items():
            path, relative = slot / filename, f'{name}/{filename}'
            if filename in LOGS:
                target = f'{name}/{filename.replace(".private.log", ".sanitized.log")}'
                preserve(target, path, sanitize_log(data.decode(errors='replace')), 'canonical graph/topology/backend/dispatch fields only')
            elif filename in SLOT_FILES:
                if filename == 'result.json':
                    try:
                        parsed = json.loads(data)
                    except (UnicodeError, json.JSONDecodeError):
                        exclude(relative, path, data, 'incomplete result requires separate privacy review')
                        continue
                    if parsed.get('startup_diagnostics', {}).get('env', {}):
                        raise ValueError('benchmark environment requires separate privacy review')
                preserve(relative, path, data)
            else:
                exclude(relative, path, data, 'private or non-whitelisted contents omitted')
        row = {**entry, 'slot': name, 'root_receipted': index < len(attempts),
               'exit_code': receipt['exit_code'] if receipt else None,
               'status': 'unreceipted-interrupted-slot' if index >= len(attempts) else 'failed-or-incomplete'}
        if 'thermal.jsonl' in raw_files:
            try:
                readings = [json.loads(line)['temperatures'] for line in raw_files['thermal.jsonl'].decode().splitlines()]
                if readings and all(len(values) == 4 and all(math.isfinite(v) for v in values) for values in readings):
                    row.update(thermal_samples=len(readings), maximum_sampled_gpu_temperature_c=max(map(max, readings)))
            except (ValueError, KeyError, TypeError, UnicodeError):
                pass  # Preserve partial raw telemetry without inventing a summary.
        if receipt and receipt['exit_code'] == 0:
            required = {'result.json', 'summary.json', 'runtime-audit.json', 'thermal.jsonl', 'cooldown.jsonl',
                        'server-final.private.log'}
            if not required <= raw_files.keys():
                raise ValueError('successful slot lacks complete public-replay inputs')
            summary = cold.summarize(json.loads(raw_files['result.json']), entry['arm'])
            if summary != json.loads(raw_files['summary.json']):
                raise ValueError('slot metric summary does not replay')
            proof = replay_runtime(sanitize_log(raw_files['server-final.private.log'].decode(errors='replace')).decode(), entry['arm'])
            if proof != json.loads(raw_files['runtime-audit.json']):
                raise ValueError('sanitized slot runtime audit does not replay')
            row.update(metrics=summary, runtime_audit=proof,
                       status='completed-slot' if index < len(attempts) else 'completed-but-not-root-receipted')
        rows.append(row)

    analysis = None
    if terminal_success:
        if len(attempts) != 10 or execution['completed_slots'] != 10 or 'analysis.json' not in staged:
            raise ValueError('successful terminal comparison lacks ten runs or analysis')
        analysis = cold.analyze(SOURCE, plan_path)
        if analysis != json.loads(staged['analysis.json']):
            raise ValueError('sealed ten-run analysis does not replay')
        if any(row['status'] != 'completed-slot' for row in rows):
            raise ValueError('successful comparison has an incomplete slot')
    report = {'schema': 'glm53-p8-fc1-cold-comparison-report.v1',
              'status': 'terminal-comparison-complete' if terminal_success else 'terminal-failure-preserved',
              'execution_exit_code': execution['exit_code'], 'terminal_state': state,
              'attempted_slots': len(directories), 'root_receipted_slots': len(attempts),
              'completed_slots': execution['completed_slots'], 'slots': rows,
              'analysis_replayed': analysis is not None,
              'speed_gate_pass': analysis['speed_gate_pass'] if analysis else None,
              'medians': analysis['medians'] if analysis else None,
              'primary_wins': analysis['primary_wins'] if analysis else None,
              'topology': cold.TOPOLOGY, 'allocation_restart': False,
              'numerical_closure_required_separately': True,
              'note': ('Five new server processes per arm are the experimental units; warm compiled caches and '
                       'different product topologies remain explicit. Primary prefill is server validation, '
                       'not the client headline. Failure/partial slots are never substituted or promoted. '
                       'FULL graph capture receipts are not a trace of measured-request replay. '
                       'No numerical-closure or allocation authorization follows from the speed gate.')}
    staged['report.json'] = json_bytes(report)
    if terminal_state() != state:
        raise ValueError('unit changed while snapshotting')
    for path, expected in observed.items():
        if {'bytes': path.stat().st_size, 'sha256': cold.pilot.sha(path)} != expected:
            raise ValueError('raw artifact changed during snapshot')
    manifest = {'schema': 'glm53-p8-fc1-cold-comparison-snapshot.v1', 'unit': UNIT,
                'source': str(SOURCE), 'plan_sha256': plan_hash, 'terminal_state': state,
                'status': report['status'], 'files': files, 'omitted_raw_files': omitted,
                'analysis_replayed': report['analysis_replayed'],
                'missing_artifacts': missing, 'weight_audit_sha256': plan['weight_audit_sha256'],
                'report_sha256': sha(staged['report.json']), 'allocation_restart': False,
                'protected_roles_opened': [],
                'note': 'No raw source mutation. Private container/launch/error contents and arbitrary log lines excluded; hashes retained.'}
    staged['snapshot.json'] = json_bytes(manifest)
    DEST.mkdir(parents=True, exist_ok=False)
    for relative, data in staged.items():
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(data)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    report = snapshot(args.plan)
    print(json.dumps({'status': report['status'], 'destination': str(DEST),
                      'completed_slots': report['completed_slots'], 'speed_gate_pass': report['speed_gate_pass']}))
