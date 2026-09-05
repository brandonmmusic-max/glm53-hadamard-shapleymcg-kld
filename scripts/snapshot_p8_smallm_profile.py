#!/usr/bin/env python3
"""Snapshot a completed synthetic trace, preserving its failed strict gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from snapshot_p8_decode_profile import save

SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/profile-v1')
REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'evidence/opened/codec-v2/p8-smallm-profile-v1'
UNIT = 'glm53-p8-smallm-profile-v1.service'
FILES = (
    'analysis-diagnostic.json', 'strict-analysis-failure.json',
    'execution.json', 'export.log', 'models.json',
    'smallm-c1.nsys-rep', 'smallm-c1.sqlite',
    'nvidia-before.xml', 'nvidia-after.xml', 'server-final.log',
    'thermal.jsonl', 'graph-capture-audit.json',
    'client/config.json', 'client/prompt-construction.json',
    'client/receipt.json', 'client/receipt.sha256',
)


def main():
    state = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                    '-p', 'ActiveState', '--value'], text=True).strip()
    result = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                     '-p', 'Result', '--value'], text=True).strip()
    if (state, result) != ('inactive', 'success'):
        raise ValueError(f'capture is not terminal-success: {state}/{result}')
    execution = json.loads((SOURCE/'execution.json').read_text())
    analysis = json.loads((SOURCE/'analysis-diagnostic.json').read_text())
    failure = json.loads((SOURCE/'strict-analysis-failure.json').read_text())
    if execution['exit_code'] != 0 or execution['protected_roles_opened'] != []:
        raise ValueError('execution failed or protected roles opened')
    for service in ('backend', 'timer'):
        if execution[f'prior_{service}_active'] != execution[f'restored_{service}_active']:
            raise ValueError('prior service state not restored')
    if analysis['status'] != 'diagnostic-complete-prefix' or failure['status'] != 'fail':
        raise ValueError('diagnostic/failure status changed')
    if analysis['original_trace_gate_passed'] is not False:
        raise ValueError('strict gate must remain failed')
    rows = []
    for relative in FILES:
        data = (SOURCE/relative).read_bytes()
        save(DEST/relative, data)
        rows.append({'path': relative, 'source': str(SOURCE/relative),
                     'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    snapshot = {
        'schema': 'glm53-p8-smallm-profile-snapshot.v1',
        'unit': UNIT, 'terminal_state': state, 'result': result, 'files': rows,
        'original_trace_gate_passed': False,
        'note': 'Complete-prefix diagnostic only. Original strict failure and unmodified raw trace preserved. Private environment dumps omitted. No protected inputs.',
    }
    save(DEST/'snapshot.json', (json.dumps(snapshot, indent=2, sort_keys=True)+'\n').encode())
    print(json.dumps({'status': 'snapshotted', 'files': len(rows), 'destination': str(DEST)}))


if __name__ == '__main__':
    main()
