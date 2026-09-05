#!/usr/bin/env python3
"""Preserve the terminal P8 decode trace and its safe reproduction receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1/decode-profile-v1')
REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'evidence/opened/codec-v2/p8-decode-profile-v1'
UNIT = 'glm53-p8-decode-profile-v1.service'
FILES = (
    'analysis.json',
    'execution.json',
    'export.log',
    'launch-spec.json',
    'models.json',
    'native-p8-c1.nsys-rep',
    'native-p8-c1.sqlite',
    'nvidia-after.xml',
    'nvidia-before.xml',
    'server-final.log',
    'thermal-max.log',
    'client/config.json',
    'client/prompt-construction.json',
    'client/receipt.json',
    'client/receipt.sha256',
)


def save(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'immutable snapshot differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        out.write(data)


def main() -> None:
    state = subprocess.check_output(
        ['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '--value'],
        text=True).strip()
    result = subprocess.check_output(
        ['systemctl', '--user', 'show', UNIT, '-p', 'Result', '--value'],
        text=True).strip()
    if (state, result) != ('inactive', 'success'):
        raise ValueError(f'profile is not terminal-success: {state}/{result}')
    analysis = json.loads((SOURCE / 'analysis.json').read_text())
    execution = json.loads((SOURCE / 'execution.json').read_text())
    if analysis.get('status') != 'pass' or execution.get('protected_roles_opened') != []:
        raise ValueError('analysis/protected-role gate differs')
    rows = []
    for relative in FILES:
        source = SOURCE / relative
        if not source.is_file():
            raise ValueError(f'missing profile artifact: {relative}')
        data = source.read_bytes()
        destination_name = 'client-receipt.json' if relative == 'client/receipt.json' else relative
        save(DEST / destination_name, data)
        rows.append({
            'path': destination_name,
            'source': str(source),
            'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        })
    snapshot = {
        'schema': 'glm53-p8-decode-profile-snapshot.v1',
        'unit': UNIT,
        'terminal_state': state,
        'result': result,
        'files': rows,
        'note': ('Exact trace, exported SQLite, diagnostic analysis, logs and '
                 'hardware receipts. Synthetic prompt bytes and container '
                 'environment dump are omitted; their construction and hashes '
                 'remain receipted.'),
    }
    save(DEST / 'snapshot.json',
         (json.dumps(snapshot, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'status': 'pass', 'files': len(rows), 'destination': str(DEST)}))


if __name__ == '__main__':
    main()
