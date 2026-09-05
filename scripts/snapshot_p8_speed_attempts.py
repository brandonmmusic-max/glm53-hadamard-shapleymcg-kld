#!/usr/bin/env python3
"""Preserve terminal speed attempts, including failures, as exact artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1')
REPO = Path(__file__).resolve().parents[1]
ALLOWED = ('speed-v1', 'speed-v2', 'speed-v2a1', 'speed-v2a2', 'speed-v2a3')


def save(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'immutable snapshot differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as out:
            out.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', nargs='+', choices=ALLOWED, required=True)
    args = parser.parse_args()
    for attempt in args.attempt:
        unit = f'glm53-uniform-p8-{attempt}.service'
        state = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                         '-p', 'ActiveState', '--value'], text=True).strip()
        if state not in ('failed', 'inactive'):
            raise ValueError(f'not a terminal attempt: {unit}: {state}')
        source = ROOT / attempt
        dest = REPO / 'evidence/opened/codec-v2/p8-speed-attempts' / attempt
        paths = [source / name for name in ('followon.log', 'execution.json', 'server-final.log',
                                            'fullmodel-completion-audit.json', 'analysis.json',
                                            'speed-receipt-audit.json')]
        for slot in sorted(source.glob('round-*')):
            paths.extend(slot / name for name in ('benchmark.json', 'benchmark.log', 'server.log',
                                                  'models.json', 'receipt.json',
                                                  'nvidia-before.xml', 'nvidia-after.xml'))
        rows = []
        for path in paths:
            if not path.is_file():
                continue
            data = path.read_bytes()
            relative = str(path.relative_to(source))
            save(dest / relative, data)
            rows.append({'path': relative, 'source': str(path), 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()})
        save(dest / 'snapshot.json', (json.dumps({'schema': 'glm53-p8-terminal-speed-snapshot.v1',
              'unit': unit, 'terminal_state': state, 'files': rows,
              'note': 'Exact allowlisted logs, metrics, and hardware receipts. No weights, logits, input token arrays, or container environment dumps.'},
              indent=2, sort_keys=True) + '\n').encode())
        print(json.dumps({'attempt': attempt, 'state': state, 'files': len(rows)}))


if __name__ == '__main__':
    main()
