#!/usr/bin/env python3
"""Snapshot completed, opened-role P8 evidence without logits or token IDs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1')
CAMPAIGN = Path('/media/brandonmusic/klcstore/bmxfp4-glm53')
RUN = 'p8-uniform-all42-native-cf32-v1'
DEST = REPO / 'evidence/opened/codec-v2/uniform-p8-all42-v1'


def save(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to replace different evidence: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    audit = json.loads((ROOT / 'fullmodel-completion-audit.json').read_text())
    assert audit['status'] == 'pass' and audit['windows'] == 32
    assert audit['causal_positions'] == 65504
    assert audit['native_dispatch'] == {'forward': 168, 'weights_ready': 168}
    sources = {}
    for name in ('manifest.json', 'fullmodel-kld-execution.json',
                 'fullmodel-kld-vs-decoded-gptq-context.json',
                 'fullmodel-completion-audit.json', 'runtime-patch-manifest-efd250b.json'):
        sources[name] = ROOT / name
    for layer in range(3, 45):
        name = f'receipts/layer-{layer:03d}.json'
        sources[name] = ROOT / name
    sources['run.json'] = CAMPAIGN / f'kld-v3/run-{RUN}.json'
    for name in ('server-final.log', 'session.log'):
        sources[name] = CAMPAIGN / f'kld-v3/sessions/{RUN}/{name}'
    # Preserve the terminal speed attempt, not an in-progress benchmark snapshot.
    for name in ('execution.json', 'round-01-exl3/server.log'):
        sources[f'speed-v1-failed/{name}'] = ROOT / 'speed-v1' / name
    source_rows = []
    for relative, source in sorted(sources.items()):
        data = source.read_bytes()
        save(DEST / relative, data)
        source_rows.append({'source': str(source), 'snapshot': relative,
                            'sha256': hashlib.sha256(data).hexdigest()})
    records = CAMPAIGN / f'kld-v3/records/{RUN}'
    assert len(list(records.glob('*.parquet'))) == 32
    projected_rows = []
    for source in sorted(records.glob('*.parquet')):
        table = pq.read_table(source, columns=['window_id', 'domain', 'pos', 'kld'])
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(table.column_names)
        for row in zip(*(table[name].to_pylist() for name in table.column_names)):
            writer.writerow(row)
        relative = f'causal-kld/{source.stem}.csv'
        data = stream.getvalue().encode()
        save(DEST / relative, data)
        projected_rows.append({'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                               'snapshot': relative, 'snapshot_sha256': hashlib.sha256(data).hexdigest(),
                               'rows': table.num_rows})
    save(DEST / 'snapshot.json', (json.dumps({
        'schema': 'glm53-p8-fullmodel-opened-snapshot.v1',
        'exact_copies': source_rows, 'projected_causal_kld': projected_rows,
        'projection_note': 'CSV contains only window_id, domain, pos, kld; not byte-identical to source Parquet. No token IDs, logits, weights, or protected roles are copied.',
    }, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'exact_copies': len(sources), 'projected_windows': len(projected_rows),
                      'positions': sum(row['rows'] for row in projected_rows)}))


if __name__ == '__main__':
    main()
