"""Fresh, read-only payload identity audit before repeated serving launches."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

P8_MANIFEST = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1/manifest.json')
P8_SHA = '9521dae70165d5ba930ca447a37fef97801a6aa8638aef2417a4b206bdc8f191'
EXL3_RECEIPT = Path('/home/brandonmusic/KLC_SANDBOXES/glm-5.3-flash-exl3-4bpw-release/results/materialization-receipt.json')
EXL3_SHA = 'afe588284702c0676b7af5df48bc0e0568bb42b1821808ab71c6dfa1f0c48b61'
EXL3_ROOT = Path('/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(path):
    st = path.stat()
    return {'resolved_path': str(path.resolve()), 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino, 'device': st.st_dev}


def audit_one(row):
    path = Path(row['path'])
    before = fingerprint(path)
    actual = sha(path)
    if (fingerprint(path) != before or actual != row['expected_sha256']
            or row.get('expected_bytes', before['bytes']) != before['bytes']):
        raise ValueError(f'payload changed or differs: {path.name}')
    return {**row, 'sha256': actual, 'stat': before}


def audit():
    if sha(P8_MANIFEST) != P8_SHA or sha(EXL3_RECEIPT) != EXL3_SHA:
        raise ValueError('authoritative payload manifest identity differs')
    p8, exl3 = json.loads(P8_MANIFEST.read_text()), json.loads(EXL3_RECEIPT.read_text())
    if ([x['layer'] for x in p8['layers']] != list(range(3, 45))
            or any([r['rank'] for r in x['ranks']] != [0, 1, 2, 3] for x in p8['layers'])
            or not exl3['complete'] or len(exl3['shards']) != 120
            or set(exl3['shards']) != set(exl3['shard_sha256'])):
        raise ValueError('complete payload inventory differs')
    rows = [{'arm': 'p8', 'path': rank['path'], 'expected_sha256': rank['sha256'],
             'expected_bytes': rank['bytes']} for layer in p8['layers'] for rank in layer['ranks']]
    for name in sorted(exl3['shards']):
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('nonlocal EXL3 shard name')
        rows.append({'arm': 'exl3', 'path': str(EXL3_ROOT / name),
                     'expected_sha256': exl3['shard_sha256'][name]})
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(audit_one, rows))
    return {'schema': 'glm53-p8-exl3-fresh-payload-identity.v1', 'status': 'pass',
            'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
            'manifests': {str(P8_MANIFEST): P8_SHA, str(EXL3_RECEIPT): EXL3_SHA},
            'files': results, 'counts': {'p8': 168, 'exl3': 120},
            'bytes': sum(r['stat']['bytes'] for r in results),
            'protected_roles_opened': [],
            'limits': 'P8 routed sidecars and all materialized EXL3 weight shards only; not a fresh audit of the P8 NVFP4 carrier or unrelated metadata.'}


def verify_stats(receipt):
    if receipt.get('schema') != 'glm53-p8-exl3-fresh-payload-identity.v1' or receipt.get('status') != 'pass':
        raise ValueError('fresh identity prerequisite failed')
    if (receipt['counts'] != {'p8': 168, 'exl3': 120} or len(receipt['files']) != 288
            or len({row['path'] for row in receipt['files']}) != 288
            or dict(Counter(row['arm'] for row in receipt['files'])) != receipt['counts']):
        raise ValueError('payload identity inventory differs')
    for row in receipt['files']:
        if row['sha256'] != row['expected_sha256'] or fingerprint(Path(row['path'])) != row['stat']:
            raise ValueError('payload metadata changed after fresh byte audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh receipt path required')
    value = audit()
    with args.output.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'status': value['status'], 'counts': value['counts'],
                      'bytes': value['bytes'], 'receipt_sha256': sha(args.output)}))
