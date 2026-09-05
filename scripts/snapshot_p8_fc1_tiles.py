#!/usr/bin/env python3
"""Preserve terminal FC1 device attempts, including failed gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from snapshot_p8_decode_profile import save

ROOT=Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1')
REPO=Path(__file__).resolve().parents[1]


def main(version):
    if not 1 <= version <= 20:
        raise ValueError('invalid version')
    name=f'fc1tiles-device-v{version}'
    source=ROOT/name
    dest=REPO/'evidence/opened/codec-v2'/f'p8-{name}'
    unit=f'glm53-p8-fc1tiles-device-v{version}.service'
    state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
    if state not in ('inactive','failed'):
        raise ValueError('attempt is not terminal')
    execution=json.loads((source/'execution.json').read_text())
    if execution['protected_roles_opened'] != []:
        raise ValueError('unexpected protected role')
    files=['execution.json']
    files += [name for name in ('nvidia-before.xml','analysis.json','failed-probe.log','failed-container-state.json') if (source/name).is_file()]
    for gpu in range(4):
        for name in ('launch.json','container-id.txt','container-state.json','probe.log','thermal.jsonl','result.json'):
            relative=f'gpu{gpu}/{name}'
            if (source/relative).is_file():
                files.append(relative)
    records=[]
    for relative in files:
        data=(source/relative).read_bytes()
        save(dest/relative,data)
        records.append({'path':relative,'source':str(source/relative),
                        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    save(dest/'snapshot.json',(json.dumps({
        'schema':'glm53-p8-fc1-tiles-snapshot.v1','unit':unit,'terminal_state':state,
        'files':records,'note':'Preserves original attempts and failures; execution exit zero does not imply correctness or speed gates passed. Synthetic inputs only; no environment dumps.'},indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps({'destination':str(dest),'files':len(records)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--version',type=int,required=True)
    main(p.parse_args().version)
