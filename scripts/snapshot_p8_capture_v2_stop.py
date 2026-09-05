#!/usr/bin/env python3
"""Snapshot v2's unpaired N128 capture and N64 prelaunch port stop, never logits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-capture-v2')
DEPENDENCIES = Path('/home/brandonmusic/.local/lib/python3.12/site-packages')
PLAN = ORIGINAL / 'experiments/p8-forced-m1-v2-v2.json'
PLAN_SHA = '0faf9049d5f565e887570a6eff6ceac431ad4b6d767f2a72ab4b324b348d29c3'
ROOT_SHA = 'dd8c8bc17acd7f04ed0a4b43194d83719acf06571b7a335893907fad1faae5c0'
SOURCE = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/forced-m1-v2-v2')
DEST = REPO / 'evidence/opened/codec-v2/p8-forced-m1-v2-v2-stop'
IMAGE = 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9'
WINDOW = 'conditional-fit-0056'
RAW_NAME = f'captures/{WINDOW}.logits.f32'
META_NAME = f'captures/{WINDOW}.capture.json'
UNIT = 'glm53-p8-forced-m1-v2-v2.service'
PORT_ERROR = '[Errno 98] Address already in use'
N128_FILES = {META_NAME, RAW_NAME, 'container-final.private.json', 'container.cid', 'container.private.json',
              'cooldown.jsonl', 'launch-spec.private.json', 'models.json', 'nvidia-after.xml', 'nvidia-before.xml',
              f'requests/{WINDOW}.request.json', f'requests/{WINDOW}.response.json', 'runtime-final-audit.json',
              'runtime-ready-audit.json', 'server-final.private.log', 'server-ready.private.log', 'thermal.jsonl'}
N64_FILES = {'error.private.txt', 'nvidia-after.xml'}


def receipt(path):
    path = Path(path)
    if path != path.resolve() or not path.is_file():
        raise ValueError('canonical regular input required')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed while hashing')
    return {'bytes': before.st_size, 'sha256': digest}


def terminal_state():
    text = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                    '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    state = dict(line.split('=', 1) for line in text.strip().splitlines())
    if state != {'ActiveState': 'failed', 'Result': 'exit-code', 'ExecMainStatus': '1'}:
        raise ValueError('v2 unit must remain terminal failed')
    return state


def validate_original():
    """Replay original frozen validators; -I -B avoids new code and pycache writes."""
    code = '''
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]); planpath=Path(sys.argv[2]); source=Path(sys.argv[3])
plan=json.loads(planpath.read_text())
for relative,digest in plan['source_sha256'].items():
 p=root/relative
 assert p==p.resolve() and p.is_relative_to(root)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
# Isolated Python does not include user packages. Add this explicit existing
# dependency directory without site.addsitedir or execution of .pth files.
sys.path.insert(0,sys.argv[4])
sys.path.insert(0,str(root))
import numpy as np
from glm53_nvfp4 import p8_decode_protocol as protocol, p8_decode_launcher as launcher
w=plan['windows'][0]; assert w['id']=='conditional-fit-0056'
tokens=np.load(w['token_path'],allow_pickle=False)
assert protocol.sha(Path(w['token_path']))==w['input_sha256']
slot=source/'canary-n128'
request=json.loads((slot/'requests'/f"{w['id']}.request.json").read_text())
assert request==protocol.completion_request(tokens,w['id'],launcher.model_name('n128'))
response=json.loads((slot/'requests'/f"{w['id']}.response.json").read_text())
response_proof=protocol.verify_response(response,request)
array,metadata=protocol.load_capture(slot/'captures',w['id'],tokens)
assert array.shape==(2047,154880)
del array
proofs={}
for phase,completed in [('ready',None),('final',[w])]:
 proof=launcher.runtime_audit((slot/f'server-{phase}.private.log').read_text(),'n128',completed)
 assert proof==json.loads((slot/f'runtime-{phase}-audit.json').read_text())
 proofs[phase]=proof
print(json.dumps({'response':response_proof,'runtime':proofs,'rows':metadata['rows_completed'],
 'numpy_version':np.__version__,'numpy_module':np.__file__,
 'raw_sha256':metadata['raw_sha256'],'source_validation':'all original plan source hashes match'}))
'''
    return json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', code,
                                              str(ORIGINAL), str(PLAN), str(SOURCE), str(DEPENDENCIES)], text=True))


def snapshot(destination=DEST):
    destination = Path(destination)
    if not destination.is_absolute() or destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical output required')
    terminal = terminal_state()
    inputs, initial_stats = {}, {}
    def check(path, expected=None):
        actual = receipt(path)
        if expected is not None and actual != expected:
            raise ValueError(f'raw receipt mismatch: {Path(path).name}')
        inputs[str(path)] = actual
        st = Path(path).stat()
        initial_stats[str(path)] = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
        return actual
    if check(PLAN)['sha256'] != PLAN_SHA or PLAN.with_suffix('.sha256').read_text().split()[0] != PLAN_SHA:
        raise ValueError('original plan seal mismatch')
    check(PLAN.with_suffix('.sha256'))
    plan = json.loads(PLAN.read_text())
    if Path(plan['output']) != SOURCE or plan['capture_image'] != IMAGE:
        raise ValueError('v2 target identity mismatch')
    if check(SOURCE / 'execution.json')['sha256'] != ROOT_SHA:
        raise ValueError('v2 terminal receipt mismatch')
    root = json.loads((SOURCE / 'execution.json').read_text())
    if (root['exit_code'] != 1 or root['plan_sha256'] != PLAN_SHA or root['capture_image'] != IMAGE
            or root['protected_roles_opened'] != [] or root['allocation_restart'] is not False
            or root['speed_measurement_valid'] is not False or root['final_identity_audit'] != {'ok': True}
            or root['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
            or root['prior'] != {'backend': True, 'timer': False}
            or root['restoration'] != {'backend': True, 'timer': False, 'errors': []}
            or [s['slot'] for s in root['stages']] != ['canary-n128', 'canary-n64']):
        raise ValueError('v2 stop or restoration shape mismatch')
    stages = {}
    for index, name in enumerate(('canary-n128', 'canary-n64')):
        folder = SOURCE / name
        if check(folder / 'execution.json')['sha256'] != root['stages'][index]['execution_sha256']:
            raise ValueError('stage receipt chain mismatch')
        stage = stages[name] = json.loads((folder / 'execution.json').read_text())
        if (stage['exit_code'] != index or stage['capture_image'] != IMAGE or stage['plan_sha256'] != PLAN_SHA
                or stage['cleanup'] != {'ok': True, 'errors': []} or stage['protected_roles_opened'] != []
                or stage['unreadable_artifacts'] != []
                or set(stage['files']) != (N128_FILES if index == 0 else N64_FILES)):
            raise ValueError('stage outcome or artifact inventory mismatch')
        for relative, expected in stage['files'].items():
            check(folder / relative, expected)
    reference, stopped = stages['canary-n128'], stages['canary-n64']
    if (len(reference['windows']) != 1 or reference['windows'][0]['id'] != WINDOW
            or reference['windows'][0]['rows'] != 2047 or stopped['windows'] != []
            or stopped.get('container_id') is not None or stopped['error_type'] != 'OSError'
            or (SOURCE / 'canary-n64/error.private.txt').read_text() != PORT_ERROR):
        raise ValueError('expected unpaired capture and prelaunch port stop missing')
    for child in ('requests', 'captures'):
        folder = SOURCE / 'canary-n64' / child
        if folder != folder.resolve() or not folder.is_dir() or any(folder.iterdir()):
            raise ValueError('N64 evaluation artifact unexpectedly exists')
    for child in ('full-n128', 'full-n64', 'canary-exact.json', 'full-exact.json'):
        if (SOURCE / child).exists():
            raise ValueError('unexpected pair/full-stage result')
    c = json.loads((SOURCE / 'canary-n128/container-final.private.json').read_text())
    if (c['Id'] != reference['container_id'] or c['Image'] != IMAGE or c['State']['Running'] is not False):
        raise ValueError('N128 container stop unproven')
    proof = validate_original()
    if proof['rows'] != 2047 or proof['raw_sha256'] != reference['windows'][0]['raw_sha256']:
        raise ValueError('original validator disagrees with reference capture')
    outputs = {'execution.json': (SOURCE / 'execution.json').read_bytes()}
    for name, stage in stages.items():
        outputs[name + '/execution.json'] = (SOURCE / name / 'execution.json').read_bytes()
    for name in (META_NAME, 'runtime-ready-audit.json', 'runtime-final-audit.json', 'models.json'):
        outputs['canary-n128/' + name] = (SOURCE / 'canary-n128' / name).read_bytes()
    def encode(value):
        return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    outputs['validation.json'] = encode(proof)
    outputs['port-stop.sanitized.json'] = encode({
        'error_type': 'OSError', 'errno': 98, 'message': 'Address already in use', 'port': 8023,
        'failure_phase': 'prelaunch port reservation before docker create',
        'diagnostic': {'evidence_level': 'main-agent-observed diagnostic; not independently replayable from v2 execution logs',
                       'reported_listener_present': False, 'reported_time_wait_connections': 1,
                       'reported_reuseaddr_bind_succeeded': True,
                       'interpretation': 'Consistent with TIME_WAIT blocking bind without SO_REUSEADDR; not an active competing listener.'}})
    outputs['report.json'] = encode({
        'schema': 'glm53-p8.capture-v2-stop-snapshot.v1', 'status': 'unpaired-reference-captured-candidate-not-started',
        'plan_sha256': PLAN_SHA, 'capture_image': IMAGE, 'n128_captured_windows': 1, 'n128_causal_rows': 2047,
        'n128_one_token_prefill_rows': 1, 'n128_true_decode_rows': 2046, 'n64_started': False, 'n64_captured_rows': 0,
        'paired_closure_passed': False, 'kld_measured': False, 'speed_measurement_valid': False,
        'allocation_restart': False, 'protected_roles_opened': [], 'restoration_verified_from_receipt': True,
        'raw_logits': {**inputs[str(SOURCE / 'canary-n128' / RAW_NAME)], 'local_path': str(SOURCE / 'canary-n128' / RAW_NAME), 'copied_to_repository': False},
        'limits': ['One complete N128 window is useful capture evidence, not a paired pass or domain-general quality evidence.',
                   'Private requests, responses, logs and environment remain local with hash references. No logits are copied.',
                   'TIME_WAIT diagnostic was reported by the main agent after the run; it is labeled separately from immutable run receipts.']})
    for key, expected in initial_stats.items():
        st = Path(key).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) != expected:
            raise ValueError('raw input changed during verification')
    if terminal_state() != terminal:
        raise ValueError('terminal state changed')
    manifest = {'schema': 'glm53-p8.capture-v2-stop-manifest.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'generator': receipt(Path(__file__)), 'terminal_unit': terminal, 'inputs': inputs,
                'outputs': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in outputs.items()}}
    destination.mkdir(parents=True)
    for relative, data in outputs.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with (destination / 'snapshot.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return {'status': 'v2-unpaired-stop-preserved', 'output': str(destination), 'files': len(outputs) + 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
