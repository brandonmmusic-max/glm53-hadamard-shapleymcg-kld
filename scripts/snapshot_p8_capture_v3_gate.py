#!/usr/bin/env python3
"""Preserve v3 exact-gate failure and label cross-run comparisons post hoc."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_stop_receipts', REPO / 'scripts/snapshot_p8_capture_v2_stop.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
SOURCE = prior.SOURCE.parent / 'forced-m1-v2-v3'
PLAN = REPO / 'experiments/p8-forced-m1-v2-v3.json'
PLAN_SHA = '7b675117a4a2c8a5abe87056dae6a3b6bcd1db2d44d8d0b5c45ad0f33b219085'
ROOT_SHA = '4a1c85cc91f0f89a858ad4509d5cb76b05e5c73fe607de01518313b0c8f7e63c'
DEST = REPO / 'evidence/opened/codec-v2/p8-forced-m1-v2-v3-gate'
UNIT = 'glm53-p8-forced-m1-v2-v3.service'


def terminal_state():
    text = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                   '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    state = dict(line.split('=', 1) for line in text.strip().splitlines())
    if state != {'ActiveState': 'failed', 'Result': 'exit-code', 'ExecMainStatus': '1'}:
        raise ValueError('v3 unit is not terminal failed')
    return state


def compare_verified():
    """Read only three existing captures with pinned code; never open teachers."""
    code = '''
import sys,json,hashlib
from pathlib import Path
repo,old,planpath,oldplanpath,root,oldroot=map(Path,sys.argv[1:7])
plans=[json.loads(oldplanpath.read_text()),json.loads(planpath.read_text())]
for source,plan in zip([old,repo],plans):
 for relative,digest in plan['source_sha256'].items():
  p=source/relative
  assert p==p.resolve() and p.is_relative_to(source)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
assert plans[0]['source_sha256']['glm53_nvfp4/p8_decode_protocol.py']==plans[1]['source_sha256']['glm53_nvfp4/p8_decode_protocol.py']
sys.path.insert(0,sys.argv[7]);sys.path.insert(0,str(repo))
import numpy as np
from glm53_nvfp4 import p8_decode_protocol as protocol,p8_decode_launcher as launcher
arrays={};proofs={};metadata={}
for name,source,plan,arm in [('v2_n128',oldroot,plans[0],'n128'),('v3_n128',root,plans[1],'n128'),('v3_n64',root,plans[1],'n64')]:
 w=plan['windows'][0]; assert w['id']=='conditional-fit-0056'
 tokenpath=Path(w['token_path']);assert protocol.sha(tokenpath)==w['input_sha256']
 tokens=np.load(tokenpath,allow_pickle=False);slot=source/f'canary-{arm}'
 request=json.loads((slot/'requests'/f"{w['id']}.request.json").read_text())
 assert request==protocol.completion_request(tokens,w['id'],f'glm53-p8-{source.name}-{arm}')
 response=json.loads((slot/'requests'/f"{w['id']}.response.json").read_text())
 proofs[name]={'response':protocol.verify_response(response,request),'runtime':{}}
 arrays[name],metadata[name]=protocol.load_capture(slot/'captures',w['id'],tokens)
 for phase,completed in [('ready',None),('final',[w])]:
  proof=launcher.runtime_audit((slot/f'server-{phase}.private.log').read_text(),arm,completed)
  assert proof==json.loads((slot/f'runtime-{phase}-audit.json').read_text())
  proofs[name]['runtime'][phase]=proof
for value in metadata.values():
 assert value['sequence_token_ids_sha256']==metadata['v2_n128']['sequence_token_ids_sha256']
 assert value['original_logit_dtype']==metadata['v2_n128']['original_logit_dtype']
 assert value['original_logit_width']==metadata['v2_n128']['original_logit_width']
comparisons={}
for name,a,b in [('v3_n128_vs_v3_n64','v3_n128','v3_n64'),('v2_n128_vs_v3_n128','v2_n128','v3_n128'),('v2_n128_vs_v3_n64','v2_n128','v3_n64')]:
 comparisons[name]=protocol.exact_logits(arrays[a],arrays[b],chunk_rows=8)
gate=json.loads((root/'canary-exact.json').read_text())
assert gate['all_exact'] is False and len(gate['windows'])==1
assert all(gate['windows'][0][k]==v for k,v in comparisons['v3_n128_vs_v3_n64'].items())
print(json.dumps({'comparisons':comparisons,'capture_proofs':proofs,'protocol_sha256':plans[1]['source_sha256']['glm53_nvfp4/p8_decode_protocol.py'],
 'original_sources_verified':True,'chunk_rows':8,'teacher_logits_opened':False,'kld_measured':False,
 'primary_gate_replayed':True,'gate_passed':False,
 'cross_run_evidence_level':'post-hoc diagnostic only; does not alter or excuse original failed exact gate',
 'limits':['Both N128 runs differ in process and campaign; this is not proof of a specific kernel cause.','No paired KLD or teacher scoring is performed.']}))
'''
    return json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', code,
        str(REPO), str(prior.ORIGINAL), str(PLAN), str(prior.PLAN), str(SOURCE), str(prior.SOURCE), str(prior.DEPENDENCIES)], text=True))


def snapshot(destination=DEST):
    destination = Path(destination)
    if not destination.is_absolute() or destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical output required')
    terminal = terminal_state()
    inputs, stats, stages = {}, {}, {}
    def check(path, expected=None):
        actual = prior.receipt(path)
        if expected is not None and actual != expected:
            raise ValueError('raw artifact hash or byte count differs')
        inputs[str(path)] = actual
        st = Path(path).stat()
        stats[str(path)] = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
        return actual
    for source, planpath, planhash, executionhash, slots in (
            (SOURCE, PLAN, PLAN_SHA, ROOT_SHA, ['canary-n128', 'canary-n64']),
            (prior.SOURCE, prior.PLAN, prior.PLAN_SHA, prior.ROOT_SHA, ['canary-n128'])):
        if check(planpath)['sha256'] != planhash or planpath.with_suffix('.sha256').read_text().split()[0] != planhash:
            raise ValueError('campaign plan seal differs')
        check(planpath.with_suffix('.sha256'))
        plan = json.loads(planpath.read_text())
        if Path(plan['output']) != source or plan['capture_image'] != prior.IMAGE:
            raise ValueError('campaign target differs')
        if check(source / 'execution.json')['sha256'] != executionhash:
            raise ValueError('terminal execution differs')
        execution = json.loads((source / 'execution.json').read_text())
        if (execution['exit_code'] != 1 or execution['plan_sha256'] != planhash
                or execution['protected_roles_opened'] != [] or execution['allocation_restart'] is not False
                or execution['restoration'] != {'backend': True, 'timer': False, 'errors': []}
                or execution['restoration_safety'] != {'ok': True, 'errors': [], 'containers': []}
                or [x['slot'] for x in execution['stages']] != ['canary-n128', 'canary-n64']):
            raise ValueError('terminal failure/restoration differs')
        for slotname in slots:
            slot = source / slotname
            digest = next(x['execution_sha256'] for x in execution['stages'] if x['slot'] == slotname)
            if check(slot / 'execution.json')['sha256'] != digest:
                raise ValueError('root-stage chain differs')
            stage = stages[(str(source), slotname)] = json.loads((slot / 'execution.json').read_text())
            if (stage['exit_code'] != 0 or stage['cleanup'] != {'ok': True, 'errors': []}
                    or stage['protected_roles_opened'] != [] or stage['unreadable_artifacts'] != []
                    or set(stage['files']) != prior.N128_FILES or len(stage['windows']) != 1
                    or stage['windows'][0]['id'] != prior.WINDOW or stage['windows'][0]['rows'] != 2047):
                raise ValueError('complete canary inventory differs')
            for relative, expected in stage['files'].items():
                check(slot / relative, expected)
            actual_files = {str(p.relative_to(slot)) for p in slot.rglob('*') if p.is_file()}
            if actual_files != prior.N128_FILES | {'execution.json'}:
                raise ValueError('unreceipted stage artifact exists')
            c = json.loads((slot / 'container-final.private.json').read_text())
            if c['Id'] != stage['container_id'] or c['Image'] != prior.IMAGE or c['State']['Running'] is not False:
                raise ValueError('container stop proof differs')
    if any((SOURCE / name).exists() for name in ('full-n128', 'full-n64', 'full-exact.json')):
        raise ValueError('full stage unexpectedly exists')
    check(SOURCE / 'canary-exact.json')
    gate = json.loads((SOURCE / 'canary-exact.json').read_text())
    if gate['all_exact'] is not False:
        raise ValueError('expected failed primary gate')
    diagnostic = compare_verified()
    if diagnostic['primary_gate_replayed'] is not True or diagnostic['gate_passed'] is not False:
        raise ValueError('failed primary gate did not replay')
    def encode(value):
        return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    outputs = {name: (SOURCE / name).read_bytes() for name in ('execution.json', 'canary-exact.json')}
    raw_refs = {}
    for (source, slotname), stage in stages.items():
        prefix = ('v3' if source == str(SOURCE) else 'v2') + '-' + slotname
        slot = Path(source) / slotname
        for name in ('execution.json', prior.META_NAME, 'runtime-ready-audit.json', 'runtime-final-audit.json'):
            outputs[prefix + '/' + name] = (slot / name).read_bytes()
        for name in ('thermal.jsonl', 'cooldown.jsonl'):
            rows = []
            for line in (slot / name).read_text().splitlines():
                row = json.loads(line)
                datetime.fromisoformat(row['at'])
                values = row['temperatures']
                if len(values) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 150 for v in values):
                    raise ValueError('thermal data malformed')
                rows.append(json.dumps({'at': row['at'], 'temperatures': values}, sort_keys=True))
            outputs[prefix + '/' + name] = ('\n'.join(rows) + '\n').encode()
        raw_refs[prefix] = {'local_path': str(slot / prior.RAW_NAME), **inputs[str(slot / prior.RAW_NAME)], 'copied_to_repository': False}
    outputs['three-capture-diagnostics.json'] = encode(diagnostic)
    outputs['report.json'] = encode({'schema': 'glm53-p8.capture-v3-gate-snapshot.v1', 'status': 'exact-gate-failed',
        'plan_sha256': PLAN_SHA, 'n128_rows': 2047, 'n64_rows': 2047, 'paired_gate_passed': False,
        'full_stage_started': False, 'kld_measured': False, 'speed_measurement_valid': False,
        'allocation_restart': False, 'protected_roles_opened': [], 'raw_logits': raw_refs,
        'repeat_comparison_role': 'post-hoc diagnostic; original failed gate remains unchanged',
        'limits': ['The N128 cross-run difference prevents assigning the paired discrepancy uniquely to N64.',
                   'The diagnostic does not identify a root cause or qualify either runtime.',
                   'No teacher logits or KLD scores were read or produced. Private requests, responses and logs remain hash-referenced locally.']})
    for path, expected in stats.items():
        st = Path(path).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) != expected:
            raise ValueError('input changed during snapshot')
    if terminal_state() != terminal:
        raise ValueError('terminal unit changed')
    manifest = {'schema': 'glm53-p8.capture-v3-gate-manifest.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'generator': prior.receipt(Path(__file__)), 'receipt_helper': prior.receipt(REPO / 'scripts/snapshot_p8_capture_v2_stop.py'),
        'terminal_unit': terminal, 'inputs': inputs,
        'outputs': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in outputs.items()}}
    destination.mkdir(parents=True)
    for name, data in outputs.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with (destination / 'snapshot.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return {'status': 'v3-failed-gate-preserved', 'output': str(destination), 'files': len(outputs) + 1,
            'comparison_summary': {name: {'first_differing_row': c['differing_rows'][0] if c['differing_rows'] else None,
                                         'maximum_absolute_difference': c['maximum_absolute_difference'],
                                         'unequal_values': c['unequal_values']} for name, c in diagnostic['comparisons'].items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
