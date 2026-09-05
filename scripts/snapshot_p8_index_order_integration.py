#!/usr/bin/env python3
"""Terminal-only CPU preservation of the sealed three-start integration gate.

Never launches a container, reads teacher logits, or copies raw captures/logs.
Existing numerical comparisons are replayed, not replaced by a new experiment.
"""
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
PLAN = REPO / 'experiments/p8-index-order-integration-v1.json'
PLAN_SHA = '5514b3f29db3aad7e88327f1af336013b35f951aa49c96d3846bd0d0a7833e0f'
ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-order-integration-v1')
UNIT = 'glm53-p8-index-order-integration-v1.service'
DEST = REPO / 'evidence/opened/codec-v2/p8-index-order-integration-v1'
WINDOW = 'conditional-fit-0056'
SLOTS = ['repeat-01-n128', 'repeat-02-n128', 'repeat-03-n64']
DEPENDENCIES = '/home/brandonmusic/.local/lib/python3.12/site-packages'
LABEL = 'org.klc.p8-fc1-cold-plan-slot'


def receipt(path):
    path = Path(path)
    if path != path.resolve() or not path.is_file():
        raise ValueError('canonical regular input required')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if signature(path) != (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev):
        raise ValueError('input changed while hashing')
    return {'bytes': before.st_size, 'sha256': digest}


def signature(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ino, value.st_dev


def terminal_state():
    result = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
    state = dict(line.split('=', 1) for line in result.strip().splitlines())
    valid = [dict(ActiveState='inactive', MainPID='0', Result='success', ExecMainStatus='0'),
             dict(ActiveState='failed', MainPID='0', Result='exit-code', ExecMainStatus='1')]
    if state not in valid:
        raise ValueError('unit must be terminal with MainPID zero')
    return state


def sources(plan):
    mapping = plan['source_sha256']
    if len(mapping) != 52:
        raise ValueError('exact52 frozen sources required')
    for name, digest in mapping.items():
        path = REPO / name
        if not path.is_relative_to(REPO) or path != path.resolve() or receipt(path)['sha256'] != digest:
            raise ValueError('sealed source identity differs')


def replay(plan, completed):
    """Pinned-code CPU validation; subprocess cannot inherit Python import path."""
    code = '''
import json,sys,hashlib
from pathlib import Path
repo,planpath=map(Path,sys.argv[1:3]); plan=json.loads(planpath.read_text())
for name,digest in plan['source_sha256'].items():
 p=repo/name
 assert p==p.resolve() and p.is_relative_to(repo)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
sys.path.insert(0,sys.argv[3]);sys.path.insert(0,str(repo))
import numpy as np
from glm53_nvfp4 import p8_index_order_integration as i
root=Path(plan['output']);w=plan['windows'][0]
assert w['id']=='conditional-fit-0056' and i.protocol.sha(Path(w['token_path']))==w['input_sha256']
tokens=np.load(w['token_path'],allow_pickle=False)
proofs={}
for entry in i.ORDER[:int(sys.argv[4])]:
 slot=root/i.slot_name(entry)
 with i.stage_adapter(plan,entry):
  request=json.loads((slot/'requests'/f"{w['id']}.request.json").read_text())
  assert request==i.protocol.completion_request(tokens,w['id'],i.base.model_name(entry['arm']))
  response=json.loads((slot/'requests'/f"{w['id']}.response.json").read_text())
  response_proof=i.protocol.verify_response(response,request)
  array,meta=i.protocol.load_capture(slot/'captures',w['id'],tokens)
  assert array.shape==(2047,154880);del array
  runtime={}
  for phase,completed in [('ready',None),('final',[w])]:
   value=i.base.runtime_audit((slot/f'server-{phase}.private.log').read_text(),entry['arm'],completed)
   assert value==json.loads((slot/f'runtime-{phase}-audit.json').read_text())
   runtime[phase]={'validated':True,'index_order_ranks':value['index_order_ranks'],
                 'kernel_emitted_sha256':value['kernel_emitted_sha256'],'observer_enabled':False}
  proofs[i.slot_name(entry)]={'response':response_proof,'runtime':runtime,
      'capture':{k:meta[k] for k in ('rows_completed','raw_sha256','original_logit_dtype','original_logit_width','sequence_token_ids_sha256')}}
comparisons={}
if (root/'n128-repeat-exact.json').exists():
 value=i.compare_pair(plan,i.ORDER[0],i.ORDER[1])
 assert value==json.loads((root/'n128-repeat-exact.json').read_text())
 comparisons['n128-repeat-exact.json']=value
if (root/'n64-exact.json').exists():
 pairs=[i.compare_pair(plan,e,i.ORDER[2]) for e in i.ORDER[:2]]
 value={'pairs':pairs,'all_exact':all(i.full_rows_exact(p) for p in pairs)}
 assert value==json.loads((root/'n64-exact.json').read_text())
 comparisons['n64-exact.json']=value
print(json.dumps({'completed_captures':proofs,'comparisons':comparisons,
 'teacher_logits_opened':False,'kld_measured':False,'source_count':len(plan['source_sha256'])}))
'''
    return json.loads(subprocess.check_output([sys.executable, '-I', '-B', '-c', code,
        str(REPO), str(PLAN), DEPENDENCIES, str(completed)], text=True))


def validate_execution(value, state):
    if (value.get('schema') != 'glm53-p8.index-order-integration-execution.v1'
            or value.get('plan_sha256') != PLAN_SHA or type(value.get('exit_code')) is not int
            or value['exit_code'] not in (0, 1) or value['exit_code'] != int(state['ExecMainStatus'])
            or value.get('teacher_logits_opened') is not False or value.get('protected_roles_opened') != []
            or value.get('observer_enabled') is not False or value.get('allocation_restart') is not False
            or value.get('speed_measurement_valid') is not False or value.get('full_panel_authorized') is not False
            or not value.get('finished_at')):
        raise ValueError('terminal execution contract differs')
    entries = value['stages']
    expected = [{'index': n, 'arm': 'n128' if n < 3 else 'n64'} for n in range(1, len(entries)+1)]
    if len(entries) > 3 or [{k: e[k] for k in ('index','arm')} for e in entries] != expected:
        raise ValueError('stages not a declared prefix')
    if value.get('restoration_safety') != {'ok': True, 'errors': [], 'containers': []}:
        raise ValueError('owned cleanup is not proven')
    if (set(value['prior']) != {'backend','timer'} or any(type(v) is not bool for v in value['prior'].values())
            or value.get('restoration') != {**value['prior'], 'errors': []}):
        raise ValueError('restoration is not proven')
    if value['exit_code'] == 0 and (len(entries) != 3 or value.get('canary_exact_gate_passed') is not True
            or value.get('final_identity_audit') != {'ok': True}):
        raise ValueError('success lacks full gate evidence')


def timestamp(value):
    parsed=datetime.fromisoformat(value)
    if parsed.utcoffset() is None: raise ValueError('timezone-aware receipt timestamp required')
    return parsed


def chronology(plan,execution,stages):
    begin,end=timestamp(execution['started_at']),timestamp(execution['finished_at'])
    if not timestamp(plan['created_at'])<=begin<=end: raise ValueError('plan/root chronology differs')
    previous=begin
    for value in stages:
        start,stop=timestamp(value['started_at']),timestamp(value['finished_at'])
        if not previous<=start<=stop<=end: raise ValueError('stage chronology differs')
        previous=stop


def validate_gate_chain(execution, stages, comparisons):
    complete = sum(s['exit_code'] == 0 for s in stages)
    if any(s['exit_code'] != 0 for s in stages[:-1]):
        raise ValueError('run continued after stage failure')
    repeat = comparisons.get('n128-repeat-exact.json')
    n64 = comparisons.get('n64-exact.json')
    def pair(value):
        if value.get('rows') != 2047 or value.get('vocabulary') != 154880 or type(value.get('exact')) is not bool:
            raise ValueError('full2047 comparison required')
    if repeat is not None:
        pair(repeat)
        if complete < 2:
            raise ValueError('repeat without two complete captures')
    if len(stages) == 3 and (repeat is None or repeat['exact'] is not True):
        raise ValueError('N64 started before repeat pass')
    if n64 is not None:
        if complete != 3 or len(n64['pairs']) != 2:
            raise ValueError('N64 comparison lacks three captures')
        for item in n64['pairs']: pair(item)
        if n64['all_exact'] is not all(item['exact'] for item in n64['pairs']):
            raise ValueError('N64 aggregate differs')
    if execution['exit_code'] == 0 and (not n64 or not n64['all_exact']):
        raise ValueError('success without exact pair replay')


def snapshot(destination=DEST):
    destination = Path(destination)
    if not destination.is_absolute() or destination != destination.resolve() or destination.exists():
        raise ValueError('fresh canonical output required')
    terminal = terminal_state()
    inputs, stats = {}, {}
    def check(path, expected=None):
        actual = receipt(path)
        if expected is not None and actual != expected:
            raise ValueError('artifact receipt differs')
        inputs[str(path)] = actual; stats[str(path)] = signature(path)
        return actual
    if check(PLAN)['sha256'] != PLAN_SHA or PLAN.with_suffix('.sha256').read_text().split() != [PLAN_SHA, PLAN.name]:
        raise ValueError('plan seal differs')
    check(PLAN.with_suffix('.sha256'))
    plan = json.loads(PLAN.read_text()); sources(plan)
    if plan['output'] != str(ROOT) or [w['id'] for w in plan['windows']] != [WINDOW]:
        raise ValueError('campaign scope differs')
    check(ROOT/'execution.json'); execution=json.loads((ROOT/'execution.json').read_text())
    validate_execution(execution, terminal)
    for path in ROOT.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink in raw inventory')
        if path.is_file(): check(path)
    allowed_root = {'execution.json','hardware-inventory.json','error.private.txt','n128-repeat-exact.json','n64-exact.json'}
    allowed_dirs = set(SLOTS[:len(execution['stages'])])
    if any((p.is_dir() and p.name not in allowed_dirs) or (p.is_file() and p.name not in allowed_root) for p in ROOT.iterdir()):
        raise ValueError('unreceipted stage or root artifact')
    stages=[]; container_ids=[]; comparisons={}
    for entry,name in zip(execution['stages'],SLOTS):
        slot=ROOT/name; path=slot/'execution.json'
        if inputs[str(path)]['sha256'] != entry['execution_sha256']: raise ValueError('root-stage hash chain differs')
        value=json.loads(path.read_text()); stages.append(value)
        if (value.get('schema')!='glm53-p8.forced-m1-v2-stage.v1'
                or value['plan_sha256'] != PLAN_SHA or value['capture_image'] != plan['capture_image']
                or value['arm'] != entry['arm'] or value['stage'] != 'canary'
                or type(value['exit_code']) is not int or value['exit_code'] not in (0,1) or value['cleanup'] != {'ok':True,'errors':[]}
                or value['protected_roles_opened'] != [] or value['allocation_restart'] is not False
                or value['unreadable_artifacts']): raise ValueError('stage cleanup or scope differs')
        actual={str(p.relative_to(slot)):inputs[str(p)] for p in slot.rglob('*') if p.is_file() and p!=path}
        if actual != value['files']: raise ValueError('stage artifact inventory differs')
        if value['exit_code']==0 and 'container_id' not in value:
            raise ValueError('successful capture lacks fresh container identity')
        if 'container_id' in value:
            final=json.loads((slot/'container-final.private.json').read_text())
            expected_name=f'/glm53-p8-index-order-integration-v1-repeat-{entry["index"]:02d}-canary-{entry["arm"]}'
            if (not re.fullmatch('[a-f0-9]{64}',value['container_id'])
                    or final['Id'] != value['container_id'] or final['Image'] != plan['capture_image']
                    or final['Name']!=expected_name
                    or final['Config'].get('Labels',{}).get(LABEL)!=f'{PLAN_SHA}:canary-{entry["arm"]}'
                    or final['State']['Running'] is not False):
                raise ValueError('container stop receipt differs')
            container_ids.append(value['container_id'])
        if value['exit_code']==0 and (len(value['windows'])!=1 or value['windows'][0]['id']!=WINDOW
                or value['windows'][0]['rows']!=2047): raise ValueError('completed stage lacks full window')
        for window in value['windows']:
            if (window['id']!=WINDOW or window['rows']!=2047
                    or window['capture_sha256']!=actual[f'captures/{WINDOW}.capture.json']['sha256']
                    or window['raw_sha256']!=actual[f'captures/{WINDOW}.logits.f32']['sha256']):
                raise ValueError('stage-window hash chain differs')
    if len(container_ids)!=len(set(container_ids)): raise ValueError('containers are not fresh')
    chronology(plan,execution,stages)
    for name,key in [('n128-repeat-exact.json','n128_repeat_sha256'),('n64-exact.json','n64_comparison_sha256')]:
        path=ROOT/name
        if path.exists():
            if inputs[str(path)]['sha256']!=execution.get(key): raise ValueError('root-comparison chain differs')
            comparisons[name]=json.loads(path.read_text())
        elif key in execution: raise ValueError('missing comparison artifact')
    validate_gate_chain(execution,stages,comparisons)
    completed=sum(s['exit_code']==0 for s in stages)
    proof=replay(plan,completed)
    if (proof['comparisons']!=comparisons or list(proof['completed_captures'])!=SLOTS[:completed]
            or proof['source_count']!=52 or proof['teacher_logits_opened'] is not False or proof['kld_measured'] is not False):
        raise ValueError('frozen CPU replay differs')
    # Only constructed typed summaries and successfully replayed numeric gates
    # are portable. Private environment, model output and free-form error text
    # are never copied, including on early failures.
    report={'schema':'glm53-p8.index-order-integration-snapshot.v1',
        'status':'canary-exact-pass' if execution['exit_code']==0 else 'terminal-failure-preserved',
        'plan_sha256':PLAN_SHA,'execution_sha256':inputs[str(ROOT/'execution.json')]['sha256'],
        'exit_code':execution['exit_code'],'completed_captures':completed,'started_stages':len(stages),
        'canary_exact_gate_passed':execution['exit_code']==0,'source_count':52,
        'terminal_unit':terminal,'restoration_verified':True,'owned_cleanup_verified':True,
        'chronology_verified':True,'container_ownership_verified':True,
        'started_at':execution['started_at'],'finished_at':execution['finished_at'],
        'kld_measured':False,'speed_measurement_valid':False,'full_panel_authorized':False,
        'teacher_logits_opened':False,'protected_roles_opened':[], 'allocation_restart':False,
        'stages':[{'slot':name,'exit_code':v['exit_code'],'completed_windows':len(v['windows']),
                   'started_at':v['started_at'],'finished_at':v['finished_at'],
                   'execution_sha256':inputs[str(ROOT/name/'execution.json')]['sha256']} for name,v in zip(SLOTS,stages)],
        'limits':['One opened canary only; not32-window KLD or throughput qualification.',
                  'All raw captures, private logs, requests, responses and environment remain local by hash.']}
    def encode(value): return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    outputs={'report.json':encode(report),'cpu-replay.json':encode(proof),
             'raw-inventory.json':encode({'inputs':inputs,'raw_data_copied':False}),
             'source-identities.json':encode(plan['source_sha256'])}
    for name,value in comparisons.items(): outputs[name]=encode(value)
    sources(plan)
    if any(signature(path)!=expected for path,expected in stats.items()) or terminal_state()!=terminal:
        raise ValueError('inputs or terminal state changed during preservation')
    final_inventory={str(p) for p in ROOT.rglob('*') if p.is_file()}
    if final_inventory!={path for path in inputs if Path(path).is_relative_to(ROOT)}:
        raise ValueError('raw inventory changed during preservation')
    manifest={'schema':'glm53-p8.index-order-integration-manifest.v1','created_at':datetime.now(timezone.utc).isoformat(),
        'generator':receipt(Path(__file__)),'outputs':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in outputs.items()}}
    destination.mkdir(parents=True)
    for name,data in outputs.items():
        with (destination/name).open('xb') as stream: stream.write(data)
    with (destination/'snapshot.json').open('x') as stream: json.dump(manifest,stream,indent=2,sort_keys=True);stream.write('\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEST)
    print(json.dumps(snapshot(parser.parse_args().output)))
