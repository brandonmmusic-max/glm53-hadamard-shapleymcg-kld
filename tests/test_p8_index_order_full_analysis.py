"""Independent two-stage CF32 receipt fixtures; no model/GPU/teacher reads.

Real token arrays, request/response validation and file hashing are used. Only
source-lineage validation, log parsing and giant-logit comparison are mocked;
each mock is explicit and independently exercised as a failure boundary.
"""
from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from glm53_nvfp4 import p8_index_order_full_analysis as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value))


@pytest.fixture
def panel(tmp_path,monkeypatch):
    root=tmp_path/'capture';root.mkdir();planpath=tmp_path/'plan.json'
    windows=[]
    for n in range(32):
        path=tmp_path/'tokens'/f'{n:02d}.npy';path.parent.mkdir(exist_ok=True)
        np.save(path,np.arange(2048,dtype=np.int64)+n)
        windows.append({'id':f'conditional-fit-{n:04d}','domain':sorted(a.protocol.DOMAINS)[n//8],
                        'token_path':str(path),'input_sha256':a.protocol.sha(path)})
    plan={'output':str(root),'capture_image':'sha256:'+'a'*64,'windows':windows,
          'created_at':'2026-09-04T23:00:00+00:00'}
    write(planpath,plan);digest=a.protocol.sha(planpath)
    base=datetime(2026,9,5,tzinfo=timezone.utc)
    def iso(seconds):return (base+timedelta(seconds=seconds)).isoformat()
    source_calls=[]
    def verify_sources(value):
        source_calls.append(value)
        if value.get('fixture_source_drift'):raise ValueError('source drift')
    monkeypatch.setattr(a.launcher,'verify_identities',verify_sources)
    runtime_calls=[]
    def runtime(log,arm,completed):
        expected='final' if completed is not None else 'ready'
        if log!=f'{arm}-{expected}' or (completed is not None and completed!=windows):
            raise ValueError('runtime graph/source proof failed')
        runtime_calls.append((arm,expected))
        return {'fixture_proof':f'{arm}-{expected}'}
    monkeypatch.setattr(a.launcher,'runtime_audit',runtime)
    exact={'all_exact':True,'windows':[{'window_id':w['id'],'rows':2047,'vocabulary':154880,'exact':True} for w in windows]}
    comparison_calls=[]
    def compare(value):comparison_calls.append(value);return copy.deepcopy(exact)
    monkeypatch.setattr(a.launcher,'compare_full',compare)
    stages={};top=[]
    for index,arm in enumerate(('n128','n64')):
        name=f'full-{arm}';folder=root/name
        begin,end=60+120*index,120+120*index
        cid=f'{index+1:064x}'
        files={'cooldown.jsonl':json.dumps({'temperatures':[70,71,72,73]})+'\n',
               'thermal.jsonl':json.dumps({'temperatures':[85,86,87,89]})+'\n',
               'server-ready.private.log':f'{arm}-ready','server-final.private.log':f'{arm}-final',
               'runtime-ready-audit.json':{'fixture_proof':f'{arm}-ready'},
               'runtime-final-audit.json':{'fixture_proof':f'{arm}-final'},
               'container-final.private.json':{'Id':cid,'Image':plan['capture_image'],
                   'Name':f'/{a.launcher.PREFIX}-{name}','Created':iso(begin+1),
                   'State':{'Running':False,'StartedAt':iso(begin+2),'FinishedAt':iso(end-1)},
                   'Config':{'Labels':{a.launcher.base.cold.LABEL:f'{digest}:{name}'}}}}
        rows=[]
        for window in windows:
            wid=window['id'];tokens=np.load(window['token_path'],allow_pickle=False)
            request=a.protocol.completion_request(tokens,wid,a.launcher.base.model_name(arm))
            response={'model':request['model'],'choices':[{'token_ids':tokens[1:].tolist(),'finish_reason':'length'}],
                      'usage':{'prompt_tokens':1,'completion_tokens':2047}}
            files.update({f'requests/{wid}.request.json':request,f'requests/{wid}.response.json':response,
                          f'captures/{wid}.capture.json':{'fixture_metadata':wid},
                          f'captures/{wid}.logits.f32':'tiny placeholder; comparator is mocked'})
            rows.append({'id':wid,'domain':window['domain'],'rows':2047,
                         'response_audit':a.protocol.verify_response(response,request)})
        for relative,value in files.items():write(folder/relative,value)
        receipts={name:{'bytes':(folder/name).stat().st_size,'sha256':a.protocol.sha(folder/name)} for name in files}
        for row in rows:
            row['capture_sha256']=receipts[f'captures/{row["id"]}.capture.json']['sha256']
            row['raw_sha256']=receipts[f'captures/{row["id"]}.logits.f32']['sha256']
        stages[name]={'schema':'glm53-p8.forced-m1-v2-stage.v1','stage':'full','arm':arm,
                      'exit_code':0,'cleanup':{'ok':True,'errors':[]},'capture_image':plan['capture_image'],
                      'plan_sha256':digest,'unreadable_artifacts':[],'protected_roles_opened':[],
                      'allocation_restart':False,'container_id':cid,'started_at':iso(begin),'finished_at':iso(end),
                      'windows':rows,'files':receipts}
        top.append({'slot':name,'execution_sha256':''})
    rootvalue={'schema':'glm53-p8.index-order-full-execution.v1','plan_sha256':digest,
               'capture_image':plan['capture_image'],'exit_code':0,'capture_protocol_complete':True,
               'allocation_restart':False,'opened_roles':['conditional-fit'],'protected_roles_opened':[],
               'observer_enabled':False,'speed_measurement_valid':False,'final_identity_audit':{'ok':True},
               'restoration_safety':{'ok':True,'errors':[],'containers':[]},
               'prior':{'backend':True,'timer':False},'restoration':{'backend':True,'timer':False,'errors':[]},
               'stages':top,'started_at':iso(0),'finished_at':iso(300),'full_exact':True}
    def reseal(slot=None):
        if slot:
            write(root/slot/'execution.json',stages[slot])
            next(e for e in top if e['slot']==slot)['execution_sha256']=a.protocol.sha(root/slot/'execution.json')
        write(root/'execution.json',rootvalue)
    def file(slot,name,value):
        path=root/slot/name;write(path,value)
        stages[slot]['files'][name]={'bytes':path.stat().st_size,'sha256':a.protocol.sha(path)}
        reseal(slot)
    def seal_comparison():
        write(root/'full-exact.json',exact);rootvalue['full_exact_sha256']=a.protocol.sha(root/'full-exact.json');reseal()
    for name in stages:reseal(name)
    seal_comparison()
    return dict(plan=plan,path=planpath,root=root,stages=stages,execution=rootvalue,
                reseal=reseal,file=file,exact=exact,seal_comparison=seal_comparison,
                source_calls=source_calls,runtime_calls=runtime_calls,comparison_calls=comparison_calls)


def verify(f):return a.verify_execution(f['plan'],f['path'])


def test_positive_full_two_stage_balanced_panel(panel):
    execution,exact=verify(panel)
    assert execution['capture_protocol_complete'] and exact['all_exact']
    assert len(exact['windows'])==32
    assert panel['source_calls']==[panel['plan']]
    assert panel['runtime_calls']==[('n128','ready'),('n128','final'),('n64','ready'),('n64','final')]
    assert panel['comparison_calls']==[panel['plan']]


def test_nonexact_capture_protocol_can_be_scored_without_calling_it_closure(panel):
    panel['exact']['windows'][7]['exact']=False;panel['exact']['all_exact']=False
    panel['execution']['full_exact']=False;panel['seal_comparison']()
    _,exact=verify(panel)
    assert exact['all_exact'] is False


@pytest.mark.parametrize('field,value',[
    ('opened_roles',['confirmation']),('protected_roles_opened',['selection']),('observer_enabled',True),
    ('allocation_restart',True),('speed_measurement_valid',True),('exit_code',True),
    ('capture_protocol_complete',False),('final_identity_audit',{'ok':False}),
    ('restoration_safety',{'ok':True,'errors':[],'containers':[{'running':False}]}),
    ('restoration',{'backend':False,'timer':False,'errors':[]}),('capture_image','wrong'),
    ('plan_sha256','0'*64),('full_exact',False)])
def test_root_contract_failclosed(panel,field,value):
    panel['execution'][field]=value;panel['reseal']()
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('kind',['missing','duplicate','role','unbalanced'])
def test_exact_role_inventory(panel,kind):
    windows=panel['plan']['windows']
    if kind=='missing':windows.pop()
    elif kind=='duplicate':windows[1]['id']=windows[0]['id']
    elif kind=='role':windows[0]['id']='confirmation-0000'
    else:windows[0]['domain']=windows[8]['domain']
    with pytest.raises(ValueError,match='balanced'):verify(panel)


def test_source_verifier_failure_propagates(panel):
    panel['plan']['fixture_source_drift']=True
    with pytest.raises(ValueError,match='source drift'):verify(panel)
    assert not panel['runtime_calls']


@pytest.mark.parametrize('kind',['missing','reverse','extra','hash'])
def test_stage_order_count_and_chain(panel,kind):
    entries=panel['execution']['stages']
    if kind=='missing':entries.pop()
    elif kind=='reverse':entries.reverse()
    elif kind=='extra':entries.append(copy.deepcopy(entries[0]))
    else:entries[0]['execution_sha256']='0'*64
    panel['reseal']()
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('field,value',[
    ('exit_code',True),('arm','n64'),('stage','canary'),('plan_sha256','0'*64),
    ('cleanup',{'ok':False,'errors':[]}),('unreadable_artifacts',[{'path':'missing'}]),
    ('protected_roles_opened',['final']),('allocation_restart',True),('container_id','short')])
def test_stage_contract(panel,field,value):
    panel['stages']['full-n128'][field]=value;panel['reseal']('full-n128')
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('kind',['mandatory','unreceipted','bytes','sha','symlink'])
def test_file_inventory(panel,kind):
    slot='full-n128';name='thermal.jsonl';path=panel['root']/slot/name
    if kind=='mandatory':panel['stages'][slot]['files'].pop(name);path.unlink()
    elif kind=='unreceipted':write(panel['root']/slot/'extra.private.txt','extra')
    elif kind=='bytes':panel['stages'][slot]['files'][name]['bytes']+=1
    elif kind=='sha':panel['stages'][slot]['files'][name]['sha256']='0'*64
    else:
        outside=panel['root']/'external';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    panel['reseal'](slot)
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('name,rows',[
    ('thermal.jsonl',[[89,90,80,80]]),('thermal.jsonl',[[80,float('nan'),80,80]]),
    ('thermal.jsonl',[[80,80,80]]),('thermal.jsonl',[[False,80,80,80]]),
    ('thermal.jsonl',[]),('cooldown.jsonl',[[75,76,70,70]]),('cooldown.jsonl',[])])
def test_thermal_bounds(panel,name,rows):
    panel['file']('full-n128',name,''.join(json.dumps({'temperatures':v})+'\n' for v in rows))
    with pytest.raises(ValueError,match='thermal'):verify(panel)


@pytest.mark.parametrize('kind',['runtime','runtime_receipt','request','response','row_response','row_count','row_domain','row_order','tokens','capture_hash','raw_hash'])
def test_runtime_request_sequence_and_capture_binding(panel,kind):
    slot='full-n128';wid=panel['plan']['windows'][0]['id'];stage=panel['stages'][slot]
    if kind=='runtime':panel['file'](slot,'server-ready.private.log','wrong backend')
    elif kind=='runtime_receipt':panel['file'](slot,'runtime-final-audit.json',{})
    elif kind in ('request','response'):
        name=f'requests/{wid}.{kind}.json';value=json.loads((panel['root']/slot/name).read_text())
        if kind=='request':value['vllm_xargs']['p8_decode_capture_forced_token_ids'][258]+=1
        else:value['choices'][0]['token_ids'][258]+=1
        panel['file'](slot,name,value)
    elif kind=='row_response':stage['windows'][0]['response_audit']={}
    elif kind=='row_count':stage['windows'][0]['rows']=1788
    elif kind=='row_domain':stage['windows'][0]['domain']='wrong'
    elif kind=='row_order':stage['windows'].reverse()
    elif kind=='tokens':Path(panel['plan']['windows'][0]['token_path']).write_bytes(b'changed')
    elif kind=='capture_hash':stage['windows'][0]['capture_sha256']='0'*64
    elif kind=='raw_hash':stage['windows'][0]['raw_sha256']='0'*64
    panel['reseal'](slot)
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('kind',['name','label','image','running','id','created','stopped','duplicate'])
def test_owned_fresh_containers(panel,kind):
    slot='full-n128';name='container-final.private.json';value=json.loads((panel['root']/slot/name).read_text())
    if kind=='name':value['Name']='/foreign'
    elif kind=='label':value['Config']['Labels'][a.launcher.base.cold.LABEL]='foreign'
    elif kind=='image':value['Image']='foreign'
    elif kind=='running':value['State']['Running']=True
    elif kind=='id':value['Id']='0'*64
    elif kind=='created':value['Created']='2026-09-04T00:00:00+00:00'
    elif kind=='stopped':value['State']['FinishedAt']='2026-09-05T01:00:00+00:00'
    elif kind=='duplicate':
        value['Id']=panel['stages']['full-n64']['container_id'];panel['stages'][slot]['container_id']=value['Id']
    panel['file'](slot,name,value)
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('kind',['root_reverse','overlap','outside','plan_after','naive_root','naive_stage'])
def test_chronology(panel,kind):
    root=panel['execution'];stage=panel['stages']['full-n128']
    if kind=='root_reverse':root['finished_at']=root['started_at']
    elif kind=='overlap':panel['stages']['full-n64']['started_at']=stage['started_at'];panel['reseal']('full-n64')
    elif kind=='outside':stage['finished_at']='2026-09-05T01:00:00+00:00'
    elif kind=='plan_after':panel['plan']['created_at']='2026-09-06T00:00:00+00:00'
    elif kind=='naive_root':root['started_at']=root['started_at'].replace('+00:00','');root['finished_at']=root['finished_at'].replace('+00:00','')
    elif kind=='naive_stage':stage['started_at']=stage['started_at'].replace('+00:00','')
    panel['reseal']('full-n128')
    with pytest.raises(ValueError):verify(panel)


@pytest.mark.parametrize('kind',['hash','replay','missing','rows','vocab','order','duplicate','aggregate','type'])
def test_comparison_identity_and_full_geometry(panel,kind):
    exact=panel['exact']
    if kind=='hash':panel['execution']['full_exact_sha256']='0'*64;panel['reseal']()
    elif kind=='replay':write(panel['root']/'full-exact.json',{'changed':True});panel['execution']['full_exact_sha256']=a.protocol.sha(panel['root']/'full-exact.json');panel['reseal']()
    else:
        if kind=='missing':exact['windows'].pop()
        elif kind=='rows':exact['windows'][0]['rows']=1788
        elif kind=='vocab':exact['windows'][0]['vocabulary']=152576
        elif kind=='order':exact['windows'].reverse()
        elif kind=='duplicate':exact['windows'][0]['window_id']=exact['windows'][1]['window_id']
        elif kind=='aggregate':exact['windows'][0]['exact']=False
        elif kind=='type':exact['windows'][0]['exact']=1
        panel['seal_comparison']()
    with pytest.raises(ValueError):verify(panel)
