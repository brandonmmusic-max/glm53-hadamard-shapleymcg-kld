"""CPU-only fake receipts: no services, model, GPU, or teacher access."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('snapshot_index_order',Path(__file__).resolve().parents[1]/'scripts/snapshot_p8_index_order_integration.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def terminal(success=True):
    return dict(ActiveState='inactive' if success else 'failed', MainPID='0',
                Result='success' if success else 'exit-code',ExecMainStatus='0' if success else '1')


def execution(success=True,count=3):
    return {'schema':'glm53-p8.index-order-integration-execution.v1','plan_sha256':s.PLAN_SHA,
        'exit_code':0 if success else 1,'teacher_logits_opened':False,'protected_roles_opened':[],
        'observer_enabled':False,'allocation_restart':False,'speed_measurement_valid':False,
        'full_panel_authorized':False,'started_at':'2026-09-05T00:00:00+00:00','finished_at':'2026-09-05T04:00:00+00:00',
        'stages':[{'index':n,'arm':'n128' if n<3 else 'n64','execution_sha256':'a'*64} for n in range(1,count+1)],
        'restoration_safety':{'ok':True,'errors':[],'containers':[]},'prior':{'backend':True,'timer':False},
        'restoration':{'backend':True,'timer':False,'errors':[]},
        'canary_exact_gate_passed':success,'final_identity_audit':{'ok':True}}


def pair(exact=True):
    return {'rows':2047,'vocabulary':154880,'exact':exact}


@pytest.mark.parametrize('bad',[
    {'MainPID':'1'}, {'ActiveState':'active'}, {'Result':'signal'}, {'ExecMainStatus':'2'}])
def test_terminal_rejects_live_or_unknown(monkeypatch,bad):
    value={**terminal(),**bad}
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**k:'\n'.join(f'{k}={v}' for k,v in value.items()))
    with pytest.raises(ValueError,match='terminal'):s.terminal_state()


@pytest.mark.parametrize('success',[True,False])
def test_terminal_accepted(monkeypatch,success):
    value=terminal(success)
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**k:'\n'.join(f'{k}={v}' for k,v in value.items()))
    assert s.terminal_state()==value


@pytest.mark.parametrize('field,value',[
    ('teacher_logits_opened',True),('observer_enabled',True),('allocation_restart',True),
    ('full_panel_authorized',True),('speed_measurement_valid',True),('protected_roles_opened',['confirmation']),
    ('restoration_safety',{'ok':True,'errors':[],'containers':[{'running':False}]}),
    ('restoration',{'backend':False,'timer':False,'errors':[]}),('exit_code',True),
    ('finished_at',''),('final_identity_audit',{'ok':False})])
def test_execution_failclosed(field,value):
    root=execution();root[field]=value
    with pytest.raises(ValueError):s.validate_execution(root,terminal())


def test_execution_failure_accepts_zero_started_stages():
    s.validate_execution(execution(False,0),terminal(False))


@pytest.mark.parametrize('variant',['repeatfail','tailrows','missing','aggregate','continuedfailure'])
def test_gate_failclosed(variant):
    stages=[{'exit_code':0} for _ in range(3)]
    gates={'n128-repeat-exact.json':pair(),'n64-exact.json':{'pairs':[pair(),pair()],'all_exact':True}}
    if variant=='repeatfail':gates['n128-repeat-exact.json']['exact']=False
    if variant=='tailrows':gates['n64-exact.json']['pairs'][0]['rows']=1788
    if variant=='missing':gates.pop('n64-exact.json')
    if variant=='aggregate':gates['n64-exact.json']['pairs'][0]['exact']=False
    if variant=='continuedfailure':stages[0]['exit_code']=1
    with pytest.raises(ValueError):s.validate_gate_chain(execution(),stages,gates)


def test_failed_repeat_is_preserved_not_relaxed():
    s.validate_gate_chain(execution(False,2),[{'exit_code':0},{'exit_code':0}],{'n128-repeat-exact.json':pair(False)})


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    repo=tmp_path/'repo';root=tmp_path/'raw';root.mkdir();repo.mkdir()
    planpath=repo/'experiments/plan.json'
    monkeypatch.setattr(s,'REPO',repo);monkeypatch.setattr(s,'PLAN',planpath);monkeypatch.setattr(s,'ROOT',root)
    mapping={}
    for n in range(52):
        p=repo/f'sources/{n}.py';write(p,{'n':n});mapping[str(p.relative_to(repo))]=s.receipt(p)['sha256']
    plan={'output':str(root),'windows':[{'id':s.WINDOW}],'source_sha256':mapping,'capture_image':'sha256:'+'7'*64,
          'created_at':'2026-09-04T23:00:00+00:00'}
    write(planpath,plan);digest=s.receipt(planpath)['sha256'];monkeypatch.setattr(s,'PLAN_SHA',digest)
    planpath.with_suffix('.sha256').write_text(digest+'  '+planpath.name+'\n')
    rootvalue=execution(False,0);write(root/'execution.json',rootvalue)
    (root/'error.private.txt').write_text('SECRET_ENV=private failure message')
    monkeypatch.setattr(s,'terminal_state',lambda:terminal(False))
    monkeypatch.setattr(s,'replay',lambda plan,count:{'comparisons':{},'completed_captures':{},'source_count':52,'teacher_logits_opened':False,'kld_measured':False})
    return root,rootvalue,tmp_path/'snapshot',plan


def test_snapshot_early_failure_is_safe_and_no_overwrite(campaign):
    root,value,dest,plan=campaign
    report=s.snapshot(dest)
    assert report['status']=='terminal-failure-preserved' and not report['canary_exact_gate_passed']
    assert report['started_stages']==0 and report['completed_captures']==0
    assert set(p.name for p in dest.iterdir())=={'report.json','cpu-replay.json','raw-inventory.json','source-identities.json','snapshot.json'}
    assert 'SECRET_ENV' not in ''.join(p.read_text() for p in dest.iterdir())
    assert str(root/'error.private.txt') in json.loads((dest/'raw-inventory.json').read_text())['inputs']
    manifest=json.loads((dest/'snapshot.json').read_text())
    for name,digest in manifest['outputs'].items():assert s.receipt(dest/name)==digest
    with pytest.raises(ValueError,match='fresh'):s.snapshot(dest)


@pytest.mark.parametrize('kind',['source','seal','unreceipted','symlink','replay'])
def test_snapshot_rejects_tampering(campaign,monkeypatch,kind):
    root,value,dest,plan=campaign
    if kind=='source':(s.REPO/'sources/0.py').write_text('changed')
    elif kind=='seal':s.PLAN.with_suffix('.sha256').write_text('0'*64+' plan.json')
    elif kind=='unreceipted':(root/'repeat-03-n64').mkdir()
    elif kind=='symlink':(root/'private-link').symlink_to(s.PLAN)
    elif kind=='replay':monkeypatch.setattr(s,'replay',lambda *a:{'comparisons':{},'completed_captures':{},'source_count':51,'teacher_logits_opened':False,'kld_measured':False})
    with pytest.raises(ValueError):s.snapshot(dest)
    assert not dest.exists()


def test_isolated_replay_calls_only_python(monkeypatch):
    seen=[]
    def call(args,**kwargs):seen.append(args);return '{}'
    monkeypatch.setattr(s.subprocess,'check_output',call)
    s.replay({},0)
    assert seen[0][1:4]==['-I','-B','-c']
    assert not any('docker'==v or 'systemctl'==v for v in seen[0])
    code=seen[0][4]
    assert 'load_role_inputs' not in code and 'safe_open' not in code
    assert 'chunk_rows' not in code # uses the frozen compare_pair(chunk_rows=8)


def complete_campaign(campaign,monkeypatch,*,success=True):
    root,_,dest,plan=campaign
    value=execution(success,3)
    gates={'n128-repeat-exact.json':pair(),
           'n64-exact.json':{'pairs':[pair(success),pair(success)],'all_exact':success}}
    for name,key in [('n128-repeat-exact.json','n128_repeat_sha256'),('n64-exact.json','n64_comparison_sha256')]:
        write(root/name,gates[name]);value[key]=s.receipt(root/name)['sha256']
    for entry,name in zip(value['stages'],s.SLOTS):
        directory=root/name;directory.mkdir()
        cid=f'{entry["index"]:064x}'
        write(directory/'container-final.private.json',{'Id':cid,'Image':plan['capture_image'],'State':{'Running':False},
            'Name':f'/glm53-p8-index-order-integration-v1-repeat-{entry["index"]:02d}-canary-{entry["arm"]}',
            'Config':{'Labels':{s.LABEL:f'{s.PLAN_SHA}:canary-{entry["arm"]}'}},'SECRET_ENV':'must not copy'})
        capture=f'captures/{s.WINDOW}.capture.json';raw=f'captures/{s.WINDOW}.logits.f32'
        write(directory/capture,{'private_metadata':'never copy verbatim'})
        write(directory/raw,{'tiny_fake_data':'not real logits'})
        files={str(p.relative_to(directory)):s.receipt(p) for p in directory.rglob('*') if p.is_file()}
        stage={'schema':'glm53-p8.forced-m1-v2-stage.v1','plan_sha256':s.PLAN_SHA,'capture_image':plan['capture_image'],'arm':entry['arm'],'stage':'canary',
               'started_at':f'2026-09-05T0{entry["index"]}:00:00+00:00','finished_at':f'2026-09-05T0{entry["index"]}:30:00+00:00',
               'exit_code':0,'cleanup':{'ok':True,'errors':[]},'protected_roles_opened':[],
               'allocation_restart':False,'unreadable_artifacts':[],'container_id':cid,'files':files,
               'windows':[{'id':s.WINDOW,'rows':2047,'capture_sha256':files[capture]['sha256'],'raw_sha256':files[raw]['sha256']}]}
        write(directory/'execution.json',stage);entry['execution_sha256']=s.receipt(directory/'execution.json')['sha256']
    write(root/'execution.json',value)
    monkeypatch.setattr(s,'terminal_state',lambda:terminal(success))
    monkeypatch.setattr(s,'replay',lambda plan,count:{'comparisons':gates,
        'completed_captures':{name:{'rows':2047} for name in s.SLOTS},'source_count':52,
        'teacher_logits_opened':False,'kld_measured':False})
    return root,value,dest,plan


@pytest.mark.parametrize('success',[True,False])
def test_full_gate_outcome_preserved_exactly(campaign,monkeypatch,success):
    root,value,dest,plan=complete_campaign(campaign,monkeypatch,success=success)
    report=s.snapshot(dest)
    assert report['canary_exact_gate_passed'] is success
    assert report['status']==('canary-exact-pass' if success else 'terminal-failure-preserved')
    assert report['completed_captures']==3 and report['started_stages']==3
    assert not report['kld_measured'] and not report['speed_measurement_valid']
    combined=''.join(p.read_text() for p in dest.iterdir())
    assert 'SECRET_ENV' not in combined and 'tiny_fake_data' not in combined and 'private_metadata' not in combined
    assert json.loads((dest/'n64-exact.json').read_text())['all_exact'] is success


@pytest.mark.parametrize('kind',['stagehash','rawbytes','windowhash','livecontainer','duplicatecontainer','gatehash','owner','name','chronology','missingcontainer'])
def test_complete_receipt_chain_rejects_corruption(campaign,monkeypatch,kind):
    root,value,dest,plan=complete_campaign(campaign,monkeypatch)
    slot=root/s.SLOTS[0];path=slot/'execution.json';stage=json.loads(path.read_text())
    if kind=='stagehash':value['stages'][0]['execution_sha256']='0'*64
    elif kind=='rawbytes':(slot/f'captures/{s.WINDOW}.logits.f32').write_text('changed raw bytes')
    elif kind=='windowhash':
        stage['windows'][0]['raw_sha256']='0'*64;write(path,stage)
        value['stages'][0]['execution_sha256']=s.receipt(path)['sha256']
    elif kind in ('livecontainer','duplicatecontainer','owner','name'):
        finalpath=slot/'container-final.private.json';final=json.loads(finalpath.read_text())
        if kind=='livecontainer':final['State']['Running']=True
        elif kind=='owner':final['Config']['Labels'][s.LABEL]='wrong'
        elif kind=='name':final['Name']='/foreign-container'
        else:final['Id']=f'{2:064x}';stage['container_id']=final['Id']
        write(finalpath,final);stage['files'][finalpath.name]=s.receipt(finalpath)
        write(path,stage);value['stages'][0]['execution_sha256']=s.receipt(path)['sha256']
    elif kind=='gatehash':value['n64_comparison_sha256']='0'*64
    elif kind=='chronology':
        stage['finished_at']='2026-09-05T03:30:00+00:00';write(path,stage)
        value['stages'][0]['execution_sha256']=s.receipt(path)['sha256']
    elif kind=='missingcontainer':
        stage.pop('container_id');write(path,stage)
        value['stages'][0]['execution_sha256']=s.receipt(path)['sha256']
    write(root/'execution.json',value)
    with pytest.raises(ValueError):s.snapshot(dest)
    assert not dest.exists()


def test_failed_first_stage_without_container_preserved(campaign):
    root,_,dest,plan=campaign;value=execution(False,1);slot=root/s.SLOTS[0];slot.mkdir()
    stage={'schema':'glm53-p8.forced-m1-v2-stage.v1','plan_sha256':s.PLAN_SHA,'capture_image':plan['capture_image'],'arm':'n128','stage':'canary',
           'started_at':'2026-09-05T01:00:00+00:00','finished_at':'2026-09-05T01:10:00+00:00',
           'exit_code':1,'cleanup':{'ok':True,'errors':[]},'protected_roles_opened':[],
           'allocation_restart':False,'unreadable_artifacts':[],'files':{},'windows':[]}
    write(slot/'execution.json',stage);value['stages'][0]['execution_sha256']=s.receipt(slot/'execution.json')['sha256']
    write(root/'execution.json',value)
    report=s.snapshot(dest)
    assert report['started_stages']==1 and report['completed_captures']==0 and not report['canary_exact_gate_passed']
