import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

import numpy as np
import pytest

from glm53_nvfp4 import p8_index_order_integration as integration

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capture_base_tests', REPO / 'tests/test_p8_decode_launcher.py')
base_tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(base_tests)


def markers():
    return [{'schema': 'glm53-p8.index-order-serving-receipt.v1', 'tp_rank': rank,
             'mode': 'logical-short-v1', 'tp_world_size': 4,
             'global_rank': rank, 'pid': 100 + rank, 'local_rank_env': str(rank), 'rank_env': str(rank),
             'module': 'b12x.attention.nsa_indexer.fused_indexer', 'original_sha256': integration.ORIGINAL,
             'emitted_sha256': integration.EMITTED, 'inspect_source_sha256': integration.EMITTED,
             'cache_suffix': '_p8logicalshortv1', 'num_heads': 32, 'topk': 512,
             'output_physical_slots': False} for rank in range(4)]


def log(values):
    return '\n'.join('(Worker pid=1) GLM53_P8_INDEX_ORDER_RECEIPT ' + json.dumps(value) for value in values)


def test_four_successful_exact_worker_markers():
    result = integration.order_runtime_audit(log(markers()))
    assert result['index_order_ranks'] == list(range(4)) and result['observer_enabled'] is False


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'source', 'physical', 'heads', 'rank', 'pid', 'observer'])
def test_receipt_failclosed(kind):
    values = markers()
    if kind == 'missing': values.pop()
    elif kind == 'duplicate': values.append(values[0])
    elif kind == 'source': values[0]['inspect_source_sha256'] = '0' * 64
    elif kind == 'physical': values[0]['output_physical_slots'] = True
    elif kind == 'heads': values[0]['num_heads'] = 64
    elif kind == 'rank': values[0]['global_rank'] = 1
    elif kind == 'pid': values[0]['pid'] = values[1]['pid']
    text = log(values) + ('\nGLM53_P8_INDEX_TRACE_COMPLETE' if kind == 'observer' else '')
    with pytest.raises(ValueError): integration.order_runtime_audit(text)


def test_adapter_preserves_lifecycle_and_has_no_observer(tmp_path):
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities', 'clone_argv', 'runtime_audit')
    before = {key: getattr(integration.base, key) for key in names}
    for entry in integration.ORDER:
        with pytest.raises(RuntimeError, match='sentinel'):
            with integration.stage_adapter({}, entry):
                assert integration.base.PREFIX == integration.prefix(entry)
                args = integration.base.clone_argv(base_tests.pilot_recipe(), integration.IMAGE,
                    {'stage': 'canary', 'arm': entry['arm']}, tmp_path, [{'id': integration.WINDOW}], 'seal')
                env = dict(args[i+1].split('=', 1) for i,v in enumerate(args) if v == '--env')
                assert env['GLM53_P8_INDEX_ORDER'] == 'logical-short-v1'
                assert env['GLM53_P8_INDEX_ORDER_RECEIPT'] == '1'
                assert env['GLM53_P8_INDEX_TRACE'] == ''
                assert env['GLM53_P8_FC1_TILE_N'] == ('128' if entry['index'] < 3 else '64')
                assert env['GLM53_P8_FUSED_SCRATCH'] == ('' if entry['index'] < 3 else '1')
                assert env['GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS'] == '2047'
                assert not any('/p8-index-traces' in value for value in args)
                with pytest.raises(ValueError, match='full panel'):
                    integration.base.stage_windows({'windows': []}, 'full')
                raise RuntimeError('sentinel')
    assert {key: getattr(integration.base, key) for key in names} == before


@pytest.mark.parametrize('collision', ['env', 'mount'])
def test_adapter_rejects_observer_recipe(tmp_path, collision):
    recipe = base_tests.pilot_recipe()
    if collision == 'env': recipe['Config']['Env'].append('GLM53_P8_INDEX_TRACE=1')
    else: recipe['HostConfig']['Binds'].append('/tmp/traces:/p8-index-traces:rw')
    with integration.stage_adapter({}, integration.ORDER[0]):
        with pytest.raises(ValueError):
            integration.base.clone_argv(recipe, integration.IMAGE, {'stage': 'canary','arm': 'n128'},tmp_path,[], 'seal')


def test_comparison_reads_one_token_file_and_full2047_rows(tmp_path, monkeypatch):
    token = tmp_path / 'tokens.npy'; np.save(token, np.arange(2048))
    reads, captures = [], []
    original = np.load
    monkeypatch.setattr(integration.np, 'load', lambda p, **kw: reads.append(p) or original(p, **kw))
    def capture(path, window, tokens):
        assert len(tokens) == 2048 and window == integration.WINDOW
        captures.append(path)
        return np.zeros((2047,2)), {'original_logit_dtype': 'bf16', 'original_logit_width': 154880,
                                   'sequence_token_ids_sha256': 'same'}
    monkeypatch.setattr(integration.protocol, 'load_capture', capture)
    def exact(a,b,*,chunk_rows):
        assert chunk_rows == 8 and a.shape[0] == b.shape[0] == 2047
        return {'exact': True, 'rows': 2047, 'vocabulary': 154880}
    monkeypatch.setattr(integration.protocol, 'exact_logits', exact)
    monkeypatch.setattr(integration.protocol, 'load_role_inputs', lambda *a,**kw: pytest.fail('teacher open'))
    result = integration.compare_pair({'output': str(tmp_path), 'windows': [{'token_path': str(token)}]}, *integration.ORDER[:2])
    assert reads == [str(token)] and len(captures) == 2 and result['rows'] == 2047
    assert result['teacher_logits_opened'] is False and result['kld_measured'] is False


def test_plan_and_verification_never_use_teacher_loader(tmp_path, monkeypatch):
    repo, raw = tmp_path / 'repo', tmp_path / 'raw'; repo.mkdir();raw.mkdir()
    source = repo/'source.py';source.write_text('pinned')
    preflight=tmp_path/'preflight.json';preflight.write_text('{}')
    monkeypatch.setattr(integration,'REPO',repo);monkeypatch.setattr(integration,'ROOT',raw)
    monkeypatch.setattr(integration,'source_files',lambda: {'source.py'})
    monkeypatch.setattr(integration,'import_receipt',lambda p: None)
    old={'image_receipt':'/pinned.json','image_receipt_sha256':'a'*64}
    window={'id':integration.WINDOW}
    monkeypatch.setattr(integration,'historical_inputs',lambda:(window,{},old))
    monkeypatch.setattr(integration.base,'approved_inputs',lambda *a,**kw:pytest.fail('teacher open'))
    monkeypatch.setattr(integration.protocol,'load_role_inputs',lambda *a,**kw:pytest.fail('teacher open'))
    path=tmp_path/'plan.json';out=raw/'index-order-integration-v1'
    plan=integration.make_plan(path,out,preflight)
    assert integration.authenticate(path)==plan
    assert plan['raw_capture_bytes']==3*2047*154880*4 and len(plan['order'])==3
    for key,bad in [('order',list(reversed(integration.ORDER))),('raw_capture_bytes',1),('observer_enabled',True),
                    ('teacher_logits_opened',True),('rows_per_window',256)]:
        with pytest.raises(ValueError):integration.verify_identities({**plan,key:bad})
    source.write_text('drift')
    with pytest.raises(ValueError,match='source drift'):integration.verify_identities(plan)


@pytest.mark.parametrize('failure', ['none','stage1','repeat','n64','duplicate_container','restore'])
def test_three_fresh_stages_gate_before_n64_and_restore_under_interrupts(tmp_path,monkeypatch,failure):
    path=tmp_path/'plan.json';path.write_text('{}');out=tmp_path/'out'
    plan={'output':str(out)}
    monkeypatch.setattr(integration,'authenticate',lambda p:plan)
    monkeypatch.setattr(integration,'prelaunch',lambda p:{})
    monkeypatch.setattr(integration,'verify_identities',lambda p:None)
    monkeypatch.setattr(integration.pilot,'command',lambda argv:subprocess.CompletedProcess(argv,0,'',''))
    monkeypatch.setattr(integration.pilot,'active',lambda *a:False)
    monkeypatch.setattr(integration.cold,'inventory',lambda:[])
    monkeypatch.setattr(integration,'open',lambda path,mode:(tmp_path/'lock').open(mode),raising=False)
    monkeypatch.setattr(integration.fcntl,'flock',lambda *a:None)
    monkeypatch.setattr(integration,'restoration_safety',lambda digest:{'ok':True,'errors':[],'containers':[]})
    stages,comparisons,restores=[],[],[]
    def capture(plan,entry,recipe,root,digest,hardware):
        stages.append(entry['arm']);root.mkdir();(root/'execution.json').write_text('{}')
        return {'exit_code':int(failure=='stage1'), 'container_id':('same' if failure=='duplicate_container' else str(len(stages)))}
    monkeypatch.setattr(integration.base,'capture_stage',capture)
    def compare(plan,left,right):
        comparisons.append((left['index'],right['index']))
        return {'exact':not((failure=='repeat' and right['index']==2) or (failure=='n64' and right['index']==3)),'rows':2047,'vocabulary':154880}
    monkeypatch.setattr(integration,'compare_pair',compare)
    def restore(prior,safety):
        for sig in integration.base.INTERRUPT_SIGNALS:
            assert signal.getsignal(sig)==signal.SIG_IGN;os.kill(os.getpid(),sig)
        restores.append(True)
        if failure=='restore':raise RuntimeError('daemon failure')
        return {**prior,'errors':[]}
    monkeypatch.setattr(integration.cold,'restore_if_safe',restore)
    before={sig:signal.getsignal(sig) for sig in integration.base.INTERRUPT_SIGNALS}
    if failure=='none':assert integration.run(path)['canary_exact_gate_passed'] is True
    else:
        with pytest.raises(RuntimeError,match='preserved without retry'):integration.run(path)
    assert {sig:signal.getsignal(sig) for sig in before}==before
    assert stages==['n128']*(1 if failure=='stage1' else 2)+([] if failure in ('stage1','repeat','duplicate_container') else ['n64'])
    record=json.loads((out/'execution.json').read_text())
    assert restores==[True] and record['exit_code']==int(failure!='none')
    if failure=='repeat':assert comparisons==[(1,2)] and not (out/'n64-exact.json').exists()
    if failure=='none':assert comparisons==[(1,2),(1,3),(2,3)]


@pytest.mark.parametrize('kind',['live','foreign','absent','daemon'])
def test_restoration_requires_safe_owned_inventory(monkeypatch,kind):
    cid='a'*64;entry=integration.ORDER[0]
    item={'Id':cid,'Name':'/'+integration.prefix(entry)+'-canary-n128','Image':integration.IMAGE,
          'Config':{'Labels':{integration.cold.LABEL:'seal:canary-n128'}},'State':{'Running':kind=='live'}}
    if kind=='foreign':item['Image']='foreign'
    def command(args):
        if kind=='daemon':raise RuntimeError('daemon unavailable')
        return subprocess.CompletedProcess(args,0,('' if kind=='absent' else cid) if args[1]=='ps' else json.dumps([item]),'')
    monkeypatch.setattr(integration.pilot,'command',command)
    assert integration.restoration_safety('seal')['ok'] is (kind=='absent')


@pytest.mark.parametrize('key,value',[('rows',256),('vocabulary',1000),('exact',1),('exact','true')])
def test_no_tail_only_or_partial_vocabulary_exact_gate(key,value):
    record={'rows':2047,'vocabulary':154880,'exact':True,key:value}
    with pytest.raises(ValueError,match='every2047rows'):integration.full_rows_exact(record)


def test_import_preflight_requires_all_seven_current_sources(tmp_path,monkeypatch):
    names={'runtime_patch/sitecustomize.py','runtime_patch/p8_index_order/__init__.py',
           'runtime_patch/p8_index_order/patches.py','runtime_patch/p8_index_order_receipt.py',
           'scripts/preflight_p8_index_order_serving_import.py','scripts/preflight_p8_index_order_import.py',
           'scripts/preflight_p8_index_order_device.py'}
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('pinned')
    monkeypatch.setattr(integration,'REPO',tmp_path)
    value={'schema':'glm53-p8.index-order-serving-import.v1','status':'passed','image_id':integration.IMAGE,
           'gpu_used':False,'model_loaded':False,'teacher_logits_opened':False,
           'module_source_exact':True,'cute_class_source_isolated':True,'compile_cache_identity_isolated':True,
           'receipt_wrapper_installed':True,'wrapped_source_globals_exact':True,'worker_breakable_graph_mode':True,
           'serving_rank_apis_available':True,'marker_emitted':False,'actual_distributed_call_tested':False,
           'source_sha256':{name:integration.pilot.sha(tmp_path/name) for name in names},
           'source_transformation':{'original_sha256':integration.ORIGINAL,'emitted_sha256':integration.EMITTED,
                                   'inspect_source_sha256':integration.EMITTED,'cache_suffix':'_p8logicalshortv1'}}
    path=tmp_path/'preflight.json';path.write_text(json.dumps(value));integration.import_receipt(path)
    for key,bad in [('source_sha256',{}),('source_transformation',{}),('gpu_used',True),
                    ('wrapped_source_globals_exact',False),('worker_breakable_graph_mode',False),
                    ('marker_emitted',True),('schema','legacy')]:
        path.write_text(json.dumps({**value,key:bad}))
        with pytest.raises(ValueError):integration.import_receipt(path)
