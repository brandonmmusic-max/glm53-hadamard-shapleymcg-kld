"""Serialized, sealed-plan launcher for the synthetic FC1 ownership device gate."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

from .p8_smallm_profile import command, active, temperatures, now, save, sha
from .probe_p8_fc1_tiles import CASES

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1')
DESIGN = Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-all42-runtime/experiments/p8-kld-shapley-native6-v2.json')
SIDECAR = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1/sidecars/p8-layer-003-tp4-rank-0.safetensors')


def argv(image, uuid, out, name):
    return [
        'docker','create','--name',name,'--gpus',f'device={uuid}',
        '--network','none','--ipc','private','--shm-size','1g',
        '--entrypoint','/opt/venv/bin/python',
        '-e','PYTHONPATH=/work:/runtime-patch','-e','OMP_NUM_THREADS=2',
        '-e','GLM53_P8_NATIVE=','-e','GLM53_P4_NATIVE=',
        '-v',f'{REPO}:/work:ro','-v',f'{REPO}/runtime_patch:/runtime-patch:ro',
        '-v',f'{SIDECAR}:/inputs/sidecar.safetensors:ro',
        '-v',f'{DESIGN}:/inputs/design.json:ro','-v',f'{out}:/out:rw',
        image,'-m','glm53_nvfp4.probe_p8_fc1_tiles',
        '--runtime-patch','/runtime-patch','--sidecar','/inputs/sidecar.safetensors',
        '--design','/inputs/design.json','--output','/out/result.json',
        '--image-id',image,
    ]


def run(plan_path, out):
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != plan_path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('plan seal mismatch')
    if plan['physical_gpu_order'] != [0,1,2,3] or plan['minimum_reduction'] != .35:
        raise ValueError('device order or gate differs')
    fixed = {'tile_order':[128,64,32], 'layer':3, 'payload_rank':0,
             'seed':20260905,'cases':[name for name,_,_ in CASES],
             'timeout_per_gpu_seconds':2400,'thermal_start_max_c':75,'thermal_abort_c':90}
    if any(plan.get(key) != value for key,value in fixed.items()):
        raise ValueError('plan settings differ from the fixed probe and runner')
    if out.parent != ROOT or out.exists() or out != out.resolve():
        raise ValueError('requires fresh absolute output directly under campaign root')
    if command(['git','-C',str(REPO),'status','--porcelain']).stdout.strip():
        raise ValueError('source checkout must be clean')
    for relative,expected in plan['source_sha256'].items():
        if sha(REPO/relative) != expected:
            raise ValueError(f'source hash differs: {relative}')
    if sha(SIDECAR) != plan['sidecar_sha256'] or sha(DESIGN) != plan['design_sha256']:
        raise ValueError('input identities differ')
    image = plan['candidate_image']
    if not re.fullmatch(r'sha256:[a-f0-9]{64}',image):
        raise ValueError('requires immutable image ID')
    if command(['docker','image','inspect',image,'--format','{{.Id}}']).stdout.strip() != image:
        raise ValueError('image differs')
    if shutil.disk_usage(ROOT).free < 5*2**30 or max(temperatures()) > 75:
        raise ValueError('disk or startup thermal preflight failed')
    devices = command(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader']).stdout
    uuid_by_index = {int(line.split(',')[0]):line.split(',')[1].strip()
                     for line in devices.splitlines()}
    if uuid_by_index != {int(key):value for key,value in plan['gpu_uuid_by_index'].items()}:
        raise ValueError('GPU inventory changed')
    lock = open('/run/lock/klc/model-stack.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(mode=0o700)
    prior_backend,prior_timer = active('klc-backend.service',True),active('klc-model-stack.timer')
    record = {'schema':'glm53-p8-fc1-tiles-execution.v1','started_at':now(),
              'plan_sha256':sha(plan_path),'image':image,
              'source_commit':command(['git','-C',str(REPO),'rev-parse','HEAD']).stdout.strip(),
              'prior_backend_active':prior_backend,'prior_timer_active':prior_timer,
              'protected_roles_opened':[],'ldlq':False,'exit_code':1,'devices':[]}
    cid = None
    cleanup_ok = True
    try:
        if prior_timer:
            command(['sudo','-n','systemctl','stop','klc-model-stack.timer'])
        if prior_backend:
            command(['systemctl','--user','stop','klc-backend.service'])
        if command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip():
            raise RuntimeError('another GPU compute process is active')
        save(out/'nvidia-before.xml',command(['nvidia-smi','-q','-x']).stdout)
        for gpu in plan['physical_gpu_order']:
            cell_out = out/f'gpu{gpu}'
            cell_out.mkdir()
            name = f'glm53-p8-fc1tiles-{out.name}-gpu{gpu}'
            launch = argv(image,uuid_by_index[gpu],cell_out,name)
            save(cell_out/'launch.json',{'argv':launch,'physical_gpu':gpu,'uuid':uuid_by_index[gpu]})
            cid = command(launch).stdout.strip()
            if not re.fullmatch('[a-f0-9]{64}',cid):
                raise RuntimeError('invalid created container ID')
            save(cell_out/'container-id.txt',cid+'\n')
            command(['docker','start',cid])
            deadline = time.monotonic()+2400
            with (cell_out/'thermal.jsonl').open('x') as thermal:
                while True:
                    values = temperatures()
                    thermal.write(json.dumps({'at':now(),'temperatures':values})+'\n')
                    thermal.flush()
                    if max(values) >= 90:
                        raise RuntimeError(f'thermal abort: {max(values)} C')
                    state = json.loads(command(['docker','inspect','-f','{{json .State}}',cid]).stdout)
                    if not state['Running']:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('device probe exceeded frozen 2400 second limit')
                    time.sleep(2)
            logs = command(['docker','logs',cid],check=False)
            save(cell_out/'probe.log',logs.stdout+logs.stderr)
            save(cell_out/'container-state.json',state)
            command(['docker','rm',cid])
            cid = None
            if state['ExitCode'] != 0:
                raise RuntimeError(f'GPU{gpu} probe exited {state["ExitCode"]}')
            result = json.loads((cell_out/'result.json').read_text())
            expected_cells = {(name,tile) for name,_,_ in CASES for tile in (128,64,32)}
            observed_cells = [(cell['case'],cell['tile_n']) for cell in result['cells']]
            if (result['image_id'] != image or result['sidecar_sha256'] != plan['sidecar_sha256']
                    or result['design_sha256'] != plan['design_sha256']
                    or result['probe_sha256'] != plan['source_sha256']['glm53_nvfp4/probe_p8_fc1_tiles.py']
                    or len(observed_cells) != len(expected_cells) or set(observed_cells) != expected_cells):
                raise RuntimeError('probe output identity or cell inventory differs')
            record['devices'].append({'gpu':gpu,'result_sha256':sha(cell_out/'result.json'),
                                      'decision':result['decision']})
            if not result['decision']['correctness_pass']:
                record['stop_reason'] = f'GPU{gpu} correctness failure; later devices not run'
                record['execution_status'] = 'completed-correctness-failure'
                break
        else:
            record['execution_status'] = 'completed-all-device-comparisons'
        record['exit_code'] = 0
    except BaseException as error:
        record['error'] = {'type':type(error).__name__,'message':str(error)}
    finally:
        if cid:
            try:
                command(['docker','stop','--time','10',cid],check=False,timeout=30)
                logs = command(['docker','logs',cid],check=False)
                save(out/'failed-probe.log',logs.stdout+logs.stderr)
                state = json.loads(command(['docker','inspect','-f','{{json .State}}',cid]).stdout)
                save(out/'failed-container-state.json',state)
                cleanup_ok = not state['Running']
                if cleanup_ok:
                    command(['docker','rm',cid])
            except BaseException as error:
                cleanup_ok = False
                record['cleanup_error'] = str(error)
        restoration_errors = []
        if cleanup_ok:
            for needed,restore_cmd in (
                (prior_backend,['systemctl','--user','start','klc-backend.service']),
                (prior_timer,['sudo','-n','systemctl','start','klc-model-stack.timer']),
            ):
                if needed:
                    try:
                        command(restore_cmd)
                    except BaseException as error:
                        restoration_errors.append({'command':restore_cmd,'error':str(error)})
        for key,unit,user in (
            ('backend','klc-backend.service',True),('timer','klc-model-stack.timer',False),
        ):
            try:
                record[f'restored_{key}_active'] = active(unit,user)
            except BaseException as error:
                record[f'restored_{key}_active'] = None
                restoration_errors.append({'unit':unit,'error':str(error)})
        record['restoration_errors'] = restoration_errors
        if not cleanup_ok or record['restored_backend_active'] != prior_backend or record['restored_timer_active'] != prior_timer:
            record['exit_code'] = 1
        record['finished_at'] = now()
        try:
            save(out/'execution.json',record)
        finally:
            lock.close()
    print(json.dumps(record),flush=True)
    if record['exit_code']:
        raise RuntimeError('device gate execution failed; see preserved receipt')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.plan,args.output)
