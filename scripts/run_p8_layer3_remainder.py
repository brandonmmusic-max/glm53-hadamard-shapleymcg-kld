"""Finish the fixed layer3 ranges; no retry, packing, cleanup or service changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1')
DESIGN = '4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695'
ENCODER = '0c1900e3425b310bf4f669d02d6b75d52769aa04ad2a1544b4c8904072f90bc1'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def paths(start, end):
    return (OUT / f'chunks/layer-003/p8-coupled-layer-003-experts-{start:03d}-{end:03d}.safetensors',
            OUT / f'receipts/chunks/layer-003-experts-{start:03d}-{end:03d}.json')

def validate_chunk(start, end):
    codec, receipt = paths(start, end)
    item = json.loads(receipt.read_text())
    assert item['schema'] == 'glm53-hessian-trellis-p8-coupled-scale-layer-chunk-receipt.v1'
    assert item['layer'] == 3 and item['expert_range'] == [start, end]
    assert item['design_sha256'] == DESIGN and item['protected_roles_opened'] == []
    assert item['calibration']['role'] == 'fit' and item['calibration']['samples'] == 256
    assert item['algorithm']['ldlq'] is False
    assert item['algorithm']['intermediate_draws'] == [0] * (end-start)
    declared = item['outputs']['codec']
    assert Path(declared['path']).resolve() == codec.resolve()
    assert codec.stat().st_size == declared['bytes'] and sha(codec) == declared['sha256']
    return sha(receipt)

def replace_arg(command, name, value):
    command[command.index(name)+1] = str(value)

def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Requires --execute under model-stack flock; first chunk must be terminal')
    assert json.loads((OUT / 'first-chunk-execution.json').read_text())['exit_code'] == 0
    plan_path = ROOT / 'experiments/p8-coupled-layer3-remainder-v1.json'
    plan = json.loads(plan_path.read_text())
    launch = json.loads((OUT / 'first-chunk-launch.json').read_text())
    assert sha(ROOT / 'glm53_nvfp4/quantize_p8_coupled_scale_layer.py') == ENCODER
    assert sha(ROOT / 'results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json') == DESIGN
    previous = (0,72)
    for start,end in plan['ranges']:
        prior_sha = validate_chunk(*previous)
        for service in ['klc-backend.service','klc-model-stack.timer']:
            assert subprocess.run(['systemctl','is-active','--quiet',service]).returncode != 0
        gpu = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
        memory,temp = map(int,gpu.strip().split(','))
        assert memory < 100 and temp <= 75, gpu
        used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        assert used + 1100000000 <= plan['layer3_chunk_and_logs_bound_bytes']
        assert shutil.disk_usage(OUT).free > 1100000000
        codec,receipt = paths(start,end)
        assert not codec.exists() and not receipt.exists(), 'preserve existing attempt'
        command = list(launch['command'])
        for name,value in [('--expert-start',start),('--expert-end',end),('--codec-output',codec),('--receipt',receipt)]:
            replace_arg(command,name,value)
        prefix = OUT / f'layer3-{start:03d}-{end:03d}'
        record = dict(command=command,start=time.time(),prior_receipt_sha256=prior_sha,plan_sha256=sha(plan_path),gpu_snapshot=gpu,layer_bytes_before=used)
        with Path(str(prefix)+'.launch.json').open('x') as f:
            json.dump(record,f,indent=2)
        env = dict(os.environ, **launch['env'])
        with Path(str(prefix)+'.stdout.log').open('x') as out, Path(str(prefix)+'.stderr.log').open('x') as err:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
        record.update(end=time.time(),exit_code=result.returncode)
        Path(str(prefix)+'.execution.json').write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
        validate_chunk(start,end)
        previous = (start,end)

if __name__ == '__main__':
    main()
