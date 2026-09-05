"""Fixed layer20 encoding stage, preserving each chunk and launch receipt."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_p8_layer3_remainder import ROOT, OUT, paths, sha, validate_chunk, replace_arg


def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Requires --execute under model-stack flock')
    plan_path = ROOT / 'experiments/p8-coupled-layer20-encoding-v1.json'
    plan = json.loads(plan_path.read_text())
    pins = [
        ('evidence/preparation/p8-v9-graph-v2/result.json','graph_result_sha256'),
        ('evidence/preparation/p8-layer3-complete/loader-result.json','layer3_loader_result_sha256'),
        ('results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json','preparation_sha256'),
        ('glm53_nvfp4/quantize_p8_coupled_scale_layer.py','encoder_sha256'),
    ]
    for path,key in pins:
        assert sha(ROOT / path) == plan[key], path
    for path,_ in pins[:2]:
        assert json.loads((ROOT / path).read_text())['decision'] == 'pass'
    launch = json.loads((OUT / 'first-chunk-launch.json').read_text())
    previous = None
    for start,end in plan['ranges']:
        prior = validate_chunk(*previous,layer=20) if previous else plan['layer3_loader_result_sha256']
        for service in ['klc-backend.service','klc-model-stack.timer']:
            assert subprocess.run(['systemctl','is-active','--quiet',service]).returncode != 0
        gpu = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
        memory,temp = map(int,gpu.strip().split(','))
        assert memory < 100 and temp <= 75, gpu
        before = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        assert before + 1100000000 <= plan['existing_artifact_bytes'] + plan['new_chunk_and_log_reserve_bytes']
        assert shutil.disk_usage(OUT).free >= 1100000000
        codec,receipt = paths(start,end,layer=20)
        assert not codec.exists() and not receipt.exists(), 'preserve prior attempt'
        command = list(launch['command'])
        for name,value in [('--layer',20),('--expert-start',start),('--expert-end',end),('--codec-output',codec),('--receipt',receipt)]:
            replace_arg(command,name,value)
        replace_arg(command,'--capture-root','/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/fit-capture-l20-l22-v1')
        prefix = OUT / f'layer20-{start:03d}-{end:03d}-capture-relocation-v2'
        record = dict(command=command,start=time.time(),plan_sha256=sha(plan_path),prior_receipt_sha256=prior,gpu_snapshot=gpu,artifact_bytes_before=before)
        with Path(str(prefix)+'.launch.json').open('x') as f:
            json.dump(record,f,indent=2)
        with Path(str(prefix)+'.stdout.log').open('x') as out,Path(str(prefix)+'.stderr.log').open('x') as err:
            result = subprocess.run(command,cwd=ROOT,env=dict(os.environ,**launch['env']),stdout=out,stderr=err)
        record.update(exit_code=result.returncode,end=time.time())
        Path(str(prefix)+'.execution.json').write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
        validate_chunk(start,end,layer=20)
        previous = (start,end)

if __name__ == '__main__':
    main()
