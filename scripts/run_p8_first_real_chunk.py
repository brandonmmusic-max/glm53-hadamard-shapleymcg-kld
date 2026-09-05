"""Execute the fixed first coupled chunk with durable launch and failure logs."""
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

def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Requires --execute; invoke under model-stack flock')
    plan_path = ROOT / 'experiments/p8-coupled-first-real-chunk-v1.json'
    plan = json.loads(plan_path.read_text())
    for file, key in [
        ('results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json', 'preparation_sha256'),
        ('glm53_nvfp4/quantize_p8_coupled_scale_layer.py', 'encoder_sha256'),
        ('evidence/preparation/p8-v9/m1-device-closure-v9/result.json', 'm1_closure_sha256'),
        ('evidence/preparation/p8-v9/prefill-device-closure-v9/result.json', 'prefill_closure_sha256'),
    ]:
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == plan[key], file
    for service in ['klc-backend.service', 'klc-model-stack.timer']:
        assert subprocess.run(['systemctl', 'is-active', '--quiet', service]).returncode != 0
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], text=True)
    gpu = [r.split(',') for r in rows.splitlines() if r.split(',')[0].strip() == '0'][0]
    assert int(gpu[1]) < 100 and int(gpu[2]) <= 75, rows
    assert shutil.disk_usage(OUT.parent).free > plan['new_chunk_and_logs_reserve_bytes']
    assert not OUT.exists(), 'preserve existing attempt'
    OUT.mkdir()
    source = '/media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16'
    cmd = [sys.executable, '-u', '-m', 'glm53_nvfp4.quantize_p8_coupled_scale_layer',
        '--source', source, '--source-index', source + '/model.safetensors.index.json',
        '--capture-root', '/media/brandonmusic/klcstore/bmxfp4-glm53/teacher/calibration/main-ep4-full',
        '--roles', '/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-v5.json',
        '--design', str(ROOT / 'results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json'),
        '--exl3-scales', '/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw',
        '--codec-output', str(OUT / 'chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors'),
        '--receipt', str(OUT / 'receipts/chunks/layer-003-experts-000-072.json'),
        '--layer', '3', '--expert-start', '0', '--expert-end', '72', '--samples', '256', '--device', 'cuda:0']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4')
    receipt = dict(command=cmd, start=time.time(), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), gpu_snapshot=rows, env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS']})
    (OUT / 'first-chunk-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with (OUT / 'first-chunk.stdout.log').open('x') as stdout, (OUT / 'first-chunk.stderr.log').open('x') as stderr:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    receipt.update(exit_code=result.returncode, end=time.time())
    (OUT / 'first-chunk-execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
