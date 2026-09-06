"""CPU-only fixed layer20 pack plus source-exact postwrite verification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_p8_layer3_remainder import ROOT, OUT, paths, sha, validate_chunk


def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Requires --execute after all four chunks are complete')
    plan_path = ROOT / 'experiments/p8-coupled-layer20-pack-v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(ROOT / 'glm53_nvfp4/build_p8_coupled_scale_tp4_sidecars.py') == plan['packer_sha256']
    assert sha(ROOT / 'glm53_nvfp4/verify_p8_coupled_scale_tp4_sidecars.py') == plan['verifier_sha256']
    assert plan['layer'] == 20
    assert plan['campaign_aggregate_bound_bytes'] <= plan['user_ceiling_bytes'] == 30000000000
    assert plan['campaign_aggregate_bound_bytes'] >= plan['layer_artifact_bound_bytes'] + plan['existing_external_bound_bytes']
    source_receipts = [validate_chunk(a,b,layer=20) for a,b in plan['source_ranges']]
    before = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert before + 4000000000 <= plan['layer_artifact_bound_bytes']
    assert shutil.disk_usage(OUT).free >= 4000000000
    target = OUT / 'sidecars/layer-020'
    assert not target.exists(), 'preserve existing outputs'
    pack_receipt = OUT / 'receipts/layer-020-sidecars.json'
    verify_receipt = OUT / 'receipts/layer-020-postwrite.json'
    assert not pack_receipt.exists() and not verify_receipt.exists()
    chunk_args = [item for a,b in plan['source_ranges'] for item in ['--chunk',str(paths(a,b,layer=20)[0])]]
    pack = [sys.executable,'-u','-m','glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars',*chunk_args,'--output-dir',str(target),'--receipt',str(pack_receipt),'--layer','20','--world-size','4']
    sidecar_args = [item for r in range(4) for item in ['--sidecar',str(target / f'p8-layer-020-tp4-rank-{r}.safetensors')]]
    verify = [sys.executable,'-u','-m','glm53_nvfp4.verify_p8_coupled_scale_tp4_sidecars',*chunk_args,*sidecar_args,'--packer-receipt',str(pack_receipt),'--receipt',str(verify_receipt),'--layer','20']
    for name,command in [('pack',pack),('postwrite',verify)]:
        record = dict(command=command,start=time.time(),plan_sha256=sha(plan_path),prior_chunk_receipts=source_receipts,layer_bytes_before=before)
        prefix = OUT / f'layer20-{name}'
        with Path(str(prefix)+'.launch.json').open('x') as f:
            json.dump(record,f,indent=2)
        env = dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
        with Path(str(prefix)+'.stdout.log').open('x') as out,Path(str(prefix)+'.stderr.log').open('x') as err:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
        record.update(exit_code=result.returncode,end=time.time())
        Path(str(prefix)+'.execution.json').write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    closed = json.loads(verify_receipt.read_text())
    assert closed['status'] == 'pass' and len(closed['ranks']) == 4
    assert all(rank['source_exact'] for rank in closed['ranks'])
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) <= plan['layer_artifact_bound_bytes']
    print(json.dumps({'postwrite':'pass','retirement_authorized':False,'receipt':str(verify_receipt)}))

if __name__ == '__main__':
    main()
