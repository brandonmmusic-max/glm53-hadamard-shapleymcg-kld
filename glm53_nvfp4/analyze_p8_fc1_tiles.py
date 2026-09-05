"""Verify per-device FC1 closure receipts and the frozen speed decision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from .probe_p8_fc1_tiles import CASES, comparison_state, sha


def analyze(root,plan_path):
    plan=json.loads(plan_path.read_text())
    if sha(plan_path) != plan_path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('plan seal differs')
    execution=json.loads((root/'execution.json').read_text())
    if (execution['exit_code'] != 0 or execution['plan_sha256'] != sha(plan_path)
            or execution.get('execution_status') != 'completed-all-device-comparisons'):
        raise ValueError('all-device execution did not complete successfully')
    if execution['protected_roles_opened'] != []:
        raise ValueError('unexpected protected role opening')
    for service in ('backend','timer'):
        if execution[f'prior_{service}_active'] != execution[f'restored_{service}_active']:
            raise ValueError('service state was not restored')
    records, first_payloads = [], None
    expected={(name,tile) for name,_,_ in CASES for tile in (128,64,32)}
    for gpu in plan['physical_gpu_order']:
        path=root/f'gpu{gpu}/result.json'
        result=json.loads(path.read_text())
        receipt=next(entry for entry in execution['devices'] if entry['gpu']==gpu)
        if sha(path) != receipt['result_sha256']:
            raise ValueError('result checksum differs from execution receipt')
        if result['image_id'] != plan['candidate_image'] or result['sidecar_sha256'] != plan['sidecar_sha256']:
            raise ValueError('candidate or sidecar identity differs')
        if result['probe_sha256'] != plan['source_sha256']['glm53_nvfp4/probe_p8_fc1_tiles.py']:
            raise ValueError('probe source identity differs')
        if first_payloads is None:
            first_payloads=result['payloads']
        elif result['payloads'] != first_payloads:
            raise ValueError('payload differs across devices')
        cells={(cell['case'],cell['tile_n']):cell for cell in result['cells']}
        if len(result['cells']) != len(expected) or set(cells) != expected:
            raise ValueError('cell inventory differs')
        for (name,tile),cell in cells.items():
            states=cell['eager_states']+cell['graph_states']
            control=comparison_state(cells[(name,128)]['eager_states'][0])
            if (len(cell['eager_states']) != 5 or len(cell['graph_states']) != 5
                    or not cell['finite'] or any(comparison_state(state)!=control for state in states)):
                raise ValueError(f'GPU{gpu} {name} N{tile} failed bit-exact closure')
        rounds=result['timing_rounds']
        order=[(r,tile) for r in range(5) for tile in ([128,64,32][r%3:]+[128,64,32][:r%3])]
        if [(entry['round'],entry['tile_n']) for entry in rounds] != order:
            raise ValueError('timing order differs')
        samples={str(tile):[] for tile in (128,64,32)}
        for entry in rounds:
            if len(entry['samples_ms']) != 100:
                raise ValueError('timing sample count differs')
            if any(not math.isfinite(value) or value<=0 for value in entry['samples_ms']):
                raise ValueError('invalid timing sample')
            samples[str(entry['tile_n'])].extend(entry['samples_ms'])
        if samples != result['timing_samples_ms']:
            raise ValueError('aggregate timing samples differ')
        medians={tile:statistics.median(values) for tile,values in samples.items()}
        reductions={tile:1-medians[tile]/medians['128'] for tile in ('64','32')}
        records.append({'physical_gpu':gpu,'result_sha256':sha(path),
                        'medians_ms':medians,'reduction':reductions,
                        'correctness_pass':True})
    eligible=[tile for tile in ('64','32') if all(row['reduction'][tile]>=plan['minimum_reduction'] for row in records)]
    return {'schema':'glm53-p8-fc1-tiles-analysis.v1',
            'decision':'advance_to_integrated_tp4' if eligible else 'stop_before_integrated_tp4',
            'eligible_tiles':eligible,'devices':records,'plan_sha256':sha(plan_path),
            'minimum_reduction':plan['minimum_reduction'],'protected_roles_opened':[],
            'limits':['Synthetic layer3 rank0 payload; not full-model KLD or serving tokens/s.',
                      '500 timings per arm/device are correlated subsamples.',
                      'P8 uses twice NVFP4 MMA issue count.']}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.root,args.plan)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True)
        handle.write('\n')
    print(json.dumps(result))
