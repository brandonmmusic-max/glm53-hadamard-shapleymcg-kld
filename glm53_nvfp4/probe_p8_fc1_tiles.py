"""Same-process, same-payload FC1 tile ownership closure and graph timing.

Synthetic developer inputs only. This does not establish end-to-end KLD.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from pathlib import Path
import statistics
import sys


CASES = (
    ('small', 1, 0.01), ('unit', 1, 1.0), ('clipping_stress', 1, 8.0),
    ('zero', 1, 0.0), ('tiny', 1, 2.0**-20),
    ('sparse_k_boundaries', 1, 0.0), ('route_permutation', 1, 0.01),
    ('zero_route_weight', 1, 0.01),
    ('fallback_m2', 2, 0.01), ('fallback_m3', 3, 0.01),
)
DEBUG_KEYS = {'packed_a', 'scale_flat', 'intermediate_u32', 'route_output'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def decide(cells, timings, minimum_reduction):
    if not cells or not timings:
        raise ValueError('missing correctness or timing cells')
    if set(timings) != {'128','64','32'} or any(not values for values in timings.values()):
        raise ValueError('expected nonempty N128/N64/N32 timing arms')
    if any(not math.isfinite(value) or value <= 0 for values in timings.values() for value in values):
        raise ValueError('invalid timing sample')
    closed = all(cell['matches_control'] and cell['deterministic']
                 and cell['graph_matches_eager'] and cell['finite'] for cell in cells)
    control = statistics.median(timings['128'])
    gains = {key: 1.0-statistics.median(values)/control
             for key,values in timings.items() if key != '128'}
    if not gains or control <= 0:
        raise ValueError('missing candidate or invalid timing')
    return {'correctness_pass': closed, 'reduction_by_tile': gains,
            'speed_pass_by_tile': {key: closed and gain >= minimum_reduction
                                   for key,gain in gains.items()}}


def run(args):
    import torch
    sys.path.insert(0, str(args.runtime_patch))
    from p8_native_kernel import P8NativeTPMoE

    if args.output.exists():
        raise FileExistsError(args.output)
    if not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id):
        raise ValueError('requires immutable launched image identity')
    source_modules = ('p8_native_kernel','b12x.moe._shared.kernels.dynamic',
                      'b12x.moe._shared.kernels.p8_narrow_fc1',
                      'b12x.moe._shared.kernels.p8_small_m',
                      'b12x.moe._shared.kernels.w4a8_phase1')
    sources = {}
    for name in source_modules:
        path = Path(importlib.import_module(name).__file__)
        sources[name] = {'path':str(path),'sha256':sha(path)}
    if sources['b12x.moe._shared.kernels.p8_small_m']['sha256'] != 'a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f':
        raise ValueError('executed FC2 source differs from measured control')
    if args.tiles != [128,64,32]:
        raise ValueError('frozen order is N128, N64, N32')
    if args.repeats != 5 or args.timing_repeats != 100 or args.timing_rounds != 5:
        raise ValueError('frozen repetition counts differ')
    device = torch.device('cuda')
    rng = torch.Generator(device=device).manual_seed(args.seed)
    runtimes = {
        tile: P8NativeTPMoE(args.sidecar, device=device, tp_rank=args.rank,
                           layer=3, expected_design_sha256=sha(args.design),
                           deterministic_output=True, small_m_scheduler=True,
                           fc1_tile_n=tile, debug_capture=True)
        for tile in args.tiles
    }

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8)
                              .numpy().tobytes()).hexdigest()

    def state(runtime, output):
        if set(runtime.debug_tensors) != DEBUG_KEYS:
            raise ValueError('missing or unexpected intermediate buffers')
        return {'output': tensor_hash(output),
                **{key:tensor_hash(value) for key,value in runtime.debug_tensors.items()}}

    cells, payloads, benchmark_payload = [], [], None
    for name,m,scale in CASES:
        x = (torch.randn(m,4096,device=device,generator=rng)*scale).to(torch.bfloat16)
        # Cover nontrivial expert addresses, including the first and last expert.
        ids = torch.stack([torch.randperm(286,device=device,generator=rng)[:6]+1
                           for _ in range(m)])
        ids = torch.cat([torch.zeros(m,1,device=device,dtype=ids.dtype),ids,
                         torch.full((m,1),287,device=device,dtype=ids.dtype)],1).to(torch.int32)
        weights = torch.softmax(torch.randn(m,8,device=device,generator=rng),-1).float()
        if name == 'sparse_k_boundaries':
            x[:,[0,31,32,1023,1024,4031,4032,4095]] = torch.tensor(
                [1,-1,2,-2,4,-4,8,-8],dtype=torch.bfloat16,device=device)
        elif name == 'route_permutation':
            x,old_weights,old_ids = benchmark_payload
            weights,ids = old_weights.roll(3,1),old_ids.roll(3,1)
        elif name == 'zero_route_weight':
            weights[:,3] = 0
        payloads.append({'case':name,'tokens':m,'scale':scale,
                         'sha256':{key:tensor_hash(value) for key,value in
                                   [('x',x),('ids',ids),('weights',weights)]}})
        if name == 'small':
            benchmark_payload = (x,weights,ids)
        control_state = None
        for tile,runtime in runtimes.items():
            eager = []
            finite = True
            for _ in range(args.repeats):
                output = runtime(x,weights,ids)
                torch.cuda.synchronize()
                finite = finite and bool(torch.isfinite(output).all())
                eager.append(state(runtime,output))
            if tile == 128:
                control_state = eager[0]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    runtime(x,weights,ids)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = runtime(x,weights,ids)
            graph_states = []
            for _ in range(5):
                graph.replay()
                torch.cuda.synchronize()
                graph_states.append(state(runtime,graph_output))
            cell = {'case':name,'tokens':m,'tile_n':tile,
                    'eager_states':eager,'graph_states':graph_states,
                    'matches_control':all(value == control_state for value in eager+graph_states),
                    'deterministic':all(value == eager[0] for value in eager),
                    'graph_matches_eager':all(value == eager[0] for value in graph_states),
                    'finite':finite}
            cells.append(cell)
            print(json.dumps({key:cell[key] for key in ('case','tile_n','matches_control','finite')}),flush=True)
            del graph, graph_output
    closed = all(cell['matches_control'] and cell['deterministic'] and
                 cell['graph_matches_eager'] and cell['finite'] for cell in cells)
    timing_samples, timing_rounds = {str(tile):[] for tile in args.tiles}, []
    # No timing after a correctness failure. Preserve all failure evidence.
    if closed:
        graphs = {}
        outputs = {}
        for tile,runtime in runtimes.items():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    runtime(*benchmark_payload)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[tile] = runtime(*benchmark_payload)
            graphs[tile] = graph
        for round_id in range(args.timing_rounds):
            order = args.tiles[round_id%3:]+args.tiles[:round_id%3]
            for tile in order:
                graph = graphs[tile]
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                values = []
                for _ in range(args.timing_repeats):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    values.append(float(start.elapsed_time(end)))
                timing_samples[str(tile)].extend(values)
                timing_rounds.append({'round':round_id,'tile_n':tile,'samples_ms':values,
                                      'median_ms':statistics.median(values)})
        decision = decide(cells,timing_samples,args.minimum_reduction)
    else:
        decision = {'correctness_pass':False,'speed_pass_by_tile':{},
                    'reason':'timing skipped after numerical closure failure'}
    result = {
        'schema':'glm53-p8-fc1-tiles-probe.v1', 'decision':decision,
        'image_id':args.image_id,'executed_sources':sources,
        'sidecar_sha256':sha(args.sidecar), 'design_sha256':sha(args.design),
        'probe_sha256':sha(__file__), 'rank':args.rank, 'seed':args.seed,
        'device':torch.cuda.get_device_name(), 'payloads':payloads, 'cells':cells,
        'timing_rounds':timing_rounds, 'timing_samples_ms':timing_samples,
        'minimum_reduction':args.minimum_reduction, 'protected_roles_opened':[],
        'ldlq':False, 'evidence_level':'developmental-device-comparison',
        'limits':['Synthetic inputs, one layer and payload rank; not full-model KLD.',
                  'Five timing rounds are subsamples, not independent server starts.',
                  'P8 mxf8f6f4 uses twice the NVFP4 MMA issue count.'],
    }
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True)
        handle.write('\n')
    print(json.dumps(decision),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-patch',type=Path,required=True)
    p.add_argument('--sidecar',type=Path,required=True)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-id',required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--seed',type=int,default=20260905)
    p.add_argument('--tiles',type=int,nargs='+',default=[128,64,32])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--timing-rounds',type=int,default=5)
    p.add_argument('--timing-repeats',type=int,default=100)
    p.add_argument('--minimum-reduction',type=float,default=.35)
    run(p.parse_args())
