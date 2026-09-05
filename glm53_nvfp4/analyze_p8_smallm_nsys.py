"""Attribute integrated small-M graph work using CUDA launch correlation IDs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from .analyze_p8_nsys import (
    GENERATION_RANGE, GLOBAL_PID_MASK, WORKER_NAME, infer_active_m,
    interval_union_ns, sha256,
)


def category(name: str) -> str:
    if 'P8SmallMPhase2Kernel_object' in name:
        return 'moe_fc2'
    if 'MoEDynamicKernelBackend_object' in name:
        return 'moe_fc1'
    if 'W4A16TopKSumKernel_object' in name:
        return 'topk'
    lower = name.lower()
    if 'fillfunctor<' in lower:
        return 'fill'
    if 'nccl' in lower:
        return 'nccl'
    if 'nvjet' in lower:
        return 'native_dense_mma'
    if 'gemv' in lower:
        return 'gemv'
    if 'splitkreduce' in lower:
        return 'splitk_reduction'
    if 'mhc' in lower:
        return 'mhc'
    if 'mla' in lower or 'attention' in lower or 'indexer' in lower:
        return 'attention_named'
    if 'gated_delta' in lower:
        return 'gated_delta'
    return 'other'


def group_replays(rows: list[dict], correlation_ids: list[int]) -> list[list[dict]]:
    if len(set(correlation_ids)) != len(correlation_ids):
        raise ValueError('duplicate graph-launch correlation ID')
    groups = {cid: [] for cid in correlation_ids}
    for row in rows:
        if row['correlationId'] not in groups:
            raise ValueError('graph kernel is not assigned to a recorded graph launch')
        groups[row['correlationId']].append(row)
    chunks = [groups[cid] for cid in correlation_ids]
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError('graph launch has no kernel records')
    inventories = [Counter(row['graphNodeId'] for row in chunk) for chunk in chunks]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError('per-worker graph node inventory changes across replay')
    return chunks


def ms_summary(values):
    return {'median_ms': statistics.median(values)/1e6,
            'min_ms': min(values)/1e6, 'max_ms': max(values)/1e6}


def require_c1(grids):
    active_m = infer_active_m(grids)
    if active_m != 1:
        raise ValueError(f'expected active M1, got {active_m}')
    return active_m


def select_complete_prefix(rows, launches, ranges, n):
    """Validate a bounded diagnostic prefix without changing the strict gate."""
    if not n <= len(launches) <= n + 1:
        raise ValueError('diagnostic requires exactly n complete launches and at most one tail')
    ids = [launch['correlationId'] for launch in launches]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate graph-launch correlation ID')
    if any(row['correlationId'] not in ids for row in rows):
        raise ValueError('graph kernel is not assigned to a recorded graph launch')
    prefix_ids = ids[:n]
    prefix = [row for row in rows if row['correlationId'] in prefix_ids]
    chunks = group_replays(prefix, prefix_ids)
    signature_fields = ('graphNodeId', 'demangledName', 'gridX', 'gridY', 'gridZ',
                        'blockX', 'blockY', 'blockZ')
    inventory = Counter(tuple(row[key] for key in signature_fields) for row in chunks[0])
    if any(Counter(tuple(row[key] for key in signature_fields) for row in chunk) != inventory
           for chunk in chunks[1:]):
        raise ValueError('complete-prefix symbol/geometry inventory changes')
    if any(row['end'] <= row['start'] for row in prefix):
        raise ValueError('invalid complete-prefix kernel interval')
    prefix_end = max(row['end'] for row in prefix)
    tail_rows = [row for row in rows if row['correlationId'] not in prefix_ids]
    tail_launches = launches[n:]
    if any(launch['start'] <= prefix_end or launch['end'] < launch['start']
           for launch in tail_launches):
        raise ValueError('excluded launch is not strictly after complete-prefix GPU end')
    if any(row['start'] <= prefix_end or row['end'] <= row['start'] for row in tail_rows):
        raise ValueError('excluded kernel is not strictly after complete-prefix GPU end')
    tail_inventory = Counter(tuple(row[key] for key in signature_fields) for row in tail_rows)
    if tail_launches and not tail_inventory < inventory:
        raise ValueError('trailing group must be a proper subset of the complete inventory')
    matched = set()
    for launch in launches[:n]:
        if launch['end'] < launch['start']:
            raise ValueError('invalid complete-prefix graph launch interval')
        matches = [i for i, event in enumerate(ranges)
                   if event['end'] is not None and event['end'] > event['start']
                   and event['globalTid'] == launch['globalTid']
                   and event['start'] <= launch['start'] <= launch['end'] <= event['end']]
        if len(matches) != 1 or matches[0] in matched:
            raise ValueError('complete launch must have one unique same-thread generation range')
        matched.add(matches[0])
    extra_ranges = [dict(event) for i, event in enumerate(ranges) if i not in matched]
    if len(extra_ranges) > 1:
        raise ValueError('at most one excluded generation range per worker')
    for event in extra_ranges:
        if event['start'] <= prefix_end or (event['end'] is not None and event['end'] <= prefix_end):
            raise ValueError('excluded generation range is not strictly after complete-prefix GPU end')
        event['exclusion_reason'] = ('invalid-or-incomplete-NVTX-range'
                                     if event['end'] is None or event['end'] <= event['start']
                                     else 'generation-range-enclosing-trailing-partial-graph')
        if event['exclusion_reason'] == 'generation-range-enclosing-trailing-partial-graph':
            if len(tail_launches) != 1 or not (
                event['globalTid'] == tail_launches[0]['globalTid']
                and event['start'] <= tail_launches[0]['start'] <= tail_launches[0]['end'] <= event['end']
            ):
                raise ValueError('extra valid generation range must enclose the trailing launch')
    exclusions = {
        'complete_prefix_gpu_end_ns': prefix_end,
        'raw_graph_launch_count': len(launches), 'raw_generation_range_count': len(ranges),
        'raw_graph_kernel_count': len(rows), 'included_graph_kernel_count': len(prefix),
        'excluded_graph_launches': [dict(launch) for launch in tail_launches],
        'excluded_graph_kernel_count': len(tail_rows),
        'excluded_graph_kernels': [dict(row) for row in tail_rows],
        'excluded_generation_ranges': extra_ranges,
    }
    return chunks, exclusions


def analyze(trace: Path, client: Path, report: Path | None = None,
            allow_trailing_capture_artifacts: bool = False) -> dict:
    receipt = json.loads(client.read_text())
    if receipt['status'] != 'client-completed-trace-unverified':
        raise ValueError('client did not complete')
    n = receipt['expected_profiled_decode_iterations']
    if n != 16:
        raise ValueError('expected exactly 16 recorded iterations')
    workers_result, bounds_by_worker = [], []
    cross_worker_inventory = None
    all_topk_grids = set()
    with sqlite3.connect(f'file:{trace}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = {row['id']: row['value'] for row in db.execute('select id,value from StringIds')}
        workers = list(db.execute('select globalPid,pid from PROCESSES where name=? order by pid', (WORKER_NAME,)))
        if len(workers) != 4:
            raise ValueError('trace must contain four TP workers')
        for worker in workers:
            gpid, pid = worker['globalPid'], worker['pid']
            ranges = [dict(row) for row in db.execute(
                'select n.start,n.end,n.globalTid,n.eventType from NVTX_EVENTS n left join StringIds s on n.textId=s.id '
                'where (n.globalTid & ?) = ? and coalesce(n.text,s.value)=?',
                (GLOBAL_PID_MASK, gpid, GENERATION_RANGE))]
            count = len(ranges)
            launches = [dict(row) for row in db.execute(
                "select r.start,r.end,r.correlationId,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
                "on r.nameId=s.id where (r.globalTid & ?) = ? and s.value like 'cudaGraphLaunch%' order by r.start",
                (GLOBAL_PID_MASK, gpid))]
            if not allow_trailing_capture_artifacts and (count != n or len(launches) != n):
                raise ValueError(f'worker {pid} has wrong generation/launch inventory: {count}/{len(launches)}')
            rows = [dict(row) for row in db.execute(
                'select start,end,demangledName,shortName,deviceId,gridX,gridY,gridZ,blockX,blockY,blockZ,'
                'graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL '
                'where globalPid=? and graphNodeId is not null order by start', (gpid,))]
            exclusions = None
            if allow_trailing_capture_artifacts:
                chunks, exclusions = select_complete_prefix(rows, launches, ranges, n)
            else:
                chunks = group_replays(rows, [row['correlationId'] for row in launches])
            device_ids = {row['deviceId'] for row in rows}
            if len(device_ids) != 1:
                raise ValueError('worker uses multiple CUDA devices')
            device = db.execute('select cudaId,uuid,numMultiprocessors from TARGET_INFO_CUDA_DEVICE '
                                'where pid=? and gpuId=?', (pid, next(iter(device_ids)))).fetchone()
            if device is None:
                raise ValueError('worker device identity missing')
            geometry = ('gridX','gridY','gridZ','blockX','blockY','blockZ')
            inventory = Counter((strings[row['demangledName']], *(row[key] for key in geometry)) for row in chunks[0])
            for chunk in chunks[1:]:
                current = Counter((strings[row['demangledName']], *(row[key] for key in geometry)) for row in chunk)
                if current != inventory:
                    raise ValueError('kernel symbol/geometry inventory changes across replay')
            if cross_worker_inventory is None:
                cross_worker_inventory = inventory
            elif inventory != cross_worker_inventory:
                raise ValueError('kernel symbol/geometry inventory differs across ranks')
            duration_sums, duration_unions = defaultdict(list), defaultdict(list)
            name_sums, name_counts = defaultdict(list), {}
            name_geometry = defaultdict(Counter)
            for row in chunks[0]:
                name_geometry[strings[row['demangledName']]][tuple(row[key] for key in geometry)] += 1
            spans, bounds, per_replay = [], [], []
            for chunk in chunks:
                start, end = min(row['start'] for row in chunk), max(row['end'] for row in chunk)
                spans.append(end-start)
                bounds.append((start,end))
                categories, names = defaultdict(list), defaultdict(list)
                for row in chunk:
                    name = strings[row['demangledName']]
                    cat = category(name)
                    interval = (row['start'], row['end'])
                    categories[cat].append(interval)
                    categories['all'].append(interval)
                    if cat in {'moe_fc1','moe_fc2'}:
                        categories['moe_total'].append(interval)
                    names[name].append(interval)
                    if cat == 'topk':
                        all_topk_grids.add(row['gridX'])
                for cat in ('moe_fc1','moe_fc2','topk'):
                    if len(categories[cat]) != 42:
                        raise ValueError(f'worker {pid}: expected 42 {cat} calls, got {len(categories[cat])}')
                for cat, intervals in categories.items():
                    duration_sums[cat].append(sum(b-a for a,b in intervals))
                    duration_unions[cat].append(interval_union_ns(intervals))
                for name, intervals in names.items():
                    name_sums[name].append(sum(b-a for a,b in intervals))
                    name_counts[name] = len(intervals)
                per_replay.append({
                    'span_ms': (end-start)/1e6,
                    'category_sums_ms': {cat: values[-1]/1e6 for cat,values in duration_sums.items()},
                })
            bounds_by_worker.append(bounds)
            summaries = {cat: {'sum': ms_summary(duration_sums[cat]),
                               'union': ms_summary(duration_unions[cat])} for cat in duration_sums}
            top_names = sorted(name_sums, key=lambda name: statistics.median(name_sums[name]), reverse=True)[:30]
            top = [{'name':name, 'category':category(name), 'calls_per_replay':name_counts[name],
                    'geometries':[{'geometry':list(shape),'calls_per_replay':calls}
                                  for shape,calls in sorted(name_geometry[name].items())],
                    **ms_summary(name_sums[name])} for name in top_names]
            workers_result.append({
                'pid':pid, 'cuda_id':device['cudaId'], 'uuid':device['uuid'],
                'sms':device['numMultiprocessors'], 'generation_ranges':count, 'graph_launches':len(launches),
                'nodes_per_replay':len(chunks[0]), 'graph_span':ms_summary(spans),
                'categories':summaries, 'top_kernels':top, 'replays':per_replay,
                'capture_artifact_exclusions': exclusions,
            })
    if len({worker['uuid'] for worker in workers_result}) != 4:
        raise ValueError('expected four distinct physical GPUs')
    if allow_trailing_capture_artifacts:
        global_prefix_end = max(end for bounds in bounds_by_worker for _, end in bounds)
        for worker in workers_result:
            exclusions = worker['capture_artifact_exclusions']
            for key in ('excluded_graph_launches', 'excluded_graph_kernels', 'excluded_generation_ranges'):
                if any(event['start'] <= global_prefix_end or
                       (event['end'] is not None and event['end'] <= global_prefix_end)
                       for event in exclusions[key]):
                    raise ValueError('excluded event precedes distributed complete-prefix GPU end')
    distributed = [max(bounds[i][1] for bounds in bounds_by_worker)-min(bounds[i][0] for bounds in bounds_by_worker) for i in range(n)]
    return {
        'schema':'glm53-p8-smallm-nsys-analysis.v1',
        'status': 'diagnostic-complete-prefix' if allow_trailing_capture_artifacts else 'pass',
        'original_trace_gate_passed': not allow_trailing_capture_artifacts,
        'trace_sha256':sha256(trace), 'report_sha256':sha256(report) if report else None,
        'client_receipt_sha256':sha256(client), 'graph_replay_proven':True,
        'grouping':'per-worker cudaGraphLaunch correlationId and repeated node inventory',
        'graph_span_distributed':ms_summary(distributed), 'active_m':require_c1(all_topk_grids),
        'workers':workers_result, 'protected_roles_opened':[], 'ldlq':False,
        'limits':['One server, 16 correlated replays; no independent replication.',
                  'Profiling overhead prevents substituting spans for unprofiled tokens/s.',
                  'NCCL duration includes rank waiting; it is not a transport-only cost.',
                  'Kernel names identify execution families, not exact model projection owners.',
                  'Category duration sums can overlap; use interval unions and span for elapsed time.'],
    }


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-trailing-capture-artifacts', action='store_true',
                        help='Explicit diagnostic prefix only; never reports the original trace gate as passed.')
    args=parser.parse_args()
    result=analyze(args.trace,args.client,args.report,args.allow_trailing_capture_artifacts)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True)
        handle.write('\n')
    print(json.dumps({'status':result['status'], 'span':result['graph_span_distributed'],
                      'active_m':result['active_m']}))
