"""Audit a P8 C1 Nsight Systems trace without changing runtime state.

The analysis intentionally treats CUDA kernel intervals as asynchronous device
work.  It never sums ranks to estimate distributed latency and reports both
interval unions and raw kernel-duration sums because categories can overlap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Iterable, Sequence


WORKER_NAME = 'VLLM::Worker_TP'
GENERATION_RANGE = 'execute_context_0(0)_generation_1(1)'
MOE_NAME_FRAGMENT = 'MoEDynamicKernelBackend_object'
TOPK_NAME_FRAGMENT = 'W4A16TopKSumKernel_object'
GLOBAL_PID_MASK = ~((1 << 24) - 1)


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def chunk_replays(rows: Sequence[tuple], replay_count: int) -> list[Sequence[tuple]]:
    if replay_count <= 0 or len(rows) % replay_count:
        raise ValueError('graph-kernel inventory cannot be divided into replays')
    width = len(rows) // replay_count
    if width <= 0:
        raise ValueError('empty graph replay')
    chunks = [rows[index * width:(index + 1) * width]
              for index in range(replay_count)]
    node_inventory = Counter(row[6] for row in chunks[0])
    if any(Counter(row[6] for row in chunk) != node_inventory
           for chunk in chunks[1:]):
        raise ValueError('graph-node inventory differs between replays')
    return chunks


def infer_active_m(topk_grids: set[int], hidden_size: int = 4096,
                   threads: int = 256) -> int:
    if len(topk_grids) != 1:
        raise ValueError('top-k grid differs between graph replays')
    grid_x = next(iter(topk_grids))
    tiles_per_row = (hidden_size + threads - 1) // threads
    if grid_x % tiles_per_row:
        raise ValueError('top-k grid is not an integral active-M witness')
    return grid_x // tiles_per_row


def _median_ms(values: Sequence[int]) -> float:
    return statistics.median(values) / 1e6


def _require_tables(db: sqlite3.Connection) -> None:
    required = {
        'PROCESSES', 'StringIds', 'NVTX_EVENTS',
        'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL',
        'CUDA_GRAPH_NODE_EVENTS', 'TARGET_INFO_CUDA_DEVICE',
    }
    present = {row[0] for row in db.execute(
        "select name from sqlite_master where type='table'")}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f'missing Nsight tables: {missing}')


def _string_id(db: sqlite3.Connection, fragment: str) -> int:
    rows = list(db.execute(
        'select id from StringIds where value like ?', (f'%{fragment}%',)))
    if len(rows) != 1:
        raise ValueError(f'expected one StringIds match for {fragment!r}, got {len(rows)}')
    return int(rows[0][0])


def _kernel_category(demangled: str, short: str,
                     moe_id: int, topk_id: int,
                     demangled_id: int) -> str:
    if demangled_id == moe_id:
        return 'trellismx_fused_moe'
    if demangled_id == topk_id:
        return 'deterministic_topk_sum'
    if 'FillFunctor<' in demangled:
        return 'fill_kernels'
    if short.lower().startswith('nccl'):
        return 'nccl'
    return 'other'


def analyze(sqlite_path: Path, receipt_path: Path, report_path: Path | None = None) -> dict:
    receipt = json.loads(receipt_path.read_text())
    expected_replays = int(receipt['expected_profiled_decode_iterations'])
    if expected_replays <= 0:
        raise ValueError('invalid expected replay count')
    db = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True)
    try:
        _require_tables(db)
        moe_id = _string_id(db, MOE_NAME_FRAGMENT)
        topk_id = _string_id(db, TOPK_NAME_FRAGMENT)
        strings = {int(key): value for key, value in db.execute(
            'select id,value from StringIds')}
        workers = list(db.execute(
            'select globalPid,pid,name from PROCESSES where name=? order by pid',
            (WORKER_NAME,)))
        if len(workers) != 4:
            raise ValueError(f'expected four TP workers, got {len(workers)}')

        rank_rows = []
        replay_intervals_by_rank: list[list[tuple[int, int]]] = []
        common_nodes: Counter[int] | None = None
        all_topk_grids: set[int] = set()
        for global_pid, pid, _ in workers:
            devices = list(db.execute(
                'select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL '
                'where globalPid=?', (global_pid,)))
            if len(devices) != 1:
                raise ValueError(f'worker {pid} has ambiguous CUDA device inventory')
            device_id = int(devices[0][0])
            device = db.execute(
                'select cudaId,uuid,numMultiprocessors from TARGET_INFO_CUDA_DEVICE '
                'where pid=? and gpuId=?', (pid, device_id)).fetchone()
            if device is None:
                raise ValueError(f'worker {pid} device identity is missing')

            generation_count = db.execute(
                "select count(*) from NVTX_EVENTS n left join StringIds s "
                "on n.textId=s.id where (n.globalTid & ?) = ? and "
                "coalesce(n.text,s.value)=?",
                (GLOBAL_PID_MASK, global_pid, GENERATION_RANGE)).fetchone()[0]
            launches = list(db.execute(
                "select r.start,r.end,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r "
                "join StringIds s on s.id=r.nameId where (r.globalTid & ?) = ? "
                "and s.value like 'cudaGraphLaunch%' order by r.start",
                (GLOBAL_PID_MASK, global_pid)))
            if generation_count != expected_replays or len(launches) != expected_replays:
                raise ValueError(
                    f'worker {pid} range/graph-launch inventory differs: '
                    f'{generation_count}/{len(launches)}')

            kernel_rows = list(db.execute(
                'select k.start,k.end,k.demangledName,k.shortName,k.gridX,k.graphId,'
                'k.graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL k '
                'where k.globalPid=? and k.graphNodeId is not null order by k.start',
                (global_pid,)))
            chunks = chunk_replays(kernel_rows, expected_replays)
            node_sequence = Counter(int(row[6]) for row in chunks[0])
            if common_nodes is None:
                common_nodes = node_sequence
            elif node_sequence != common_nodes:
                raise ValueError('graph-node sequence differs between TP workers')

            metrics: dict[str, list[int]] = {
                'graph_span': [], 'all_kernel_sum': [], 'all_kernel_union': [],
                'trellismx_fused_moe_sum': [], 'trellismx_fused_moe_union': [],
                'deterministic_topk_sum_sum': [], 'deterministic_topk_sum_union': [],
                'fill_kernels_sum': [], 'fill_kernels_union': [],
                'nccl_sum': [], 'nccl_union': [], 'other_sum': [], 'other_union': [],
            }
            replay_intervals: list[tuple[int, int]] = []
            graph_ids: set[int] = set()
            for chunk in chunks:
                first = min(row[0] for row in chunk)
                last = max(row[1] for row in chunk)
                replay_intervals.append((first, last))
                metrics['graph_span'].append(last - first)
                metrics['all_kernel_sum'].append(sum(row[1] - row[0] for row in chunk))
                metrics['all_kernel_union'].append(interval_union_ns(
                    (row[0], row[1]) for row in chunk))
                categories: dict[str, list[tuple[int, int]]] = {
                    name: [] for name in (
                        'trellismx_fused_moe', 'deterministic_topk_sum',
                        'fill_kernels', 'nccl', 'other')}
                for start, end, demangled_id, short_id, grid_x, graph_id, _ in chunk:
                    graph_ids.add(int(graph_id))
                    category = _kernel_category(
                        strings[int(demangled_id)], strings[int(short_id)],
                        moe_id, topk_id, int(demangled_id))
                    categories[category].append((start, end))
                    if int(demangled_id) == topk_id:
                        all_topk_grids.add(int(grid_x))
                for category, intervals in categories.items():
                    metrics[f'{category}_sum'].append(
                        sum(end - start for start, end in intervals))
                    metrics[f'{category}_union'].append(interval_union_ns(intervals))
            replay_intervals_by_rank.append(replay_intervals)
            medians = {f'median_{name}_ms': _median_ms(values)
                       for name, values in metrics.items()}
            span = medians['median_graph_span_ms']
            medians['fused_moe_fraction_of_graph_span'] = (
                medians['median_trellismx_fused_moe_sum_ms'] / span)
            medians['nccl_fraction_of_graph_span'] = medians['median_nccl_sum_ms'] / span
            rank_rows.append({
                'worker_pid': int(pid), 'trace_device_id': device_id,
                'cuda_id': int(device[0]), 'device_uuid': device[1],
                'multiprocessors': int(device[2]),
                'generation_ranges': int(generation_count),
                'cuda_graph_launches': len(launches),
                'graph_nodes_per_replay': len(chunks[0]),
                'graph_ids': sorted(graph_ids),
                **medians,
            })

        active_m = infer_active_m(all_topk_grids)
        distributed_spans = []
        for replay_index in range(expected_replays):
            starts = [rows[replay_index][0] for rows in replay_intervals_by_rank]
            ends = [rows[replay_index][1] for rows in replay_intervals_by_rank]
            distributed_spans.append(max(ends) - min(starts))
        critical = sorted(rank_rows,
                          key=lambda row: row['median_trellismx_fused_moe_sum_ms'],
                          reverse=True)[:2]
        result = {
            'schema': 'glm53-p8-nsys-analysis.v1',
            'status': 'pass',
            'claim': ('Diagnostic attribution only; not a throughput, KLD, or '
                      'matched-product qualification.'),
            'trace_sha256': sha256(sqlite_path),
            'report_sha256': sha256(report_path) if report_path else None,
            'client_receipt_sha256': sha256(receipt_path),
            'expected_replays': expected_replays,
            'workers': rank_rows,
            'graph_replay': {
                'proven': True,
                'basis': ('four workers each have the expected cudaGraphLaunch count '
                          'and an identical repeated CUDA graph-node sequence'),
                'nodes_per_replay': sum((common_nodes or Counter()).values()),
                'median_distributed_device_span_ms': _median_ms(distributed_spans),
            },
            'active_m_witness': {
                'topk_grid_x': sorted(all_topk_grids),
                'hidden_size': 4096, 'threads': 256, 'active_m': active_m,
                'formula': 'grid_x = ceil(active_m * hidden_size / threads)',
            },
            'critical_compute_workers': [row['worker_pid'] for row in critical],
            'protected_roles_opened': [],
            'notes': [
                'Kernel-duration category sums are not additive across streams or ranks.',
                'Distributed latency uses latest completion minus earliest start; ranks are never summed.',
                'Fill kernels are identified only by the explicit CUDA FillFunctor demangled name.',
            ],
        }
        return result
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.receipt, args.report)
    if args.output:
        with args.output.open('x') as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
