"""Exercise observer copies on synthetic data; never load a model or teacher.

CPU mode requires TRITON_INTERPRET=1. CUDA mode additionally checks graph
replay with device-varying positions. Neither mode is a throughput measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime_patch'))

import torch
from p8_index_trace.observer import _copy_bytes, _copy_cache
from p8_index_trace.patches import transform, INDEXER, BACKEND, CAPTURE


def exercise(device):
    if device == 'cpu' and os.environ.get('TRITON_INTERPRET') != '1':
        raise RuntimeError('CPU kernel test requires explicit Triton interpreter')
    if device == 'cuda' and os.environ.get('TRITON_INTERPRET') == '1':
        raise RuntimeError('CUDA test forbids interpreter')
    sources = {
        INDEXER: Path('/opt/infernal-invocation/vllm/vllm/model_executor/layers/sparse_attn_indexer_kpool.py'),
        BACKEND: Path('/opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py'),
        CAPTURE: Path(__file__).resolve().parents[1] / 'runtime_patch/p8_decode_capture/v2_hook.py',
    }
    transformations = {name: transform(name, path.read_bytes())[1] for name, path in sources.items()}
    fused = Path('/opt/infernal-invocation/b12x/b12x/attention/nsa_indexer/fused_indexer.py')
    fused_sha = hashlib.sha256(fused.read_bytes()).hexdigest()
    if fused_sha != '69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af':
        raise RuntimeError('unmodified B12X fused indexer identity differs')
    position = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    source = torch.arange(513, dtype=torch.int64, device=device).to(torch.uint8)
    dest = torch.zeros((9, 513), dtype=torch.uint8, device=device)
    counts = torch.zeros(9, dtype=torch.int32, device=device)

    def copy():
        _copy_bytes[(3,)](source, position, dest, counts, 513, 0, 256)

    copy()
    if device == 'cuda':
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy()
        invoke = graph.replay
    else:
        invoke = copy
    for pos in range(254, 265):
        position.fill_(pos)
        source.fill_(pos % 256)
        invoke()
    assert counts.cpu().tolist() == [1] * 9
    expected = torch.tensor([pos % 256 for pos in range(255, 264)], dtype=torch.uint8)[:, None].expand(9, 513)
    assert torch.equal(dest.cpu(), expected)

    tests = ['copy bytes: dynamic positions, multiple CTAs, inclusive bounds, once-per-row counts']
    for pooled, page, records, width in ((True, 64, 66, 132), (False, 128, 264, 304)):
        raw = torch.arange(4 * page * width, dtype=torch.int64).remainder(251).to(torch.uint8)
        cache = raw.to(device).reshape(4, page, width)
        table = torch.tensor([[2, 0, 3]], dtype=torch.int32, device=device)
        output = torch.zeros((9, records * width), dtype=torch.uint8, device=device)
        count = torch.zeros(9, dtype=torch.int32, device=device)
        errors = torch.zeros(9, dtype=torch.int32, device=device)
        def gather():
            _copy_cache[(records,)](cache, position, lengths, table, output, count, errors,
                0, page, page * width, 4, 3, records, width, pooled, 256 if pooled else 512)
        position.fill_(254)
        gather()
        if device == 'cuda':
            torch.cuda.synchronize()
            graph2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph2):
                gather()
            invoke2 = graph2.replay
        else:
            invoke2 = gather
        expected = torch.zeros((9, records, width), dtype=torch.uint8)
        for pos in range(254, 265):
            nvalid = (pos + 1) // 4 if pooled else pos + 1
            position.fill_(pos)
            lengths.fill_(nvalid)
            invoke2()
            if 255 <= pos <= 263:
                for item in range(min(nvalid, records)):
                    physical, slot = [2, 0, 3][item // page], item % page
                    origin = physical * page * width
                    if pooled:
                        expected[pos - 255, item, :128] = raw[origin + slot * 128:origin + (slot + 1) * 128]
                        start = origin + page * 128 + slot * 4
                        expected[pos - 255, item, 128:] = raw[start:start + 4]
                    else:
                        expected[pos - 255, item] = raw[origin + slot * width:origin + (slot + 1) * width]
        assert torch.equal(output.cpu().reshape(9, records, width), expected)
        assert count.cpu().tolist() == [1] * 9
        assert errors.cpu().tolist() == [0] * 9
        # Invalid mapped pages must be masked, observed as error, not read.
        table.fill_(-1)
        position.fill_(259)
        invoke2()
        assert errors.cpu()[4].item() == min(nvalid, records)
        assert not bool(output.cpu()[4].any())
        tests.append(f'cache pooled={pooled}: permuted physical pages, split/AoS byte layout, invalid rows and addresses')
    return {'schema': 'glm53-p8.index-trace-preflight.v1', 'status': 'passed',
        'device': device, 'cuda_graph_replay_tested': device == 'cuda',
        'cpu_interpreter': device == 'cpu', 'checks': tests,
        'source_transformations': transformations,
        'unmodified_sources': {str(fused): fused_sha},
        'model_loaded': False, 'teacher_logits_opened': False,
        'speed_measurement_valid': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    print(json.dumps(exercise(args.device), indent=2))
