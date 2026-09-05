#!/usr/bin/env python3
"""Bounded synthetic GPU gate; importing this module never initializes CUDA.

No model, teacher, token role, endpoint or throughput measurement is involved.
The host must authenticate the supplied immutable image and serialize GPU access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

IMAGE = 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9'
SOURCE = Path('/opt/infernal-invocation/b12x/b12x/attention/nsa_indexer/fused_indexer.py')
SOURCE_SHA = '69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af'
MODULE = 'b12x.attention.nsa_indexer.fused_indexer'
TOPK, HEADS, CHANNELS, PAGE = 512, 32, 128, 64
THRESHOLD = 1024
LENGTHS = (0, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025)
TRANSITIONS = (512, 513) * 5 + (513, 512) * 5
REPEATS = 5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def synthetic_inputs(seed, capacity=1088):
    """All32 heads dot one-hot Q/K to 1; score=32*(logical_index+1) exactly."""
    if capacity < max(LENGTHS) or capacity % PAGE:
        raise ValueError('page-aligned capacity must cover all declared cases')
    pages = capacity // PAGE
    table = np.random.default_rng(seed).permutation(pages).astype(np.int32)
    if np.array_equal(table, np.arange(pages)):
        table = np.roll(table, 1)
    packed = np.zeros((pages, PAGE * (CHANNELS + 4)), dtype=np.uint8)
    keys = np.ndarray((pages, PAGE, CHANNELS), dtype=np.uint8, buffer=packed,
                      strides=(packed.strides[0], CHANNELS, 1))
    scales = np.ndarray((pages, PAGE), dtype='<f4', buffer=packed,
                        offset=PAGE * CHANNELS, strides=(packed.strides[0], 4))
    keys[:, :, 0] = 0x38  # E4M3FN encoding of exactly 1.0.
    for logical_page, physical_page in enumerate(table):
        scales[physical_page] = np.arange(logical_page * PAGE + 1, (logical_page + 1) * PAGE + 1, dtype=np.float32)
    query = np.zeros((1, HEADS, CHANNELS), dtype=np.uint8)
    query[:, :, 0] = 0x38
    return {'packed': packed, 'query': query, 'weights': np.ones((1, HEADS), dtype=np.float32),
            'table': table[None, :]}


def assert_result(indices, values, length, *, ordered_short, valid_ids=None):
    """Exact set plus paired score bits; legacy long output order is unspecified."""
    if (indices.shape != (1, TOPK) or values.shape != (1, TOPK)
            or indices.dtype != np.int32 or values.dtype != np.float32):
        raise AssertionError('output shape or dtype differs')
    available = np.arange(length, dtype=np.int32) if valid_ids is None else np.asarray(valid_ids, dtype=np.int32)
    if available.ndim != 1 or np.any(available < 0) or np.any(available >= length) or np.unique(available).size != available.size:
        raise AssertionError('invalid explicit reference IDs')
    expected_ids = np.sort(available)[-TOPK:]
    count = len(expected_ids)
    ids, scores = indices[0, :count], values[0, :count]
    if not np.array_equal(np.sort(ids), expected_ids):
        raise AssertionError('selected unique logical index set differs')
    expected_scores = (HEADS * (ids.astype(np.int64) + 1)).astype(np.float32)
    if not np.array_equal(scores.view(np.uint32), expected_scores.view(np.uint32)):
        raise AssertionError('paired FP32 score bits differ from exact analytic reference')
    if not np.all(indices[0, count:] == -1) or not np.all(values[0, count:].view(np.uint32) == 0xff800000):
        raise AssertionError('invalid suffix must be index-1 / negative infinity')
    if ordered_short and length <= TOPK and not np.array_equal(ids, expected_ids):
        raise AssertionError('candidate short rows are not logical ascending')
    order = np.argsort(ids)
    return {'length': length, 'valid_count': count,
            'pair_sha256': digest(ids[order].tobytes() + scores[order].tobytes()),
            'ordered_output_sha256': digest(indices.tobytes() + values.tobytes())}


def assert_counters(state):
    # 768,769 are fused total/cleanup; 770,771 arrival/output. Histograms are scratch.
    if state.dtype != np.int32 or state.shape != (772,) or np.any(state[768:772] != 0):
        raise AssertionError('live merge counters did not reset to zero')


def load_modules():
    raw = SOURCE.read_bytes()
    if digest(raw) != SOURCE_SHA:
        raise ValueError('immutable original kernel source differs')
    if os.environ.get('GLM53_P8_INDEX_ORDER') != 'logical-short-v1':
        raise ValueError('explicit candidate environment required')
    hook = importlib.import_module('p8_index_order')
    # sitecustomize may already have installed the exact finder.
    if MODULE not in sys.modules:
        installed = any(type(f).__module__ == 'p8_index_order' for f in sys.meta_path)
        if not installed:
            hook.install()
    candidate = importlib.import_module(MODULE)
    alias = 'b12x.attention.nsa_indexer._p8_original_fused_indexer'
    if alias in sys.modules:
        raise ValueError('original alias already loaded')
    spec = importlib.util.spec_from_file_location(alias, SOURCE)
    original = importlib.util.module_from_spec(spec)
    sys.modules[alias] = original
    # Original source remains inspectable at its real pinned path; no inherited future flags.
    exec(compile(raw, str(SOURCE), 'exec', dont_inherit=True), original.__dict__)
    transformations = getattr(hook, 'TRANSFORMATIONS', {})
    if set(transformations) != {MODULE}:
        raise ValueError('exactly one transformed module required')
    receipt = transformations[MODULE]
    if not isinstance(receipt, dict) or receipt.get('original_sha256') != SOURCE_SHA:
        raise ValueError('candidate emitted-source receipt missing')
    if receipt.get('emitted_sha256') == SOURCE_SHA:
        raise ValueError('candidate source was not transformed')
    if receipt.get('cache_suffix') != '_p8logicalshortv1':
        raise ValueError('candidate distinct compile-cache identity missing')
    if original.run_fused_paged_indexer.__globals__ is candidate.run_fused_paged_indexer.__globals__:
        raise ValueError('original/candidate function globals are aliased')
    return original, candidate, receipt


def run_device(record, seed):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one host-authorized visible GPU required')
    original, candidate, transformation = load_modules()
    record['source_transformations'] = transformation
    hook = sys.modules['p8_index_order']
    hook_root = Path(hook.__file__).parent
    record['source_sha256'].update({
        'runtime_patch/p8_index_order/' + name: digest((hook_root / name).read_bytes())
        for name in ('__init__.py', 'patches.py')})
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0) or props.multi_processor_count < 4:
        raise ValueError('SM120 with at least4 SMs required')
    record['gpu'] = {'name': props.name, 'uuid': str(getattr(props, 'uuid', 'unavailable')),
        'compute_capability': [props.major, props.minor], 'multiprocessors': props.multi_processor_count,
        'total_memory': props.total_memory, 'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda}
    inputs = synthetic_inputs(seed, max(1088, props.multi_processor_count * PAGE))
    record['input_sha256'] = {k: digest(v.tobytes()) for k, v in inputs.items()}
    packed = torch.from_numpy(inputs['packed']).cuda()
    # As-strided views mirror page-wide key payload followed by page-wide FP32 scales.
    keys = packed.as_strided((packed.shape[0], PAGE, CHANNELS), (8448, 128, 1))
    scales = packed.view(torch.float32).as_strided((packed.shape[0], PAGE), (2112, 1), storage_offset=2048)
    query = torch.from_numpy(inputs['query']).cuda()
    weights = torch.from_numpy(inputs['weights']).cuda()
    table = torch.from_numpy(inputs['table']).cuda()
    seq = torch.zeros(1, dtype=torch.int32, device='cuda')
    oi = torch.full((1, TOPK), -987, dtype=torch.int32, device='cuda')
    ov = torch.full((1, TOPK), float('nan'), dtype=torch.float32, device='cuda')
    # Actual production-style capacity determines CTA count; include full-SM grid explicitly.
    serving_ctas = min(table.shape[1], props.multi_processor_count)
    policies = [('forced-boundary', 4, THRESHOLD, LENGTHS),
                ('serving-default-resolver', serving_ctas,
                 candidate._resolve_default_merge_threshold(ctas_per_group=serving_ctas, num_heads=HEADS, topk=TOPK),
                 (65, 512, 513))]
    reference_pairs = {}
    determinism = hashlib.sha256()
    for policy_name, ctas, threshold, lengths in policies:
        # State and output allocations are shared across both arms and ALL live lengths.
        pack_v = torch.empty(ctas * TOPK, dtype=torch.float32, device='cuda')
        pack_i = torch.empty(ctas * TOPK, dtype=torch.int32, device='cuda')
        state = torch.zeros(772, dtype=torch.int32, device='cuda')
        common = dict(q_bytes=query, weights=weights, k_quant_bytes=keys, k_scales=scales,
            real_page_table=table, seqlens=seq, num_heads=HEADS, topk=TOPK,
            out_indices=oi, out_values=ov, ctas_per_group=ctas, merge_threshold=threshold,
            pack_values=pack_v, pack_indices=pack_i, merge_state=state,
            merge_state_preinitialized=True, output_physical_slots=False)
        for arm, module in [('original', original), ('candidate', candidate)]:
            call = module.run_fused_paged_indexer

            def check(length, mode, repeat, *, scenario='valid', valid_ids=None):
                torch.cuda.synchronize()
                cpu_i, cpu_v = oi.cpu().numpy(), ov.cpu().numpy()
                result = assert_result(cpu_i, cpu_v, length,
                    ordered_short=arm == 'candidate' and valid_ids is None, valid_ids=valid_ids)
                if arm == 'candidate' and valid_ids is None and length <= TOPK and mode in ('eager', 'graph'):
                    determinism.update(json.dumps([policy_name, mode, length, repeat]).encode())
                    determinism.update(cpu_i.tobytes())
                    determinism.update(cpu_v.tobytes())
                assert_counters(state.cpu().numpy())
                key = (policy_name, scenario, length)
                if key in reference_pairs and reference_pairs[key] != result['pair_sha256']:
                    raise AssertionError('original/candidate semantic map differs')
                reference_pairs[key] = result['pair_sha256']
                row = {'arm': arm, 'policy': policy_name, 'ctas_per_group': ctas,
                       'merge_threshold': threshold, 'mode': mode, 'repeat': repeat,
                       'scenario': scenario, 'counter_zero': True, **result}
                return row

            # Compile on a side stream before capture, with explicit stream dependencies.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                seq.fill_(512)
                call(**common)
            torch.cuda.current_stream().wait_stream(stream)
            check(512, 'compile-warmup', 0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(**common)
            for mode in ('eager', 'graph'):
                for length in lengths:
                    for repeat in range(REPEATS):
                        seq.fill_(length)
                        if mode == 'eager':
                            call(**common)
                        else:
                            graph.replay()
                        record['cases'].append(check(length, mode, repeat))
            for repeat, length in enumerate(TRANSITIONS):
                seq.fill_(length)
                graph.replay()
                record['state_transitions'].append(check(length, 'same-graph-512-513', repeat))
            if policy_name == 'forced-boundary':
                for missing_page in (0, 1):
                    table[0, missing_page] = -1
                    seq.fill_(65)
                    valid_ids = np.array([i for i in range(65) if i // PAGE != missing_page], dtype=np.int32)
                    for mode in ('eager', 'graph'):
                        for repeat in range(REPEATS):
                            if mode == 'eager':
                                call(**common)
                            else:
                                graph.replay()
                            record['cases'].append(check(65, mode, repeat,
                                scenario=f'negative-page-{missing_page}', valid_ids=valid_ids))
                    table[0, missing_page] = int(inputs['table'][0, missing_page])
    record['semantic_signature'] = canonical_digest([
        {'policy': key[0], 'scenario': key[1], 'length': key[2], 'pair_sha256': value}
        for key, value in sorted(reference_pairs.items())])
    record['output_digest'] = canonical_digest({'cases': record['cases'], 'state_transitions': record['state_transitions']})
    record['determinism_sha256'] = determinism.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--seed', type=int, default=20260905)
    args = parser.parse_args(argv)
    if args.output != args.output.resolve() or args.output.exists() or not args.output.parent.is_dir():
        raise ValueError('fresh canonical result path with existing parent required')
    record = {'schema': 'glm53-p8.index-order-device.v1', 'status': 'failed',
        'image_id': args.image_id, 'seed': args.seed, 'cases': [], 'state_transitions': [],
        'source_transformations': {}, 'script_sha256': digest(Path(__file__).read_bytes()),
        'source_sha256': {'scripts/preflight_p8_index_order_device.py': digest(Path(__file__).read_bytes())},
        'synthetic_only': True, 'teacher_logits_opened': False, 'model_loaded': False,
        'prefix_gate_points': list(LENGTHS), 'negative_page_cases': ['first-page=-1', 'second-page=-1'],
        'dynamic_length_transitions': list(TRANSITIONS), 'eager_repeats': REPEATS, 'graph_repeats': REPEATS,
        'kld_measured': False, 'speed_measurement_valid': False, 'qualification_pass': False,
        'limits': ['GPU0-only synthetic gate, not full-model/all-rank numerical closure.',
                   'Long legacy rows compare exact selected pairs; legacy output order is unspecified.',
                   'Forced merge threshold1024 exercises legacy cooperative and serial arms; not a serving policy change.',
                   'Only -1 invalid-page sentinels are executed. Upper-range invalid pages remain a caller-precondition violation before the merge guard and are not GPU-tested.',
                   'semantic_signature is order-independent for legacy rows; output_digest may differ across fresh processes.']}
    try:
        if args.image_id != IMAGE or args.seed != 20260905:
            raise ValueError('undeclared image or seed')
        run_device(record, args.seed)
        record['status'] = 'passed'
    except BaseException as error:
        record['failure'] = {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': record['status'], 'output': str(args.output),
                      'semantic_signature': record.get('semantic_signature')}))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
