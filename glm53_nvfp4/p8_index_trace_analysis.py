"""Predeclared, bitwise localization analysis; never a teacher-KLD estimator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DTYPES = {'torch.int32': '<i4', 'torch.int64': '<i8', 'torch.uint8': 'u1',
          'torch.float32': '<f4', 'torch.bfloat16': '<u2', 'torch.float8_e4m3fn': 'u1'}
LAYERS = list(range(3, 44, 4))
POSITIONS = list(range(255, 264))


def decode(raw, spec):
    dtype = DTYPES.get(spec['torch_dtype'])
    if dtype is None:
        raise ValueError('undeclared trace tensor dtype')
    shape = tuple(spec['logical_shape'])
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[0] != 9:
        raise ValueError('trace data must be nine byte rows')
    if raw.shape[1] != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError('logical tensor geometry does not match raw bytes')
    # BF16 is returned as raw uint16, not mislabeled as numerical FP16.
    return raw.copy().view(dtype).reshape((9, *shape))


def verify_row(row, position, cache_layout):
    pools = row['pool_ids'].reshape(-1)
    logical = row['logical_tokens'].reshape(-1)
    physical = row['physical_slots'].reshape(-1)
    plen = int(row['pool_lengths'].reshape(-1)[0])
    length = int(row['attention_lengths'].reshape(-1)[0])
    if plen != (position + 1) // 4 or length != position + 1 or pools.size != 512:
        raise ValueError('trace length/position mismatch')
    if np.any(pools < -1) or not np.array_equal(np.sort(pools[pools >= 0]), np.arange(plen)):
        raise ValueError('pool selection missing, duplicated or outside complete causal pools')
    expected = np.where(pools[:, None] >= 0, pools[:, None] * 4 + np.arange(4), -1).reshape(-1)
    if logical.size < 2051 or not np.array_equal(logical[:2048], expected):
        raise ValueError('logical token expansion differs from observed pool order')
    tail = np.full(3, -1, dtype=np.int64)
    tail[:length % 4] = np.arange(plen * 4, length)
    if not np.array_equal(logical[2048:2051], tail) or np.any(logical[2051:] != -1):
        raise ValueError('tail or padding differs from the pinned expansion')
    if not np.array_equal(np.sort(logical[logical >= 0]), np.arange(length)):
        raise ValueError('logical selected set is not the complete causal prefix')
    if physical.size != 2048:
        raise ValueError('pinned backend must consume 2048 physical selection columns')
    # The indexer appends three tail columns, but this pinned backend slices
    # only the first 2048. Record that distinction; do not "fix" the runtime.
    consumed = logical[:physical.size]
    page = cache_layout['page_size']
    table = row['attention_table'].reshape(-1)
    good = consumed >= 0
    if np.any(consumed[good] // page >= table.size):
        raise ValueError('captured page table does not cover selected tokens')
    mapped = np.full(physical.shape, -1, dtype=np.int64)
    mapped[good] = table[consumed[good] // page] * page + consumed[good] % page
    if not np.array_equal(physical, mapped):
        raise ValueError('observed physical slots differ from logical page mapping')
    if np.any(row['pool_cache'][plen:] != 0) or np.any(row['attention_cache'][length:] != 0):
        raise ValueError('invalid cache rows were not masked')


def bitwise_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                                    np.ascontiguousarray(b).view(np.uint8))))


def compare_row(a, b):
    same = {label: bitwise_equal(a[label], b[label]) for label in a}
    pa, pb = a['pool_ids'].reshape(-1), b['pool_ids'].reshape(-1)
    la, lb = a['logical_tokens'].reshape(-1), b['logical_tokens'].reshape(-1)
    pool_sets_equal = bool(np.array_equal(np.sort(pa[pa >= 0]), np.sort(pb[pb >= 0])))
    token_sets_equal = bool(np.array_equal(np.sort(la[la >= 0]), np.sort(lb[lb >= 0])))
    ca, cb = la[:a['physical_slots'].size], lb[:b['physical_slots'].size]
    consumed_sets_equal = bool(np.array_equal(np.sort(ca[ca >= 0]), np.sort(cb[cb >= 0])))
    scorer_inputs_equal = all(same[label] for label in ('index_query', 'index_weights', 'pool_cache', 'pool_lengths'))
    attention_content_equal = all(same[label] for label in ('attention_query', 'attention_cache', 'attention_lengths'))
    order_differs = not same['pool_ids']
    return {'bitwise_equal': same, 'pool_sets_equal': pool_sets_equal,
        'logical_token_sets_equal': token_sets_equal,
        'consumed_token_sets_equal': consumed_sets_equal,
        'expanded_tail_tokens_a': la[2048:][la[2048:] >= 0].tolist(),
        'expanded_tail_tokens_b': lb[2048:][lb[2048:] >= 0].tolist(),
        'tail_columns_consumed': False,
        'scorer_inputs_bitwise_equal': scorer_inputs_equal,
        'attention_content_bitwise_equal': attention_content_equal,
        'index_order_divergence_with_same_inputs': scorer_inputs_equal and pool_sets_equal and order_differs,
        'order_linked_attention_divergence': (scorer_inputs_equal and pool_sets_equal
            and attention_content_equal and consumed_sets_equal
            and order_differs and not same['attention_output']),
        'pool_order_a': pa[pa >= 0].tolist(), 'pool_order_b': pb[pb >= 0].tolist()}


def compare(root, *, transformations):
    from .p8_index_trace_launcher import trace_manifests, repeat_name
    directories = [Path(root) / repeat_name(index) / 'index-traces' for index in (1, 2)]
    for directory in directories:
        trace_manifests(directory, transformations)
    rows = []
    for rank in range(4):
        metas = [json.loads((d / f'rank-{rank}.json').read_text()) for d in directories]
        # Keep only one rank-pair resident at a time; compressed files are local.
        with np.load(directories[0] / f'rank-{rank}-buffers.npz', allow_pickle=False) as left, \
             np.load(directories[1] / f'rank-{rank}-buffers.npz', allow_pickle=False) as right:
            for layer in LAYERS:
                prefix = f'layer{layer:03d}__'
                labels = [key[len(prefix):] for key in left.files
                          if key.startswith(prefix) and not key.endswith(('__counts', '__errors'))]
                for label in labels:
                    specs = [meta['arrays'][prefix + label] for meta in metas]
                    if any(specs[0].get(field) != specs[1].get(field)
                           for field in ('logical_shape', 'torch_dtype', 'cache_layout')):
                        raise ValueError('paired trace tensor representations differ')
                pairs = []
                for archive, meta in zip((left, right), metas):
                    pairs.append({label: decode(archive[prefix + label], meta['arrays'][prefix + label]) for label in labels})
                layouts = [meta['arrays'][prefix + 'attention_cache']['cache_layout'] for meta in metas]
                if layouts[0] != layouts[1]:
                    raise ValueError('paired attention cache layouts differ')
                for index, position in enumerate(POSITIONS):
                    a, b = ({label: data[index] for label, data in pair.items()} for pair in pairs)
                    for values in (a, b):
                        verify_row(values, position, layouts[0])
                    rows.append({'tp_rank': rank, 'layer': layer, 'position': position, **compare_row(a, b)})
    rows.sort(key=lambda row: (row['position'], row['layer'], row['tp_rank']))
    index_matches = [row for row in rows if row['index_order_divergence_with_same_inputs']]
    attention_matches = [row for row in rows if row['order_linked_attention_divergence']]
    def identity(row):
        return {key: row[key] for key in ('position', 'layer', 'tp_rank')} if row else None
    if index_matches and attention_matches:
        conclusion = 'index order non-repeatability reproduced with same recorded scorer inputs; order-linked attention divergence supports the proposed mechanism, not an intervention-proven fix'
    elif index_matches:
        conclusion = 'index order non-repeatability reproduced; recorded attention evidence does not yet establish the proposed downstream linkage'
    else:
        conclusion = 'same-input index order divergence not reproduced in these two instrumented runs; inconclusive, not falsification of a scheduling race'
    return {'schema': 'glm53-p8.index-order-analysis.v1', 'positions': POSITIONS, 'layers': LAYERS,
        'independent_processes': 2, 'window_count': 1,
        'experimental_unit': 'fresh serving process; rows/layers/ranks are correlated subsamples',
        'index_order_matching_rows': len(index_matches), 'attention_link_matching_rows': len(attention_matches),
        'first_index_order_match': identity(index_matches[0] if index_matches else None),
        'first_attention_link_match': identity(attention_matches[0] if attention_matches else None),
        'rows': rows, 'conclusion': conclusion, 'kld_measured': False, 'qualification_pass': False,
        'speed_measurement_valid': False, 'allocation_restart': False,
        'rivals': ['observer timing may change CTA order', 'unobserved internal scratch or reduction state',
                   'earlier numerical divergence can propagate to later layers'],
        'next_action': 'only after inspection, preregister a targeted intervention or redirect to the first differing input; never silently qualify or retry'}
