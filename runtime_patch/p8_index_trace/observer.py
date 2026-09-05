"""Read-only GPU observers for the 65th-pool boundary, never a speed endpoint.

Triton copies are recorded inside opaque serving ops. Only sampler seams reset
or download buffers. There is no sorting, scoring, weight or activation change.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
import triton
import triton.language as tl

from . import TRANSFORMATIONS
from .patches import ORIGINAL_SHA

FIRST, LAST = 255, 263
POSITIONS = list(range(FIRST, LAST + 1))
LAYERS = list(range(3, 44, 4))
_BUFFERS = {}
_WINDOW = None
_FINISHED = False
_FROZEN_KEYS = None


@triton.jit
def _copy_bytes(SOURCE, POSITION, DEST, COUNTS, N: tl.constexpr,
                OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.load(POSITION) + OFFSET
    if (pos >= 255) & (pos <= 263):
        row = pos - 255
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(SOURCE + i, i < N, other=0)
        tl.store(DEST + row * N + i, value, i < N)
        if tl.program_id(0) == 0:
            count = tl.load(COUNTS + row)
            tl.store(COUNTS + row, count + 1)


@triton.jit
def _copy_cache(CACHE, POSITION, LENGTH, TABLE, DEST, COUNTS, ERRORS,
                OFFSET: tl.constexpr, PAGE: tl.constexpr,
                PAGE_BYTES: tl.constexpr, NPAGES: tl.constexpr,
                TABLE_WIDTH: tl.constexpr, RECORDS: tl.constexpr,
                WIDTH: tl.constexpr, POOLED: tl.constexpr,
                BLOCK: tl.constexpr):
    pos = tl.load(POSITION) + OFFSET
    if (pos >= 255) & (pos <= 263):
        row = pos - 255
        item = tl.program_id(0)
        byte = tl.arange(0, BLOCK)
        length = tl.load(LENGTH)
        valid = item < length
        logical_page = item // PAGE
        physical = tl.load(TABLE + logical_page,
                           (logical_page < TABLE_WIDTH) & valid, other=-1)
        address_valid = (physical >= 0) & (physical < NPAGES)
        slot = item % PAGE
        if POOLED:
            # Page-wide FP8 payload followed by page-wide FP32 scales.
            local = tl.where(byte < 128, slot * 128 + byte,
                             PAGE * 128 + slot * 4 + byte - 128)
        else:
            local = slot * WIDTH + byte
        value = tl.load(CACHE + physical * PAGE_BYTES + local,
                        valid & address_valid & (byte < WIDTH), other=0)
        tl.store(DEST + (row * RECORDS + item) * WIDTH + byte,
                 value, byte < WIDTH)
        if valid & ~address_valid:
            tl.atomic_add(ERRORS + row, 1)
        if item == 0:
            count = tl.load(COUNTS + row)
            tl.store(COUNTS + row, count + 1)


def _layer(prefix):
    match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', prefix)
    if match is None or int(match[1]) not in LAYERS:
        raise RuntimeError(f'unexpected sparse layer identity: {prefix}')
    return int(match[1])


def _slot(layer, label, nbytes, shape, dtype, device):
    key = f'layer{layer:03d}__{label}'
    signature = {'logical_shape': list(shape), 'torch_dtype': str(dtype)}
    entry = _BUFFERS.get(key)
    if entry is None:
        if _WINDOW is not None:
            raise RuntimeError('observer allocated after the real request began')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('observer buffers must be allocated in eager warmup before capture')
        entry = {**signature, 'data': torch.zeros((9, nbytes), dtype=torch.uint8, device=device),
                 'counts': torch.zeros(9, dtype=torch.int32, device=device),
                 'errors': torch.zeros(9, dtype=torch.int32, device=device),
                 'recorded_in_cuda_graph': False}
        entry['allocation_pointers'] = {field: entry[field].data_ptr() for field in ('data', 'counts', 'errors')}
        _BUFFERS[key] = entry
    if any(entry[k] != v for k, v in signature.items()) or entry['data'].shape != (9, nbytes):
        raise RuntimeError('observer tensor geometry changed across warmup/graph/replay')
    if any(entry[field].data_ptr() != pointer for field, pointer in entry['allocation_pointers'].items()):
        raise RuntimeError('observer allocation pointer changed')
    if torch.cuda.is_current_stream_capturing():
        entry['recorded_in_cuda_graph'] = True
    return entry


def _copy(layer, label, tensor, position, *, offset=0):
    if position is None or position.numel() != 1 or tensor.shape[0] != 1:
        return
    if not tensor.is_cuda or not position.is_cuda:
        raise RuntimeError('index observer requires real CUDA tensors')
    source = tensor.contiguous().view(torch.uint8).reshape(-1)
    entry = _slot(layer, label, source.numel(), tensor.shape, tensor.dtype, tensor.device)
    _copy_bytes[(triton.cdiv(source.numel(), 256),)](
        source, position, entry['data'], entry['counts'], source.numel(), offset, 256)


def _cache(layer, label, cache, position, length, table, *, offset, records, pooled):
    if cache.dtype != torch.uint8 or cache.ndim != 3 or not cache.is_contiguous():
        raise RuntimeError('observer cache layout differs')
    if table.ndim != 2 or table.shape[0] != 1 or table.stride(1) != 1:
        raise RuntimeError('observer block table differs')
    page, width = cache.shape[1:]
    if pooled and (page, width) != (64, 132):
        raise RuntimeError('observer requires the pinned split-payload KPool4 cache')
    entry = _slot(layer, label, records * width, (records, width), torch.uint8, cache.device)
    layout = {'page_size': page, 'page_bytes': cache.stride(0),
              'layout': 'page_split_payload128_scale4' if pooled else 'record_major'}
    if 'cache_layout' in entry and entry['cache_layout'] != layout:
        raise RuntimeError('observer cache layout changed')
    entry['cache_layout'] = layout
    _copy_cache[(records,)](cache, position, length, table,
        entry['data'], entry['counts'], entry['errors'], offset, page,
        cache.stride(0), cache.shape[0], table.shape[1], records, width, pooled,
        triton.next_power_of_2(width))


def observe_pool(prefix, positions, pool_ids, cache, lengths, table, query, weights):
    if positions is None or positions.numel() != 1 or pool_ids.shape[0] != 1:
        return
    layer = _layer(prefix)
    if pool_ids.shape != (1, 512) or lengths.numel() != 1:
        raise RuntimeError('observer requires C1 DCP1 topk512 pools')
    _copy(layer, 'pool_ids', pool_ids, positions)
    _copy(layer, 'pool_lengths', lengths, positions)
    _copy(layer, 'pool_table', table[:, :2], positions)
    _copy(layer, 'index_query', query, positions)
    _copy(layer, 'index_weights', weights, positions)
    _cache(layer, 'pool_cache', cache, positions, lengths, table,
           offset=0, records=66, pooled=True)


def observe_attention_inputs(prefix, lengths, query, logical, selected, cache, table, block_size):
    if lengths.numel() != 1 or query.shape[0] != 1:
        return
    layer = _layer(prefix)
    if cache.shape[1] != block_size:
        raise RuntimeError('MLA observer page size differs')
    _copy(layer, 'attention_lengths', lengths, lengths, offset=-1)
    _copy(layer, 'attention_query', query, lengths, offset=-1)
    _copy(layer, 'logical_tokens', logical, lengths, offset=-1)
    _copy(layer, 'physical_slots', selected, lengths, offset=-1)
    _copy(layer, 'attention_table', table[:, :triton.cdiv(264, block_size)], lengths, offset=-1)
    _cache(layer, 'attention_cache', cache, lengths, lengths, table,
           offset=-1, records=264, pooled=False)


def wrap_backend(cls):
    original = cls.forward_mqa
    if getattr(original, '_p8_index_trace', False):
        raise RuntimeError('duplicate backend observer')

    @functools.wraps(original)
    def forward(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        result = original(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        lengths = attn_metadata.cache_seq_lens_per_token
        if lengths.numel() == 1:
            _copy(_layer(layer.layer_name), 'attention_output', result[0], lengths, offset=-1)
        return result

    forward._p8_index_trace = True
    cls.forward_mqa = forward


def _begin(window):
    global _WINDOW, _FROZEN_KEYS
    if _WINDOW is not None or _FINISHED:
        raise RuntimeError('diagnostic admits exactly one real request per process')
    if not _BUFFERS:
        raise RuntimeError('no observer nodes recorded during startup')
    _WINDOW = window
    _FROZEN_KEYS = set(_BUFFERS)
    for entry in _BUFFERS.values():
        # Outside the model graph, after the lexical startup warmup closes.
        entry['data'].zero_()
        entry['counts'].zero_()
        entry['errors'].zero_()


def _finish(window, rank):
    global _FINISHED
    if _FINISHED or window != _WINDOW or set(_BUFFERS) != _FROZEN_KEYS:
        raise RuntimeError('invalid trace completion lifecycle')
    required_labels = {'pool_ids', 'pool_lengths', 'pool_table', 'index_query',
        'index_weights', 'pool_cache', 'attention_lengths', 'attention_query',
        'logical_tokens', 'physical_slots', 'attention_table', 'attention_cache',
        'attention_output'}
    if set(_BUFFERS) != {f'layer{layer:03d}__{label}' for layer in LAYERS for label in required_labels}:
        raise RuntimeError('trace layer/seam inventory incomplete')
    if set(TRANSFORMATIONS) != set(ORIGINAL_SHA):
        raise RuntimeError('trace source transformation inventory incomplete')
    arrays, specs = {}, {}
    counts_valid = True
    for key, entry in sorted(_BUFFERS.items()):
        if any(entry[field].data_ptr() != pointer for field, pointer in entry['allocation_pointers'].items()):
            raise RuntimeError('observer allocation changed before download')
        for suffix, field in (('', 'data'), ('__counts', 'counts'), ('__errors', 'errors')):
            value = entry[field].detach().cpu().numpy()
            arrays[key + suffix] = value
            specs[key + suffix] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        specs[key].update({k: entry[k] for k in ('logical_shape', 'torch_dtype', 'recorded_in_cuda_graph')})
        specs[key]['allocation_pointers'] = entry['allocation_pointers']
        specs[key]['allocation_stable'] = True
        if 'cache_layout' in entry:
            specs[key]['cache_layout'] = entry['cache_layout']
        counts_valid &= bool(np.all(arrays[key + '__counts'] == 1))
        counts_valid &= bool(np.all(arrays[key + '__errors'] == 0))
        counts_valid &= entry['recorded_in_cuda_graph']
    root = Path(os.environ['GLM53_P8_INDEX_TRACE_ROOT'])
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError('invalid trace root')
    raw = root / f'rank-{rank}-buffers.npz'
    with raw.open('xb') as stream:
        np.savez(stream, **arrays)
    os.chmod(raw, 0o644)
    metadata = {'schema': 'glm53-p8.index-trace.v1', 'window_id': window,
        'tp_rank': rank, 'positions': POSITIONS, 'layers': LAYERS, 'arrays': specs,
        'raw_file': raw.name, 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'source_transformations': TRANSFORMATIONS, 'capture_complete': True,
        'counts_valid': bool(counts_valid), 'speed_measurement_valid': False,
        'protected_roles_opened': [],
        'limits': ['Observer kernels may perturb scheduling; this is not a speed run.',
                   'DCP1 scoring values are not copied; all pools fit within topk512.',
                   'Attention seams are inner MLA, not whole hidden-state projections.',
                   'Invalid cache rows are masked and saved as zero, not read.']}
    target = root / f'rank-{rank}.json'
    temporary = root / f'rank-{rank}.json.tmp'
    if target.exists():
        raise RuntimeError('existing trace manifest cannot be overwritten')
    with temporary.open('x') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.chmod(temporary, 0o644)
    # Exclusive publication avoids a reader observing partial JSON or overwrite.
    os.link(temporary, target)
    temporary.unlink()
    if not counts_valid:
        raise RuntimeError('trace rows missing/duplicated/not graphed or invalid cache address')
    _FINISHED = True
    print(f'GLM53_P8_INDEX_TRACE_COMPLETE tp_rank={rank} window={window} rows=9 layers=11', flush=True)


def wrap_capture(cls):
    original_add, original_capture = cls.add_request, cls.capture_and_force
    if getattr(original_add, '_p8_index_trace', False):
        raise RuntimeError('duplicate sampler observer')

    @functools.wraps(original_add)
    def add(self, req_idx, prompt_len, sampling_params):
        result = original_add(self, req_idx, prompt_len, sampling_params)
        if self.enabled and self._pinned_warmup is None:
            _begin(self.states[req_idx].spec.window_id)
        return result

    @functools.wraps(original_capture)
    def capture(self, logits, input_batch):
        result = original_capture(self, logits, input_batch)
        if self.enabled and self._pinned_warmup is None and self.states:
            state = self.states.get(0)
            if state is not None and state.complete:
                _finish(state.spec.window_id, self.tp_rank)
        return result

    add._p8_index_trace = True
    cls.add_request, cls.capture_and_force = add, capture
