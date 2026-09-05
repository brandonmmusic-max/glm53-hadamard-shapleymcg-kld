"""Exact-source, diagnostic-only insertions into the pinned serving image.

These are observers, not changes to the B12X selection or attention algorithms.
The original image files remain untouched; emitted source hashes are receipts.
"""
from __future__ import annotations

import hashlib

INDEXER = 'vllm.model_executor.layers.sparse_attn_indexer_kpool'
BACKEND = 'vllm.v1.attention.backends.mla.b12x_mla_sparse'
CAPTURE = 'p8_decode_capture.v2_hook'
ORIGINAL_SHA = {
    INDEXER: 'ac6b64227346e76c5cf7d8c5e5011b4e32d7c2c811982338a6c24513c1304229',
    BACKEND: '0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572',
    CAPTURE: '62f35d78931a7251cd3be594d3228b0f13868f39b3e25adb5c0ad8a96c87c683',
}


def replace_once(source: str, anchor: str, replacement: str) -> str:
    if source.count(anchor) != 1:
        raise ValueError('index-trace source anchor missing or ambiguous')
    return source.replace(anchor, replacement, 1)


def transform(name: str, raw: bytes) -> tuple[str, dict]:
    original = hashlib.sha256(raw).hexdigest()
    if name not in ORIGINAL_SHA or original != ORIGINAL_SHA[name]:
        raise ValueError(f'index-trace original source identity differs: {name}')
    source = raw.decode('utf-8')
    if name == INDEXER:
        anchor = '                topk_scores=pool_scores,\n            )\n            if index_kpool > 1:\n'
        insertion = '''            from p8_index_trace.observer import observe_pool
            observe_pool(k_cache_prefix, positions, b12x_topk, kv_cache_raw,
                         b12x_seq_lens, b12x_block_table,
                         q_quant[:score_rows], weights[:score_rows])
'''
        source = replace_once(source, anchor, anchor.replace('            if index_kpool > 1:\n', insertion + '            if index_kpool > 1:\n'))
    elif name == BACKEND:
        anchor = '        if use_ckv_gather:\n            layer_idx = self._resolve_layer_index(layer)\n'
        insertion = '''        from p8_index_trace.observer import observe_attention_inputs
        observe_attention_inputs(layer.layer_name, per_token_cache, q_all,
                                 self.topk_indices_buffer[:num_actual_toks, :self.topk_tokens + 3],
                                 selected_indices, kv_cache,
                                 attn_metadata.block_table, self.block_size)
'''
        source = replace_once(source, anchor, insertion + anchor)
        source += '\nfrom p8_index_trace.observer import wrap_backend\nwrap_backend(B12xMLASparseImpl)\n'
    else:
        source += '\nfrom p8_index_trace.observer import wrap_capture\nwrap_capture(V2ForcedDecodeCapture)\n'
    compile(source, f'<index-trace:{name}>', 'exec')
    return source, {'module': name, 'original_sha256': original,
                    'emitted_sha256': hashlib.sha256(source.encode()).hexdigest()}
