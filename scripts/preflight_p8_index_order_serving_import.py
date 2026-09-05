"""Exact-image, worker-mode CPU import check for serving receipt composition."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path


def check():
    if os.environ.get('VLLM_USE_BREAKABLE_CUDAGRAPH') != '1':
        raise ValueError('actual worker breakable-graph mode is required')
    if os.environ.get('GLM53_P8_INDEX_ORDER_RECEIPT') != '1':
        raise ValueError('serving receipt composition must be enabled')
    import vllm.models.glm5next.nvidia.model
    from scripts.preflight_p8_index_order_import import check as base_check
    result = base_check()
    candidate = importlib.import_module(result['module'])
    fn = candidate.run_fused_paged_indexer
    if not getattr(fn, '_glm53_p8_index_order_receipt', False):
        raise ValueError('serving entry point was not receipt-wrapped')
    original = inspect.unwrap(fn)
    if original is fn or original.__globals__ is not candidate.__dict__:
        raise ValueError('receipt must delegate the source-qualified candidate entry point')
    if not original.__code__.co_filename.endswith('.p8logicalshortv1.py'):
        raise ValueError('delegate does not use the transformed source filename')
    parallel = importlib.import_module('vllm.distributed.parallel_state')
    for name in ('model_parallel_is_initialized', 'get_tensor_model_parallel_rank',
                 'get_tensor_model_parallel_world_size'):
        if not callable(getattr(parallel, name, None)):
            raise ValueError('pinned vLLM rank API missing')
    root = Path(__file__).resolve().parents[1]
    extra = ('scripts/preflight_p8_index_order_serving_import.py',
             'runtime_patch/p8_index_order_receipt.py')
    result['source_sha256'].update({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in extra})
    result.update(schema='glm53-p8.index-order-serving-import.v1',
                  receipt_wrapper_installed=True, wrapped_source_globals_exact=True,
                  worker_breakable_graph_mode=True, serving_rank_apis_available=True,
                  marker_emitted=False, actual_distributed_call_tested=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
