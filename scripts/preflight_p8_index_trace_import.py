"""CPU reproduction of the worker's custom-op schema registration seam.

Run with the exact serving PYTHONPATH, GLM53_P8_INDEX_TRACE=1, no CUDA devices.
Unlike a bare CPU import (which skips CUDA custom-op registration), explicitly
infer the real indexer's schema and compare it with the original source AST.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

if os.environ.get('VLLM_USE_BREAKABLE_CUDAGRAPH') != '1':
    raise RuntimeError('import preflight requires the actual worker breakable-graph environment')

import torch
import vllm.models.glm5next.nvidia.model  # same model-first ordering as serving
import vllm.model_executor.layers.sparse_attn_indexer_kpool as indexer
import vllm.v1.attention.backends.mla.b12x_mla_sparse
import p8_decode_capture.v2_hook
import p8_index_trace
from p8_index_trace import TRANSFORMATIONS


def check():
    source_path = Path(indexer.__file__)
    original = source_path.read_bytes()
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sparse_attn_indexer_kpool')
    node.body = [ast.Pass()]
    ast.fix_missing_locations(node)
    namespace = dict(indexer.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-schema>',
                 'exec', dont_inherit=True), namespace)
    reference, observed = namespace[node.name], indexer.sparse_attn_indexer_kpool
    mutated = ['topk_indices_buffer', 'kv_cache', 'tail_kv_cache']
    expected_schema = torch.library.infer_schema(reference, mutates_args=mutated)
    error = None
    try:
        actual_schema = torch.library.infer_schema(observed, mutates_args=mutated)
    except Exception as exc:
        actual_schema = None
        error = {'type': type(exc).__name__, 'message': str(exc)}
    annotation_mode_preserved = observed.__annotations__ == reference.__annotations__
    same = expected_schema == actual_schema
    package = Path(p8_index_trace.__file__).resolve().parent
    files = {'runtime_patch/p8_index_trace/__init__.py': package / '__init__.py',
             'runtime_patch/p8_index_trace/patches.py': package / 'patches.py',
             'scripts/preflight_p8_index_trace_import.py': Path(__file__).resolve()}
    return {'schema': 'glm53-p8.index-import-preflight.v2',
        'status': 'passed' if same and annotation_mode_preserved else 'failed',
        'image_id': 'sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9',
        'schemas_equal': same, 'annotation_mode_preserved': annotation_mode_preserved,
        'breakable_cudagraph': True,
        'expected_schema': expected_schema, 'actual_schema': actual_schema, 'error': error,
        'observed_signature': str(inspect.signature(observed)),
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        'source_transformations': TRANSFORMATIONS,
        'gpu_used': False, 'model_loaded': False, 'teacher_logits_opened': False,
        'speed_measurement_valid': False}


if __name__ == '__main__':
    result = check()
    print('INDEX_IMPORT_PREFLIGHT_JSON=' + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
