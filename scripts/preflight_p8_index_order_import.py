"""CPU-only exact-image import/source-introspection gate for the CuTe patch."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from scripts.preflight_p8_index_order_device import IMAGE, MODULE, SOURCE_SHA, load_modules


def check():
    import torch
    if torch.cuda.is_available():
        raise ValueError('CPU preflight must not have CUDA device access')
    original, candidate, receipt = load_modules()
    candidate_source = inspect.getsource(candidate)
    original_source = inspect.getsource(original)
    if hashlib.sha256(candidate_source.encode()).hexdigest() != receipt['emitted_sha256']:
        raise ValueError('inspect sees different candidate source than the executed transformation')
    if hashlib.sha256(original_source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError('reference module source introspection is contaminated')
    source = inspect.getsource(candidate.SparseNSAFusedIndexerKernel)
    reference = inspect.getsource(original.SparseNSAFusedIndexerKernel)
    if 'p8_direct_short' not in source or 'p8_direct_short' in reference:
        raise ValueError('CuTe class source isolation failed')
    if '_p8logicalshortv1' not in inspect.getsource(candidate._launch_fused):
        raise ValueError('distinct candidate compile-key not inspect-visible')
    if '_p8logicalshortv1' in inspect.getsource(original._launch_fused):
        raise ValueError('original compile-key changed')
    root = Path(__file__).resolve().parents[1]
    names = ('scripts/preflight_p8_index_order_import.py',
             'scripts/preflight_p8_index_order_device.py', 'runtime_patch/sitecustomize.py',
             'runtime_patch/p8_index_order/__init__.py', 'runtime_patch/p8_index_order/patches.py')
    return {'schema': 'glm53-p8.index-order-import.v1', 'status': 'passed',
            'image_id': IMAGE, 'module': MODULE, 'source_transformation': receipt,
            'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            'module_source_exact': True, 'cute_class_source_isolated': True,
            'compile_cache_identity_isolated': True,
            'gpu_used': False, 'model_loaded': False, 'teacher_logits_opened': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
