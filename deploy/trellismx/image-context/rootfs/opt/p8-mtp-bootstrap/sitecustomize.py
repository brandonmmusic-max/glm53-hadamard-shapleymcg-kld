"""Preserve the pinned runtime bootstrap and resolve GLM MTP ModelOpt names."""
import re
import runpy
import os
import importlib.util
from pathlib import Path

if os.environ.get('GLM53_P8_VERIFY_MULTIROW_IMPORT') == '1':
    root = Path(importlib.util.find_spec('b12x').origin).parent
    dynamic = root / 'moe/_shared/kernels/dynamic.py'
    fc1 = root / 'moe/_shared/kernels/p8_h128_fc1.py'
    if ('a_input[p8_input_row, m1_col]' not in dynamic.read_text()
            or 'task_expert.shape[0]) * tasks_per_route' not in fc1.read_text()):
        # sitecustomize exceptions normally only print a warning; exit hard.
        os.write(2, b'P8_MULTIROW_IMPORT_INVALID: refusing incompatible kernels\n')
        os._exit(78)
    print(f'P8_MULTIROW_IMPORT_VERIFIED root={root}', flush=True)

runpy.run_path('/etc/python3.12/sitecustomize.py')

from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig

_original_candidates = ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates


def _mtp_candidates(prefix):
    candidates = list(_original_candidates(prefix))
    # GLM's text-only MTP module is constructed outside the multimodal tower.
    # Its optional mtp_block wrapper has no counterpart in checkpoint names.
    match = re.fullmatch(r'model\.layers\.(\d+)\.(?:mtp_block\.)?(.+)', prefix)
    if match:
        canonical = f'model.language_model.layers.{match[1]}.{match[2]}'
        candidates.extend(_original_candidates(canonical))
    return tuple(dict.fromkeys(candidates))


ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates = staticmethod(_mtp_candidates)
print('P8_MTP_MODELOPT_PREFIX_ALIAS_ACTIVE', flush=True)
