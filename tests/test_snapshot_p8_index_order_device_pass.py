import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_pass', REPO / 'scripts/snapshot_p8_index_order_device_pass.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def inputs():
    plan = json.loads(mod.PLAN.read_text())
    raw = Path(plan['output'])
    root = json.loads((raw / 'execution.json').read_text())
    results = [json.loads((raw / f'repeat-{i:02d}/result.json').read_text()) for i in range(1, 6)]
    return root, results, plan


def test_five_process_synthetic_only():
    report = mod.classify(*inputs())
    assert report['total_check_calls'] == 2300
    assert report['five_fresh_processes'] is True
    assert report['full_model_numerical_closure'] is False
    assert report['kld_measured'] is False


@pytest.mark.parametrize('fault', ['repeat_count', 'order', 'counter', 'restoration', 'signature'])
def test_fail_closed(fault):
    root, results, plan = inputs()
    if fault == 'repeat_count':
        results.pop()
    elif fault == 'order':
        root['repeats'].reverse()
    elif fault == 'counter':
        results[3]['cases'][0]['counter_zero'] = False
    elif fault == 'restoration':
        root['restoration']['safe'] = False
    else:
        results[4]['determinism_sha256'] = 'a' * 64
        root['repeats'][4]['determinism_sha256'] = 'a' * 64
    with pytest.raises(ValueError):
        mod.classify(root, results, plan)
