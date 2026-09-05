import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_stop', REPO / 'scripts/snapshot_p8_index_order_device_stop.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def inputs():
    plan = json.loads(mod.PLAN.read_text())
    raw = Path(plan['output'])
    return json.loads((raw / 'repeat-01/result.json').read_text()), plan, json.loads((raw / 'execution.json').read_text())


def test_prefix_only_is_not_five_process_pass():
    result, plan, root = inputs()
    original = copy.deepcopy(result)
    report = mod.classify(result, plan, root)
    assert result == original
    assert report['case_calls'] + report['state_transition_calls'] == 460
    assert report['five_process_gate_passed'] is False
    assert report['host_accepted_repeats'] == 0


@pytest.mark.parametrize('fault', ['uuid', 'restoration', 'counter', 'later_repeat'])
def test_fail_closed(fault):
    result, plan, root = inputs()
    if fault == 'uuid':
        result['gpu']['uuid'] = 'different'
    elif fault == 'restoration':
        root['restoration']['safe'] = False
    elif fault == 'counter':
        result['cases'][0]['counter_zero'] = False
    else:
        root['repeats'] = [1]
    with pytest.raises(ValueError):
        mod.classify(result, plan, root)
