import copy
import hashlib
import json
from pathlib import Path

import pytest

from glm53_nvfp4.preflight_p8_speed_v2 import verify

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    prior_bytes = (ROOT / 'experiments/p8-uniform-all42-tp4-vs-exl3-speed-v1.json').read_bytes()
    plan = json.loads((ROOT / 'experiments/p8-uniform-all42-tp4-vs-exl3-speed-v2.json').read_text())
    return plan, json.loads(prior_bytes), hashlib.sha256(prior_bytes).hexdigest()


def test_accepted_explicit_topology_amendment():
    verify(*inputs(), 'checkpoint=2, runtime=4')


@pytest.mark.parametrize('field', ['topology', 'duration', 'threshold', 'image', 'kv'])
def test_protocol_drift_rejected(field):
    plan, prior, digest = inputs()
    plan = copy.deepcopy(plan)
    if field == 'topology':
        plan['baseline']['topology']['dcp'] = 1
    elif field == 'duration':
        plan['benchmark']['duration_seconds_per_decode_cell'] = 30
    elif field == 'threshold':
        plan['decision_before_result']['pass'] = 'always pass'
    elif field == 'image':
        plan['candidate']['image_id'] = 'wrong'
    else:
        plan['common_serving_regime']['kv_cache_dtype'] = 'bf16'
    with pytest.raises(ValueError):
        verify(plan, prior, digest, 'checkpoint=2, runtime=4')


def test_original_failure_and_plan_required():
    with pytest.raises(ValueError):
        verify(*inputs(), 'no matching failure')
    plan, prior, _ = inputs()
    with pytest.raises(ValueError):
        verify(plan, prior, 'wrong', 'checkpoint=2, runtime=4')
