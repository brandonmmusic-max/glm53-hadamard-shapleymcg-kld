import copy
import pytest
from glm53_nvfp4 import p8_index_order_device_runner as runner


def test_synthetic_mount_boundary():
    plan = {'gpu_inventory': ['0, GPU-test, 00:00, 300'], 'seed': 20260905}
    args = runner.argv(plan, runner.ROOT / 'index-order-device-v1', 1, 'a' * 64)
    mounts = [args[i+1] for i, arg in enumerate(args) if arg == '-v']
    assert len(mounts) == 4
    assert all('teacher' not in m and 'sidecar' not in m for m in mounts)
    assert args[args.index('--gpus') + 1] == 'device=GPU-test'
    assert args[args.index('--network') + 1] == 'none'
    assert 'GLM53_P8_NATIVE=' in args
    assert 'GLM53_P8_INDEX_ORDER=logical-short-v1' in args


def sample():
    sources = {name: 'b' * 64 for name in runner.SOURCE_FILES}
    fields = ('arm', 'policy', 'scenario', 'length', 'mode', 'repeat')
    cases, transitions = runner.expected_case_inventory()
    def rows(items):
        return [{**dict(zip(fields, item)), 'counter_zero': True,
                 'ctas_per_group': 4 if item[1] == 'forced-boundary' else 188,
                 'merge_threshold': 1024} for item in items]
    return {'schema': 'glm53-p8.index-order-device.v1', 'status': 'passed',
            'image_id': runner.IMAGE, 'model_loaded': False,
            'teacher_logits_opened': False, 'speed_measurement_valid': False,
            'gpu': {'uuid': 'GPU-test', 'compute_capability': [12,0], 'multiprocessors': 188},
            'prefix_gate_points': list(runner.LENGTHS), 'dynamic_length_transitions': list(runner.TRANSITIONS),
            'eager_repeats': 5, 'graph_repeats': 5, 'cases': rows(cases), 'state_transitions': rows(transitions),
            'determinism_sha256': 'a' * 64, 'source_sha256': sources}, {
                'source_sha256': sources, 'gpu_inventory': ['0, GPU-test, 00:00, 300']}


def test_valid_result():
    result, plan = sample()
    assert runner.validate_result(result, plan) == 'a' * 64


@pytest.mark.parametrize('key,value', [('status','failed'), ('model_loaded',True),
    ('teacher_logits_opened',True), ('speed_measurement_valid',True), ('determinism_sha256','')])
def test_result_rejects_claim_drift(key, value):
    result, plan = sample()
    result[key] = value
    with pytest.raises(ValueError):
        runner.validate_result(result, plan)


def test_result_rejects_source_drift():
    result, plan = sample()
    result = copy.deepcopy(result)
    result['source_sha256']['runtime_patch/p8_index_order/patches.py'] = 'c' * 64
    with pytest.raises(ValueError):
        runner.validate_result(result, plan)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_gpu', 'counter'])
def test_case_inventory_and_device(mutation):
    result, plan = sample()
    if mutation == 'missing':
        result['cases'].pop()
    elif mutation == 'duplicate':
        result['cases'].append(result['cases'][0])
    elif mutation == 'wrong_gpu':
        result['gpu']['uuid'] = 'GPU-other'
    else:
        result['cases'][0]['counter_zero'] = False
    with pytest.raises(ValueError):
        runner.validate_result(result, plan)


def test_daemon_failure_withholds_restore_and_records_errors(monkeypatch):
    calls = []
    def broken(args, **kwargs):
        calls.append(args)
        raise RuntimeError('unavailable')
    monkeypatch.setattr(runner, 'command', broken)
    result = runner.restore({'backend': True, 'timer': False}, True)
    assert result['safe'] is False
    assert result['backend'] is None and result['timer'] is None
    assert result['errors']
    assert not any('start' in args for args in calls)
