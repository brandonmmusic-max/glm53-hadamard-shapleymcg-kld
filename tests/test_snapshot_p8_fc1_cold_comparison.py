import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from glm53_nvfp4 import p8_fc1_cold_compare as cold

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_cold', REPO / 'scripts/snapshot_p8_fc1_cold_comparison.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def runtime(arm):
    lines = ['tensor_parallel_size=4 secret=DO_NOT_PUBLISH',
             f'decode_context_parallel_size={cold.TOPOLOGY[arm]["dcp"]}',
             'speculative_config=None secret=DO_NOT_PUBLISH', 'kv_cache_dtype=nvfp4_ds_mla',
             'enforce_eager=False', 'Breakable CUDA graph enabled', 'Capturing CUDA graphs (FULL): 100%',
             *[f'(Worker_TP{rank} pid=8) Graph capturing finished secret=DO_NOT_PUBLISH' for rank in range(4)]]
    if arm == 'exl3':
        lines += ["'enable_expert_parallel': True secret=DO_NOT_PUBLISH", 'quantization=exl3',
                  'EXL3 full-expert EP runtime planned secret=DO_NOT_PUBLISH',
                  'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts secret=DO_NOT_PUBLISH']
    else:
        for layer in range(3, 45):
            for rank in range(4):
                lines += [f'GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} secret=DO_NOT_PUBLISH small_m_scheduler=true',
                          f'GLM53_P8_M1_DISPATCH layer={layer} rank={rank} fc1_tile_n=64 fused_scratch_zero=true']
    return '\n'.join(lines)


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    source = root / 'fc1-cold-comparison-v1'
    source.mkdir(parents=True)
    dest = tmp_path / 'evidence' / 'p8-fc1-cold-comparison-v1'
    monkeypatch.setattr(cold, 'ROOT', root)
    monkeypatch.setattr(snapshot, 'SOURCE', source)
    monkeypatch.setattr(snapshot, 'DEST', dest)
    weight = root / 'weights.json'
    weight.write_text('{"payload_manifest_hash":"frozen"}\n')
    plan = {'output': str(source), 'weight_audit': str(weight), 'weight_audit_sha256': cold.pilot.sha(weight)}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    plan_path.with_suffix('.sha256').write_text(cold.pilot.sha(plan_path))
    monkeypatch.setattr(snapshot, 'verify_plan_for_snapshot', lambda path: plan)
    state = {'ActiveState': 'inactive', 'Result': 'success'}
    monkeypatch.setattr(snapshot, 'terminal_state', lambda: state.copy())
    analysis = {'speed_gate_pass': True,
                'medians': {'p8': {'decode_tokens_per_second': 100}, 'exl3': {'decode_tokens_per_second': 90}},
                'primary_wins': {'decode_tokens_per_second': True, 'prefill_server_tokens_per_second': True}}
    analysis_calls = []
    monkeypatch.setattr(cold, 'analyze', lambda *args: analysis_calls.append(args) or analysis)
    (source / 'analysis.json').write_text(json.dumps(analysis))
    (source / 'hardware-inventory.json').write_text('["four-physical-GPUs"]')
    (source / 'weight-audit-prerequisite.json').write_bytes(weight.read_bytes())
    runs = []
    for index, entry in enumerate(cold.ORDER):
        slot = source / cold.slot_name(entry)
        slot.mkdir()
        data = json.loads((REPO / 'evidence/opened/codec-v2/p8-fc1-integrated-v1/result.json').read_text())
        data['metadata'].update(model=cold.model_name(entry['arm']), server=f'127.0.0.1:{cold.PORT}')
        safe_files = {'result.json': json.dumps(data), 'summary.json': json.dumps(cold.summarize(data, entry['arm'])),
                      'runtime-audit.json': json.dumps(cold.runtime_audit(runtime(entry['arm']), entry['arm'])),
                      'thermal.jsonl': '{"temperatures":[70,70,70,70]}\n',
                      'cooldown.jsonl': '{"temperatures":[60,60,60,60]}\n', 'models.json': '{}',
                      'nvidia-before.xml': '<hardware/>', 'nvidia-after.xml': '<hardware/>',
                      'server-final.private.log': runtime(entry['arm']), 'server-ready.private.log': runtime(entry['arm']),
                      'container-final.private.json': '{"secret":"DO_NOT_PUBLISH"}',
                      'launch-spec.private.json': '{"secret":"DO_NOT_PUBLISH"}',
                      'error.private.txt': 'DO_NOT_PUBLISH', 'benchmark.private.log': 'DO_NOT_PUBLISH'}
        for name, value in safe_files.items():
            (slot / name).write_text(value)
        receipt = {**entry, 'schema': 'glm53-p8-fc1-cold-run.v1', 'plan_sha256': cold.pilot.sha(plan_path),
                   'image': cold.IMAGES[entry['arm']], 'topology': cold.TOPOLOGY[entry['arm']],
                   'protected_roles_opened': [], 'allocation_restart': False, 'exit_code': 0,
                   'finished_at': 'terminal', 'files': cold.file_inventory(slot)}
        (slot / 'execution.json').write_text(json.dumps(receipt))
        runs.append({'slot': cold.slot_name(entry), 'execution_sha256': cold.pilot.sha(slot / 'execution.json')})
    execution = {'schema': 'glm53-p8-fc1-cold-comparison-execution.v1', 'plan_sha256': cold.pilot.sha(plan_path),
                 'protected_roles_opened': [], 'allocation_restart': False, 'exit_code': 0,
                 'completed_slots': 10, 'finished_at': 'terminal', 'runs': runs}

    def refresh(index=None):
        if index is not None:
            slot = source / cold.slot_name(cold.ORDER[index])
            receipt = json.loads((slot / 'execution.json').read_text())
            receipt['files'] = cold.file_inventory(slot)
            (slot / 'execution.json').write_text(json.dumps(receipt))
            execution['runs'][index]['execution_sha256'] = cold.pilot.sha(slot / 'execution.json')
        (source / 'execution.json').write_text(json.dumps(execution))

    refresh()
    return source, dest, plan_path, execution, state, refresh, analysis_calls


@pytest.mark.parametrize('arm', ['p8', 'exl3'])
def test_sanitized_logs_replay_both_runtime_audits_without_secrets(arm):
    text = snapshot.sanitize_log(runtime(arm)).decode()
    assert 'DO_NOT_PUBLISH' not in text and 'pid=' not in text
    assert snapshot.replay_runtime(text, arm) == cold.runtime_audit(runtime(arm), arm)
    with pytest.raises(ValueError):
        snapshot.replay_runtime(text.replace('tensor_parallel_size=4', 'tensor_parallel_size=40'), arm)


def test_ten_run_success_snapshot_preserves_metrics_hashes_not_private_contents(attempt):
    source, dest, plan, _, _, _, calls = attempt
    originals = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    report = snapshot.snapshot(plan)
    assert report['status'] == 'terminal-comparison-complete' and report['speed_gate_pass'] is True
    assert report['completed_slots'] == 10 and len(report['slots']) == 10 and len(calls) == 1
    assert report['analysis_replayed'] and report['slots'][0]['maximum_sampled_gpu_temperature_c'] == 70
    assert report['allocation_restart'] is False and report['numerical_closure_required_separately']
    manifest = json.loads((dest / 'snapshot.json').read_text())
    assert len(manifest['omitted_raw_files']) == 40
    for path in dest.rglob('*'):
        if path.is_file():
            assert b'DO_NOT_PUBLISH' not in path.read_bytes()
            assert '.private.' not in path.name
    slot = cold.slot_name(cold.ORDER[0])
    assert (dest / slot / 'result.json').read_bytes() == (source / slot / 'result.json').read_bytes()
    assert originals == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert snapshot.sha((dest / 'report.json').read_bytes()) == manifest['report_sha256']
    with pytest.raises(ValueError, match='no overwrite'):
        snapshot.snapshot(plan)


def test_root_analysis_failure_preserved_even_when_all_ten_slots_completed(attempt):
    source, _, plan, _, state, _, calls = attempt
    state.update(ActiveState='failed', Result='exit-code')
    (source / 'analysis.json').unlink()
    report = snapshot.snapshot(plan)
    assert report['status'] == 'terminal-failure-preserved' and report['completed_slots'] == 10
    assert report['speed_gate_pass'] is None and report['medians'] is None and not calls
    assert report['analysis_replayed'] is False


def test_failed_partial_slot_and_not_started_slots_never_promoted(attempt):
    source, dest, plan, execution, state, refresh, calls = attempt
    # Remove only this test's synthetic fixtures, not any campaign artifacts.
    for entry in cold.ORDER[2:]:
        slot = source / cold.slot_name(entry)
        for path in slot.iterdir():
            path.unlink()
        slot.rmdir()
    execution.update(exit_code=1, completed_slots=1, runs=execution['runs'][:2])
    state.update(ActiveState='failed', Result='exit-code')
    slot = source / cold.slot_name(cold.ORDER[1])
    receipt = json.loads((slot / 'execution.json').read_text())
    receipt['exit_code'] = 1
    (slot / 'execution.json').write_text(json.dumps(receipt))
    (slot / 'result.json').write_text('{"incomplete":')
    for name in ('summary.json', 'runtime-audit.json'):
        (slot / name).unlink()
    refresh(1)
    report = snapshot.snapshot(plan)
    assert report['attempted_slots'] == 2 and report['completed_slots'] == 1
    assert report['slots'][1]['status'] == 'failed-or-incomplete' and report['speed_gate_pass'] is None
    assert not calls and not (dest / cold.slot_name(cold.ORDER[1]) / 'result.json').exists()
    manifest = json.loads((dest / 'snapshot.json').read_text())
    assert any(row['reason'].startswith('incomplete result') for row in manifest['omitted_raw_files'])
    assert not (dest / 'analysis.json').exists()
    assert (dest / 'raw-unverified-analysis.json').read_bytes() == (source / 'analysis.json').read_bytes()
    raw_analysis = next(row for row in manifest['files'] if row['path'] == 'raw-unverified-analysis.json')
    assert 'UNVERIFIED' in raw_analysis['transformation'] and 'excluded' in raw_analysis['transformation']


def test_receipted_artifact_tamper_fails_before_destination_creation(attempt):
    source, dest, plan, _, _, _, _ = attempt
    (source / cold.slot_name(cold.ORDER[0]) / 'thermal.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='hash differs'):
        snapshot.snapshot(plan)
    assert not dest.exists()


def test_one_interrupted_unreceipted_slot_is_preserved_but_never_counted(attempt):
    source, dest, plan, execution, state, refresh, calls = attempt
    for entry in cold.ORDER[1:]:
        slot = source / cold.slot_name(entry)
        for path in slot.iterdir():
            path.unlink()
        slot.rmdir()
    slot = source / cold.slot_name(cold.ORDER[0])
    (slot / 'execution.json').unlink()
    execution.update(exit_code=1, completed_slots=0, runs=[])
    state.update(ActiveState='failed', Result='signal')
    refresh()
    report = snapshot.snapshot(plan)
    assert report['attempted_slots'] == 1 and report['root_receipted_slots'] == 0
    assert report['slots'][0]['status'] == 'unreceipted-interrupted-slot'
    assert report['speed_gate_pass'] is None and not calls
    assert (dest / cold.slot_name(cold.ORDER[0]) / 'result.json').is_file()


def test_env_in_full_benchmark_requires_explicit_privacy_review(attempt):
    source, dest, plan, _, _, refresh, _ = attempt
    slot = source / cold.slot_name(cold.ORDER[0])
    result = json.loads((slot / 'result.json').read_text())
    result['startup_diagnostics']['env'] = {'VLLM_API_KEY': 'DO_NOT_PUBLISH'}
    (slot / 'result.json').write_text(json.dumps(result))
    refresh(0)
    with pytest.raises(ValueError, match='privacy review'):
        snapshot.snapshot(plan)
    assert not dest.exists()


def test_terminal_guard_rejects_running_unit(monkeypatch):
    monkeypatch.setattr(snapshot.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, 'ActiveState=active\nResult=success\n', ''))
    with pytest.raises(ValueError, match='terminal'):
        snapshot.terminal_state()


def test_offline_snapshot_auth_does_not_requalify_mutable_weights(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('# frozen\n')
    benchmark = tmp_path / 'benchmark.py'
    benchmark.write_text('# frozen benchmark\n')
    monkeypatch.setattr(cold, 'REPO', tmp_path)
    monkeypatch.setattr(cold, 'SOURCES', {'source.py'})
    monkeypatch.setattr(cold, 'PINNED', {})
    monkeypatch.setattr(cold, 'FIXED', {'schema': 'test'})
    monkeypatch.setattr(cold.pilot, 'BENCH', benchmark)
    monkeypatch.setattr(cold.weights, 'verify_stats', lambda *args: pytest.fail('must not read mutable weight stats'))
    plan = {'schema': 'test', 'prerequisite_sha256': {}, 'source_sha256': {'source.py': cold.pilot.sha(source)},
            'benchmark_sha256': cold.pilot.sha(benchmark)}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    path.with_suffix('.sha256').write_text(cold.pilot.sha(path))
    assert snapshot.verify_plan_for_snapshot(path) == plan
    source.write_text('# changed\n')
    with pytest.raises(ValueError, match='frozen comparison source'):
        snapshot.verify_plan_for_snapshot(path)
