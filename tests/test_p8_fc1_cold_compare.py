import copy
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from glm53_nvfp4 import p8_fc1_cold_compare as cold

REPO = Path(__file__).resolve().parents[1]


def recipe(arm):
    # Public, synthetic recipe fields; production environment dumps stay raw.
    command = ['exec', '/opt/venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', '/model',
               '--port', '8017', '--served-model-name', 'source-name', '--tensor-parallel-size', '4',
               '--decode-context-parallel-size', str(cold.TOPOLOGY[arm]['dcp']),
               '--attention-backend', 'B12X_MLA_SPARSE', '--kv-cache-dtype', 'nvfp4_ds_mla',
               '--max-num-seqs', '1', '--max-num-batched-tokens', '2048',
               '--quantization', 'modelopt' if arm == 'p8' else 'exl3']
    env = ['CUDA_VISIBLE_DEVICES=0,1,2,3', 'OPAQUE_CONFIGURATION=preserve-exactly']
    if arm == 'p8':
        env += ['GLM53_P8_NATIVE=1', 'GLM53_P8_SMALL_M=1', 'GLM53_P8_FC1_TILE_N=64', 'GLM53_P8_FUSED_SCRATCH=1']
    else:
        command += ['--enable-expert-parallel']
    return {'Image': cold.IMAGES[arm], 'Config': {'Entrypoint': ['/bin/bash'], 'WorkingDir': '/',
            'Cmd': ['-lc', shlex.join(command)], 'Env': env},
            'HostConfig': {'NetworkMode': 'host', 'IpcMode': 'host', 'Privileged': False,
                          'SecurityOpt': ['label=disable'], 'ShmSize': 64 * 2**30, 'Runtime': 'runc',
                          'Binds': [f'/cache-{arm}:/cache:rw', f'/model-{arm}:/model:ro']}}


def result(arm):
    path = (REPO / 'evidence/opened/codec-v2/p8-fc1-integrated-v1/result.json' if arm == 'p8' else
            REPO / 'evidence/opened/codec-v2/p8-speed-attempts/speed-v2a3/round-01-exl3/benchmark.json')
    value = json.loads(path.read_text())
    value['metadata'].update(model=cold.model_name(arm), server=f'127.0.0.1:{cold.PORT}', context_lengths=[32768])
    value['results'] = [row for row in value['results'] if row['context_tokens'] == 32768]
    value['prefill'] = {'32768': value['prefill']['32768']}
    return value


def runtime(arm):
    common = ['tensor_parallel_size=4', f'decode_context_parallel_size={cold.TOPOLOGY[arm]["dcp"]}',
              'speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla', 'enforce_eager=False',
              'Breakable CUDA graph enabled', 'Capturing CUDA graphs (FULL): 100%',
              *[f'(Worker_TP{rank}) Graph capturing finished' for rank in range(4)]]
    if arm == 'exl3':
        common += ["'enable_expert_parallel': True", 'EXL3 full-expert EP runtime planned',
                   'quantization=exl3', 'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts']
    else:
        for layer in range(3, 45):
            for rank in range(4):
                common += [f'GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} small_m_scheduler=true',
                           f'GLM53_P8_M1_DISPATCH layer={layer} rank={rank} fc1_tile_n=64 fused_scratch_zero=true']
    return '\n'.join(common)


@pytest.mark.parametrize('arm', ['p8', 'exl3'])
def test_clone_preserves_environment_mounts_and_executable(tmp_path, arm):
    source = recipe(arm)
    argv = cold.clone_argv(source, arm, {'round': 1, 'arm': arm}, tmp_path, 'owner')
    original, actual = shlex.split(source['Config']['Cmd'][1]), shlex.split(argv[-1])
    for option in ('--port', '--served-model-name'):
        actual[actual.index(option) + 1] = original[original.index(option) + 1]
    assert actual == original
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == '--env'] == source['Config']['Env']
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == '--volume'] == source['HostConfig']['Binds']
    assert argv[-3] == cold.IMAGES[arm]
    assert cold.benchmark_argv(tmp_path, arm)[cold.benchmark_argv(tmp_path, arm).index('--contexts') + 1] == '32k'


@pytest.mark.parametrize('arm', ['p8', 'exl3'])
def test_primary_prefill_is_server_validation_not_client(arm):
    value = result(arm)
    summary = cold.summarize(value, arm)
    assert summary['prefill_server_tokens_per_second'] == value['prefill']['32768']['server_validation']['tok_per_sec']
    assert not summary['allocation_restart'] and summary['speed_only']
    assert cold.runtime_audit(runtime(arm), arm)['graph']['full_capture_complete']


@pytest.mark.parametrize('key,value', [('method', 'client'), ('invalid_reason', 'contaminated'),
                                    ('cached_tokens', 1), ('token_source', 'request'), ('samples', 0),
                                    ('samples', 1.5), ('prompt_tokens', 100), ('request_prompt_tokens', 32000),
                                    ('tok_per_sec', float('nan')), ('prefill_seconds', 0)])
def test_prefill_invalid_server_measurement_never_falls_back(key, value):
    data = result('p8')
    data['prefill']['32768']['server_validation'][key] = value
    with pytest.raises(ValueError):
        cold.summarize(data, 'p8')


def test_recipe_and_graph_drift_rejected():
    source = recipe('p8')
    source['Config']['Cmd'][1] += ' --enable-expert-parallel'
    with pytest.raises(ValueError):
        cold.validate_recipe(source, 'p8')
    for arm in ('p8', 'exl3'):
        with pytest.raises(ValueError):
            cold.runtime_audit(runtime(arm).replace('100%', '0%'), arm)
    with pytest.raises(ValueError):
        cold.runtime_audit(runtime('p8').replace('fc1_tile_n=64', 'fc1_tile_n=128', 1), 'p8')


def test_fixed_order_five_new_runs_per_arm():
    assert cold.ORDER == [{'round': r, 'arm': arm} for r in range(1, 6)
                          for arm in (('exl3', 'p8') if r % 2 else ('p8', 'exl3'))]
    assert len({cold.slot_name(e) for e in cold.ORDER}) == 10
    assert sum(e['arm'] == 'p8' for e in cold.ORDER) == 5
    assert cold.FIXED['allocation_restart'] is False


def test_plan_generator_seals_protocol_sources_and_requires_fresh_paths(tmp_path, monkeypatch):
    repo, root = tmp_path / 'repo', tmp_path / 'raw'
    repo.mkdir(); root.mkdir()
    source = repo / 'source.py'
    source.write_text('# frozen source\n')
    benchmark = tmp_path / 'benchmark.py'
    benchmark.write_text('# benchmark\n')
    audit = tmp_path / 'weight-audit.json'
    audit.write_text('{}')
    monkeypatch.setattr(cold, 'REPO', repo)
    monkeypatch.setattr(cold, 'ROOT', root)
    monkeypatch.setattr(cold, 'PINNED', {})
    monkeypatch.setattr(cold, 'SOURCES', {'source.py'})
    monkeypatch.setattr(cold.pilot, 'BENCH', benchmark)
    monkeypatch.setattr(cold, 'recipes', lambda: ({arm: recipe(arm) for arm in ('p8', 'exl3')}, {}))
    monkeypatch.setattr(cold, 'weight_receipt', lambda *args: {})
    path, out = tmp_path / 'plan.json', root / 'comparison-v1'
    plan = cold.make_plan(path, out, audit)
    assert cold.authenticate(path)[0] == plan and plan['order'] == cold.ORDER
    assert plan['weight_audit_sha256'] == cold.pilot.sha(audit)
    assert not out.exists()  # Plan creation does not start execution.
    with pytest.raises(ValueError, match='fresh'):
        cold.make_plan(path, out, audit)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='seal'):
        cold.authenticate(path)
    path.with_suffix('.sha256').write_text(cold.pilot.sha(path))
    source.write_text('# modified source\n')
    with pytest.raises(ValueError, match='source identity'):
        cold.authenticate(path)


def test_weight_audit_binds_exact_expected_hashes(tmp_path, monkeypatch):
    audit = tmp_path / 'audit.json'
    rows = [{'path': f'/payload/{arm}/{i}', 'arm': arm, 'sha256': 'a' * 64, 'expected_sha256': 'a' * 64}
            for arm, count in (('p8', 168), ('exl3', 120)) for i in range(count)]
    receipt = {'protected_roles_opened': [], 'files': rows,
               'manifests': {str(cold.weights.P8_MANIFEST): cold.weights.P8_SHA,
                             str(cold.weights.EXL3_RECEIPT): cold.weights.EXL3_SHA}}
    audit.write_text(json.dumps(receipt))
    monkeypatch.setattr(cold.weights, 'verify_stats', lambda value: None)
    assert cold.weight_receipt(audit, cold.pilot.sha(audit)) == receipt
    receipt['files'][0]['sha256'] = 'b' * 64
    audit.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='completeness'):
        cold.weight_receipt(audit)


def test_live_or_unknown_container_withholds_both_production_starts():
    calls = []
    result = cold.restore_if_safe({'backend': True, 'timer': True}, {'ok': False},
                                 lambda prior: calls.append(prior), lambda unit, user: False)
    assert not calls and result['withheld'] and result['errors']
    assert result['backend'] is False and result['timer'] is False


@pytest.mark.parametrize('running', [True, False])
def test_restoration_safety_checks_owned_identity_and_state(running):
    entry = cold.ORDER[0]
    cid = 'a' * 64
    container = {'Id': cid, 'Name': '/' + cold.PREFIX + '-' + cold.slot_name(entry),
                 'Image': cold.IMAGES[entry['arm']], 'Config': {'Labels': {cold.LABEL: 'plan:' + cold.slot_name(entry)}},
                 'State': {'Running': running}}
    def command(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, cid + '\n' if argv[1] == 'ps' else json.dumps([container]), '')
    assert cold.restoration_safety('plan', command)['ok'] is (not running)
    container['Config']['Labels'][cold.LABEL] = 'foreign'
    assert not cold.restoration_safety('plan', command)['ok']


def test_absence_allows_restore_even_when_log_evidence_previously_failed():
    def command(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, '', '')
    safety = cold.restoration_safety('plan', command)
    assert safety['ok']
    restored = cold.restore_if_safe({'backend': True, 'timer': False}, safety,
                                    lambda prior: {**prior, 'errors': []})
    assert restored == {'backend': True, 'timer': False, 'errors': []}


def test_cleanup_stop_timeout_kills_then_removes(tmp_path):
    entry = cold.ORDER[0]
    name, cid = cold.PREFIX + '-' + cold.slot_name(entry), 'a' * 64
    container = {'Id': cid, 'Name': '/' + name, 'Image': cold.IMAGES['exl3'],
                 'Config': {'Labels': {cold.LABEL: 'owner'}}, 'State': {'Running': True}}
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[1] == 'stop':
            raise subprocess.TimeoutExpired(argv, 45)
        if argv[1] == 'kill':
            container['State']['Running'] = False
        return subprocess.CompletedProcess(argv, 0, json.dumps([container]) if argv[1] == 'inspect' else '', '')
    ok, errors = cold.cleanup_owned(tmp_path, name, cold.IMAGES['exl3'], 'owner', command)
    assert not ok and errors and any(argv[1] == 'kill' for argv in calls) and calls[-1][1] == 'rm'


@pytest.fixture
def complete_attempt(tmp_path, monkeypatch):
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    plan_hash = cold.pilot.sha(plan_path)
    plan = {'limits': ['test-only']}
    monkeypatch.setattr(cold, 'authenticate', lambda path: (plan, {}))
    out = tmp_path / 'attempt'
    out.mkdir()
    runs = []
    for index, entry in enumerate(cold.ORDER):
        slot = out / cold.slot_name(entry)
        slot.mkdir()
        data = result(entry['arm'])
        # Ensure a deterministic strict double win independent of historical noise.
        data['results'][0]['aggregate_tps'] = 100 if entry['arm'] == 'p8' else 90
        data['prefill']['32768']['server_validation']['tok_per_sec'] = 8000 if entry['arm'] == 'p8' else 7000
        for name, value in {'result.json': data, 'summary.json': cold.summarize(data, entry['arm']),
                            'runtime-audit.json': cold.runtime_audit(runtime(entry['arm']), entry['arm'])}.items():
            (slot / name).write_text(json.dumps(value))
        (slot / 'server-final.private.log').write_text(runtime(entry['arm']))
        (slot / 'cooldown.jsonl').write_text('{"temperatures":[75,70,60,60]}\n')
        (slot / 'thermal.jsonl').write_text('{"temperatures":[80,70,60,60]}\n')
        cid = f'{index + 1:064x}'
        (slot / 'container-final.private.json').write_text(json.dumps({
            'Id': cid, 'Image': cold.IMAGES[entry['arm']],
            'Created': f'2026-09-05T00:00:{index * 2:02d}+00:00',
            'State': {'Running': False, 'StartedAt': f'2026-09-05T00:00:{index * 2:02d}.100000000Z',
                      'FinishedAt': f'2026-09-05T00:00:{index * 2:02d}.900000000Z'},
            'Config': {'Labels': {cold.LABEL: plan_hash + ':' + cold.slot_name(entry)}}}))
        receipt = {**entry, 'exit_code': 0, 'cleanup': {'ok': True}, 'plan_sha256': plan_hash,
                   'image': cold.IMAGES[entry['arm']], 'topology': cold.TOPOLOGY[entry['arm']],
                   'protected_roles_opened': [], 'container_id': cid,
                   'started_at': f'2026-09-05T00:00:{index * 2:02d}+00:00',
                   'finished_at': f'2026-09-05T00:00:{index * 2 + 1:02d}+00:00',
                   'files': cold.file_inventory(slot)}
        (slot / 'execution.json').write_text(json.dumps(receipt))
        runs.append({'slot': cold.slot_name(entry), 'execution_sha256': cold.pilot.sha(slot / 'execution.json')})
    execution = {'exit_code': 0, 'completed_slots': 10, 'plan_sha256': plan_hash, 'runs': runs,
                 'protected_roles_opened': [], 'allocation_restart': False,
                 'restoration_safety': {'ok': True},
                 'prior': {'backend': True, 'timer': False},
                 'restoration': {'backend': True, 'timer': False, 'errors': []}}
    (out / 'execution.json').write_text(json.dumps(execution))

    def reseal(index):
        slot = out / cold.slot_name(cold.ORDER[index])
        receipt = json.loads((slot / 'execution.json').read_text())
        receipt['files'] = cold.file_inventory(slot)
        (slot / 'execution.json').write_text(json.dumps(receipt))
        execution['runs'][index]['execution_sha256'] = cold.pilot.sha(slot / 'execution.json')
        (out / 'execution.json').write_text(json.dumps(execution))

    return out, plan_path, execution, reseal


def test_analysis_ten_processes_strict_medians_and_speed_only(complete_attempt):
    out, plan, _, _ = complete_attempt
    analysis = cold.analyze(out, plan)
    assert analysis['speed_gate_pass'] and len(analysis['rows']) == 10
    assert analysis['medians']['p8']['decode_tokens_per_second'] == 100
    assert analysis['allocation_restart'] is False and analysis['numerical_closure_required_separately']


def test_analysis_rejects_top_level_receipt_mismatch(complete_attempt):
    out, plan, execution, _ = complete_attempt
    execution['runs'][0]['execution_sha256'] = '0' * 64
    (out / 'execution.json').write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='execution identity'):
        cold.analyze(out, plan)


def test_primary_tie_never_qualifies_or_unblocks_allocation(complete_attempt):
    out, plan, _, reseal = complete_attempt
    for index, entry in enumerate(cold.ORDER):
        if entry['arm'] != 'p8':
            continue
        slot = out / cold.slot_name(entry)
        data = json.loads((slot / 'result.json').read_text())
        data['prefill']['32768']['server_validation']['tok_per_sec'] = 7000
        (slot / 'result.json').write_text(json.dumps(data))
        (slot / 'summary.json').write_text(json.dumps(cold.summarize(data, 'p8')))
        reseal(index)
    analysis = cold.analyze(out, plan)
    assert not analysis['speed_gate_pass'] and not analysis['allocation_restart']


@pytest.mark.parametrize('change', ['overlap', 'reused-id', 'hot', 'wrong-dispatch'])
def test_analysis_rejects_invalid_independent_runs(complete_attempt, change):
    out, plan, _, reseal = complete_attempt
    slot = out / cold.slot_name(cold.ORDER[1])
    receipt = json.loads((slot / 'execution.json').read_text())
    if change == 'overlap':
        receipt['started_at'] = '2026-09-05T00:00:00+00:00'
    elif change == 'reused-id':
        receipt['container_id'] = f'{1:064x}'
    elif change == 'hot':
        (slot / 'thermal.jsonl').write_text('{"temperatures":[90,70,60,60]}\n')
    else:
        log = (slot / 'server-final.private.log').read_text().replace('fc1_tile_n=64', 'fc1_tile_n=128', 1)
        (slot / 'server-final.private.log').write_text(log)
    (slot / 'execution.json').write_text(json.dumps(receipt))
    reseal(1)
    with pytest.raises(ValueError):
        cold.analyze(out, plan)
