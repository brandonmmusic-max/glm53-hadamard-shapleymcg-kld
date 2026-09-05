import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

import numpy as np
import pytest

from glm53_nvfp4 import p8_index_trace_launcher as launch

REPO = Path(__file__).resolve().parents[1]
TRANSFORMATIONS = json.loads((REPO / 'evidence/opened/codec-v2/p8-index-trace-preflight-v1/cpu-interpreter.json').read_text())['source_transformations']
spec = importlib.util.spec_from_file_location('capture_base_tests', REPO / 'tests/test_p8_decode_launcher.py')
base_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_tests)


@pytest.fixture
def traces(tmp_path):
    root = tmp_path / 'index-traces'
    root.mkdir()
    for rank in range(4):
        raw = root / f'rank-{rank}-buffers.npz'
        arrays, specs = {}, {}
        for key in launch.BASE_ARRAYS:
            arrays[key] = np.zeros((9, 4), dtype=np.uint8)
            arrays[key + '__counts'] = np.ones(9, dtype=np.int32)
            arrays[key + '__errors'] = np.zeros(9, dtype=np.int32)
        np.savez(raw, **arrays)
        for key, array in arrays.items():
            specs[key] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
            if key in launch.BASE_ARRAYS:
                specs[key]['recorded_in_cuda_graph'] = True
        value = {'schema': 'glm53-p8.index-trace.v1', 'window_id': launch.WINDOW, 'tp_rank': rank,
                 'positions': list(range(255, 264)), 'layers': launch.LAYERS, 'raw_file': raw.name,
                 'raw_sha256': launch.pilot.sha(raw), 'capture_complete': True,
                 'speed_measurement_valid': False, 'protected_roles_opened': [], 'counts_valid': True,
                 'arrays': specs,
                 'source_transformations': TRANSFORMATIONS}
        (root / f'rank-{rank}.json').write_text(json.dumps(value))
    return root


def test_trace_requires_all_four_complete_rank_manifests(traces):
    assert [x['tp_rank'] for x in launch.trace_manifests(traces, TRANSFORMATIONS)] == [0, 1, 2, 3]
    (traces / 'rank-3.json').unlink()
    with pytest.raises(ValueError, match='eight-file'):
        launch.trace_manifests(traces, TRANSFORMATIONS)


@pytest.mark.parametrize('field,value', [('positions', [259]), ('layers', [3]), ('counts_valid', False),
                                      ('capture_complete', False), ('protected_roles_opened', ['confirmation']),
                                      ('raw_sha256', '0' * 64), ('source_transformations', {}),
                                      ('arrays', {'a': {'shape': [99], 'dtype': 'int32'}})])
def test_trace_rejects_invalid_metadata(traces, field, value):
    p = traces / 'rank-0.json'
    metadata = json.loads(p.read_text())
    metadata[field] = value
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        launch.trace_manifests(traces, TRANSFORMATIONS)


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'errors', 'not_graphed', 'source_emitted'])
def test_trace_rejects_forged_valid_flag(traces, kind):
    path = traces / 'rank-0.json'
    value = json.loads(path.read_text())
    key = sorted(launch.BASE_ARRAYS)[0]
    expected = value['source_transformations']
    raw = traces / value['raw_file']
    if kind == 'not_graphed':
        value['arrays'][key]['recorded_in_cuda_graph'] = False
    elif kind == 'source_emitted':
        value['source_transformations'] = json.loads(json.dumps(value['source_transformations']))
        value['source_transformations'][next(iter(value['source_transformations']))]['emitted_sha256'] = 'f' * 64
    else:
        with np.load(raw) as data:
            arrays = {name: data[name] for name in data.files}
        suffix = '__errors' if kind == 'errors' else '__counts'
        arrays[key + suffix][0] = 0 if kind == 'missing' else 2
        np.savez(raw, **arrays)
        value['raw_sha256'] = launch.pilot.sha(raw)
    assert value['counts_valid'] is True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        launch.trace_manifests(traces, expected)


@pytest.mark.parametrize('kind', ['extra', 'missing', 'wrong_original', 'invalid_emitted'])
def test_source_transformation_inventory_failclosed(kind):
    values = json.loads(json.dumps(TRANSFORMATIONS))
    key = next(iter(values))
    if kind == 'extra':
        values['foreign.module'] = values[key]
    elif kind == 'missing':
        values.pop(key)
    elif kind == 'wrong_original':
        values[key]['original_sha256'] = '0' * 64
    else:
        values[key]['emitted_sha256'] = 'unverified'
    with pytest.raises(ValueError):
        launch.core_transformations(values)


def test_source_origin_is_metadata_not_part_of_core_pin():
    values = json.loads(json.dumps(TRANSFORMATIONS))
    for value in values.values():
        value['origin'] = '/runtime/original.py'
    assert launch.core_transformations(values) == TRANSFORMATIONS


def test_cpu_preflight_requires_exact_image_and_no_model_or_teacher(tmp_path):
    source = REPO / 'evidence/opened/codec-v2/p8-index-trace-preflight-v1/cpu-interpreter.json'
    assert launch.preflight_receipt(source) == TRANSFORMATIONS
    value = json.loads(source.read_text())
    for key, invalid in [('image_id', 'sha256:' + '0' * 64), ('model_loaded', True), ('teacher_logits_opened', True), ('status', 'failed')]:
        p = tmp_path / 'preflight.json'
        p.write_text(json.dumps({**value, key: invalid}))
        with pytest.raises(ValueError):
            launch.preflight_receipt(p)


def test_carrier_config_nested_sparse_layers_and_hash(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    types = ['deepseek_sparse_attention' if i in launch.LAYERS else 'linear_attention' for i in range(45)]
    path.write_text(json.dumps({'text_config': {'num_hidden_layers': 45, 'layer_types': types}}))
    monkeypatch.setattr(launch, 'MODEL_CONFIG', path)
    monkeypatch.setattr(launch, 'MODEL_CONFIG_SHA', launch.pilot.sha(path))
    launch.verify_model_config()
    types[3] = 'linear_attention'
    path.write_text(json.dumps({'text_config': {'num_hidden_layers': 45, 'layer_types': types}}))
    with pytest.raises(ValueError, match='config differs'):
        launch.verify_model_config()
    monkeypatch.setattr(launch, 'MODEL_CONFIG_SHA', launch.pilot.sha(path))
    with pytest.raises(ValueError, match='sparse-layer'):
        launch.verify_model_config()


def test_one_way_diagnostic_limits_are_sealed():
    assert 'one-way positive' in launch.FIXED['observer_limit']
    assert 'not falsification' in launch.FIXED['observer_limit']
    assert 'inconclusive' in launch.FIXED['null_rule']
    assert launch.FIXED['unmodified_image_sources'] == launch.UNMODIFIED_SOURCES


def test_comparison_retains_full_logits_and_reads_only_one_approved_token_file(tmp_path, monkeypatch):
    from glm53_nvfp4 import p8_index_trace_analysis as analysis
    token_path = tmp_path / 'approved.tokens.npy'
    np.save(token_path, np.arange(2048, dtype=np.int64))
    plan = {'output': str(tmp_path), 'windows': [{'token_path': str(token_path)}], 'source_transformations': TRANSFORMATIONS}
    reads, capture_reads, trace_reads, analyses = [], [], [], []
    original_load = np.load
    def load(path, **kwargs):
        reads.append(path)
        return original_load(path, **kwargs)
    monkeypatch.setattr(launch.np, 'load', load)
    def capture(root, window, tokens):
        capture_reads.append((root, window))
        assert tokens.shape == (2048,)
        return np.zeros((2, 3), dtype='<f4'), {'original_logit_dtype': 'torch.bfloat16',
            'original_logit_width': 154880, 'sequence_token_ids_sha256': 'same-sequence'}
    monkeypatch.setattr(launch.protocol, 'load_capture', capture)
    monkeypatch.setattr(launch, 'trace_manifests', lambda root, transformations: trace_reads.append((root, transformations)))
    def exact(a, b, *, chunk_rows):
        assert chunk_rows == 8
        return {'exact': False, 'rows': 2047, 'differing_rows': [259]}
    monkeypatch.setattr(launch.protocol, 'exact_logits', exact)
    monkeypatch.setattr(analysis, 'compare', lambda root, *, transformations: analyses.append((root, transformations)) or {'diagnostic': True})
    def forbidden(*args, **kwargs):
        pytest.fail('teacher or protected role loader invoked')
    monkeypatch.setattr(launch.protocol, 'load_role_inputs', forbidden)
    monkeypatch.setattr(launch.base, 'approved_inputs', forbidden)
    result = launch.compare_repeats(plan)
    assert reads == [str(token_path)]
    assert capture_reads == [(tmp_path / launch.repeat_name(index) / 'captures', launch.WINDOW) for index in (1, 2)]
    assert len(trace_reads) == 2 and analyses == [(tmp_path, TRANSFORMATIONS)]
    assert result['same_n128_logits']['rows'] == 2047 and result['same_n128_logits']['exact'] is False
    assert result['index_trace_analysis'] == {'diagnostic': True}
    assert result['teacher_logits_opened'] is False and result['kld_measured'] is False


def test_adapter_scope_mounts_n128_and_restores_even_on_error(tmp_path):
    names = ('PREFIX', 'stage_windows', 'verify_stage_identities', 'clone_argv', 'normalize_capture_ownership', 'runtime_audit')
    before = {name: getattr(launch.base, name) for name in names}
    window = {'id': launch.WINDOW}
    with pytest.raises(RuntimeError, match='sentinel'):
        with launch.stage_adapter({'windows': [window]}, 2):
            assert launch.base.PREFIX == launch.prefix(2)
            argv = launch.base.clone_argv(base_tests.pilot_recipe(), launch.IMAGE,
                {'stage': 'canary', 'arm': 'n128'}, tmp_path, [window], 'seal:canary-n128')
            env = dict(argv[i + 1].split('=', 1) for i, token in enumerate(argv) if token == '--env')
            assert env['GLM53_P8_INDEX_TRACE'] == '1'
            assert env['GLM53_P8_INDEX_TRACE_ROOT'] == '/p8-index-traces'
            assert env['GLM53_P8_FC1_TILE_N'] == '128' and env['GLM53_P8_FUSED_SCRATCH'] == ''
            assert env['VLLM_USE_V2_MODEL_RUNNER'] == '1'
            assert f'{tmp_path / "index-traces"}:/p8-index-traces:rw' in argv
            tokens = shlex.split(argv[-1])
            assert tokens[tokens.index('--decode-context-parallel-size') + 1] == '1'
            assert '--enable-expert-parallel' not in tokens
            assert launch.base.stage_windows({'windows': [window]}, 'canary') == [window]
            with pytest.raises(ValueError, match='no full-panel'):
                launch.base.stage_windows({'windows': [window]}, 'full')
            raise RuntimeError('sentinel')
    assert {name: getattr(launch.base, name) for name in names} == before


def test_adapter_requires_all_rank_finish_markers():
    log = base_tests.log('n128', complete=True).replace('conditional-fit-0000', launch.WINDOW)
    # Current base runtime requires a lexical warmup proof on all four ranks.
    for rank in range(4):
        log += f'\nGLM53_P8_INDEX_TRACE_COMPLETE tp_rank={rank} window={launch.WINDOW} rows=9 layers=11'
    # Existing fixture supplies current warmup records; stub only the base
    # runtime validator here so this test isolates the added marker gate.
    original = launch.base.runtime_audit
    launch.base.runtime_audit = lambda *a, **k: {'base': True}
    try:
        with launch.stage_adapter({}, 1):
            assert launch.base.runtime_audit(log, 'n128', [{'id': launch.WINDOW}])['index_trace_complete_ranks'] == [0, 1, 2, 3]
            with pytest.raises(ValueError, match='four-rank'):
                launch.base.runtime_audit(log.replace('tp_rank=3 window=', 'tp_rank=9 window='), 'n128', [{'id': launch.WINDOW}])
    finally:
        launch.base.runtime_audit = original


def test_normalize_only_eight_trace_files_of_authenticated_container(traces, monkeypatch):
    cid, name, owner = 'a' * 64, 'owned', 'seal:canary-n128'
    c = {'Id': cid, 'Name': '/' + name, 'Image': launch.IMAGE, 'State': {'Running': True},
         'Config': {'Labels': {launch.cold.LABEL: owner}}}
    calls = []
    def command(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps([c]), '')
    monkeypatch.setattr(launch.pilot, 'command', command)
    proof = launch.normalize_traces(traces, cid, launch.IMAGE, name, owner, {'uid': os.getuid(), 'gid': os.getgid()}, TRANSFORMATIONS)
    assert len(proof['manifests']) == 4
    assert len(calls[-1][calls[-1].index('--') + 1:]) == 8
    c['State']['Running'] = False
    with pytest.raises(ValueError, match='container identity'):
        launch.normalize_traces(traces, cid, launch.IMAGE, name, owner, {'uid': os.getuid(), 'gid': os.getgid()}, TRANSFORMATIONS)


def test_plan_and_authentication_never_call_teacher_loader(tmp_path, monkeypatch):
    repo, root = tmp_path / 'repo', tmp_path / 'raw'
    repo.mkdir(); root.mkdir()
    (repo / 'source.py').write_text('# source')
    (repo / 'scripts').mkdir()
    (repo / 'scripts/preflight_p8_index_trace.py').write_text('# preflight')
    preflight = tmp_path / 'preflight.json'
    preflight.write_text('{}')
    monkeypatch.setattr(launch, 'REPO', repo)
    monkeypatch.setattr(launch, 'ROOT', root)
    monkeypatch.setattr(launch, 'source_files', lambda: {'source.py'})
    monkeypatch.setattr(launch, 'preflight_receipt', lambda path: TRANSFORMATIONS)
    monkeypatch.setattr(launch, 'import_preflight_receipt', lambda path, transformations: None)
    monkeypatch.setattr(launch, 'prior_unit_terminal', lambda: None)
    window = {'id': launch.WINDOW}
    old = {'image_receipt': '/receipt.json', 'image_receipt_sha256': 'a' * 64}
    monkeypatch.setattr(launch, 'historical_inputs', lambda: (window, {}, old))
    def forbidden(*args, **kwargs):
        pytest.fail('teacher or full-panel loader invoked')
    monkeypatch.setattr(launch.base, 'approved_inputs', forbidden)
    monkeypatch.setattr(launch.protocol, 'load_role_inputs', forbidden)
    path, out = tmp_path / 'plan.json', root / 'index-trace-v1'
    plan = launch.make_plan(path, out, preflight, preflight)
    assert launch.authenticate(path) == plan
    assert plan['windows'] == [window] and plan['teacher_logits_opened'] is False
    assert plan['raw_capture_bytes'] == 2 * 2047 * 154880 * 4
    for field, bad in [('repeats', [1, 2, 3]), ('raw_capture_bytes', 1), ('fc1_tile_n', 64), ('teacher_logits_opened', True)]:
        invalid = {**plan, field: bad}
        with pytest.raises(ValueError, match='fixed protocol'):
            launch.verify_identities(invalid)


@pytest.mark.parametrize('live,foreign', [(True, False), (False, True), (False, False)])
def test_restoration_never_starts_over_live_or_foreign_container(monkeypatch, live, foreign):
    cid = 'a' * 64
    c = {'Id': cid, 'Name': '/' + launch.prefix(1) + '-canary-n128', 'Image': launch.IMAGE,
         'Config': {'Labels': {launch.cold.LABEL: ('foreign' if foreign else 'seal:canary-n128')}},
         'State': {'Running': live}}
    monkeypatch.setattr(launch.pilot, 'command', lambda argv: subprocess.CompletedProcess(argv, 0,
        cid if argv[1] == 'ps' else json.dumps([c]), ''))
    assert launch.restoration_safety('seal')['ok'] is (not live and not foreign)


@pytest.mark.parametrize('failure', [False, True])
def test_two_same_n128_repeats_and_no_stop_for_logit_difference(tmp_path, monkeypatch, failure):
    path, out = tmp_path / 'plan.json', tmp_path / 'out'
    path.write_text('{}')
    plan = {'output': str(out), 'image_receipt': '/receipt.json'}
    monkeypatch.setattr(launch, 'authenticate', lambda path: plan)
    monkeypatch.setattr(launch, 'verify_identities', lambda plan: None)
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 10**15})())
    monkeypatch.setattr(launch.base, 'verify_image_receipt', lambda *a: {'image_source_sha256': {}})
    recipe = tmp_path / 'container-final.private.json'
    recipe.write_text('{}')
    monkeypatch.setattr(launch.cold, 'PILOT', tmp_path)
    monkeypatch.setattr(launch.cold, 'PINNED', {recipe: launch.pilot.sha(recipe)})
    monkeypatch.setattr(launch.cold, 'validate_recipe', lambda *a: None)
    monkeypatch.setattr(launch, 'open', lambda path, mode: (tmp_path / 'lock').open(mode), raising=False)
    monkeypatch.setattr(launch.fcntl, 'flock', lambda *a: None)
    def command(argv):
        value = launch.IMAGE if argv[:3] == ['docker', 'image', 'inspect'] else ''
        if argv[:2] == ['docker', 'run']:
            value = '\n'.join(f'{digest}  {path}' for path, digest in launch.UNMODIFIED_SOURCES.items())
        if argv[0] == 'systemctl':
            value = 'ActiveState=failed\nResult=exit-code\n'
            if 'MainPID' in argv:
                value += 'MainPID=0\n'
        return subprocess.CompletedProcess(argv, 0, value, '')
    monkeypatch.setattr(launch.pilot, 'command', command)
    monkeypatch.setattr(launch.pilot, 'active', lambda *a: False)
    monkeypatch.setattr(launch.cold, 'inventory', lambda: [])
    monkeypatch.setattr(launch, 'restoration_safety', lambda *a: {'ok': True})
    restored, captured, compared = [], [], []
    def capture(plan, entry, recipe, root, digest, hardware):
        captured.append(entry)
        root.mkdir()
        (root / 'execution.json').write_text('{}')
        return {'exit_code': int(failure), 'container_id': str(len(captured)) * 64}
    monkeypatch.setattr(launch.base, 'capture_stage', capture)
    monkeypatch.setattr(launch, 'compare_repeats', lambda plan: compared.append(True) or {'same_n128_logits': {'exact': False}})
    def restore(prior, safety):
        for sig in launch.base.INTERRUPT_SIGNALS:
            assert signal.getsignal(sig) == signal.SIG_IGN
            os.kill(os.getpid(), sig)
        restored.append(True)
        return {**prior, 'errors': []}
    monkeypatch.setattr(launch.cold, 'restore_if_safe', restore)
    if failure:
        with pytest.raises(RuntimeError, match='evidence preserved'):
            launch.run(path)
    else:
        assert launch.run(path)['diagnostic_complete'] is True
    assert captured == [{'stage': 'canary', 'arm': 'n128'}] * (1 if failure else 2)
    assert len(compared) == (0 if failure else 1) and restored == [True]


@pytest.mark.parametrize('state', ['ActiveState=active\nResult=success\nMainPID=0',
                                  'ActiveState=failed\nResult=exit-code\nMainPID=99',
                                  'ActiveState=failed\nResult=exit-code',
                                  'ActiveState=inactive\nResult=exit-code\nMainPID=0'])
def test_v1_unit_must_be_terminal_failed(monkeypatch, state):
    monkeypatch.setattr(launch.pilot, 'command', lambda argv: subprocess.CompletedProcess(argv, 0, state, ''))
    with pytest.raises(ValueError, match='terminal failed'):
        launch.prior_unit_terminal()


def test_import_preflight_requires_exact_source_and_schema(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    for name in launch.IMPORT_SOURCES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# pinned source')
    # core_transformations depends on actual patches; isolate it in this focused receipt test.
    monkeypatch.setattr(launch, 'REPO', repo)
    monkeypatch.setattr(launch, 'core_transformations', lambda value: value)
    payload = {'schema': 'glm53-p8.index-import-preflight.v2', 'status': 'passed', 'image_id': launch.IMAGE,
               'schemas_equal': True, 'annotation_mode_preserved': True, 'breakable_cudagraph': True,
               'gpu_used': False, 'model_loaded': False, 'teacher_logits_opened': False,
               'speed_measurement_valid': False,
               'source_sha256': {name: launch.pilot.sha(repo / name) for name in launch.IMPORT_SOURCES},
               'source_transformations': TRANSFORMATIONS}
    receipt = tmp_path / 'import.json'
    receipt.write_text(json.dumps(payload))
    launch.import_preflight_receipt(receipt, TRANSFORMATIONS)
    for key, value in [('schemas_equal', False), ('annotation_mode_preserved', False), ('status', 'failed'),
                       ('breakable_cudagraph', False),
                       ('gpu_used', True), ('model_loaded', True), ('teacher_logits_opened', True),
                       ('image_id', 'foreign'), ('schema', 'v1'), ('source_sha256', {}),
                       ('source_transformations', {})]:
        receipt.write_text(json.dumps({**payload, key: value}))
        with pytest.raises(ValueError):
            launch.import_preflight_receipt(receipt, TRANSFORMATIONS)
    receipt.write_text(json.dumps(payload))
    (repo / sorted(launch.IMPORT_SOURCES)[0]).write_text('# changed')
    with pytest.raises(ValueError, match='source identity'):
        launch.import_preflight_receipt(receipt, TRANSFORMATIONS)


@pytest.fixture
def prior_failure(tmp_path, monkeypatch):
    repo, raw = tmp_path / 'v1repo', tmp_path / 'raw'
    repo.mkdir(); raw.mkdir()
    stage = raw / 'repeat-01-n128'
    stage.mkdir()
    for name in ('requests', 'captures', 'index-traces'):
        (stage / name).mkdir()
    sources = {}
    for index in range(38):
        path = repo / f'source{index}.py'
        path.write_text('# original')
        sources[path.name] = launch.pilot.sha(path)
    plan_path = repo / 'plan.json'
    plan = {'schema': 'glm53-p8.index-trace-plan.v1', 'output': str(raw),
            'capture_image': launch.IMAGE, 'source_sha256': sources}
    plan_path.write_text(json.dumps(plan))
    plan_sha = launch.pilot.sha(plan_path)
    plan_path.with_suffix('.sha256').write_text(plan_sha + '  plan.json\n')
    container = {'Id': 'a' * 64, 'Image': launch.IMAGE, 'State': {'Status': 'exited', 'Running': False}}
    (stage / 'container-final.private.json').write_text(json.dumps(container))
    for index in range(9):
        (stage / f'file{index}.txt').write_text('preserved')
    files = {path.name: {'sha256': launch.pilot.sha(path), 'bytes': path.stat().st_size}
             for path in stage.iterdir() if path.is_file()}
    execution = {'exit_code': 1, 'plan_sha256': plan_sha, 'windows': [], 'files': files,
                 'cleanup': {'ok': True, 'errors': []}, 'protected_roles_opened': [], 'container_id': 'a' * 64}
    (stage / 'execution.json').write_text(json.dumps(execution))
    stage_sha = launch.pilot.sha(stage / 'execution.json')
    root = {'exit_code': 1, 'plan_sha256': plan_sha, 'repeats': [{'index': 1, 'execution_sha256': stage_sha}],
            'restoration': {'backend': True, 'timer': False, 'errors': []},
            'restoration_safety': {'ok': True, 'errors': [], 'containers': []},
            'final_identity_audit': {'ok': True}, 'teacher_logits_opened': False, 'protected_roles_opened': []}
    (raw / 'execution.json').write_text(json.dumps(root))
    for key, value in [('V1_REPO', repo), ('V1_OUTPUT', raw), ('V1_PLAN', plan_path),
                       ('V1_PLAN_SHA', plan_sha), ('V1_STAGE_SHA', stage_sha),
                       ('V1_EXEC_SHA', launch.pilot.sha(raw / 'execution.json'))]:
        monkeypatch.setattr(launch, key, value)
    return repo, raw, stage


def test_v1_failure_authentication_checks_38_original_sources_and_ten_files(prior_failure):
    launch.verify_v1_failure()
    assert launch.PREFIX == 'glm53-p8-index-trace-v2'
    assert launch.FIXED['schema'] == 'glm53-p8.index-trace-plan.v2'
    assert launch.FIXED['amends_execution_sha256'] == '43f91db06306435eaacf4e0969349983bc3ee2b5bb23f26673495b5c2a4d1024'


@pytest.mark.parametrize('kind', ['source', 'file', 'request', 'capture', 'trace', 'second', 'comparison', 'seal', 'root'])
def test_v1_failure_authentication_rejects_drift_or_any_evaluation(prior_failure, kind):
    repo, raw, stage = prior_failure
    path = {'source': repo / 'source0.py', 'file': stage / 'file0.txt',
            'request': stage / 'requests/request.json', 'capture': stage / 'captures/logits.bin',
            'trace': stage / 'index-traces/rank-0.json', 'second': raw / 'repeat-02-n128',
            'comparison': raw / 'comparison.json', 'seal': repo / 'plan.sha256',
            'root': raw / 'execution.json'}[kind]
    path.write_text('drift')
    with pytest.raises(ValueError):
        launch.verify_v1_failure()
