"""Synthetic JSON/hash fixtures only; no service, GPU, teacher, or model access."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'snapshot_full_analysis_v2',
    REPO / 'scripts/snapshot_p8_index_order_full_analysis_v2.py')
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else s.encode(value)
    path.write_bytes(data)
    return s.receipt(path)


def seal(path):
    digest = s.receipt(path)['sha256']
    path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n')
    return digest


def fingerprint(path):
    stat = path.stat()
    return {'resolved_path': str(path.resolve()), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
            'device': stat.st_dev}


def stage(slot, plan_sha, windows, root):
    arm = slot.split('-', 1)[1]
    rows, files = [], {}
    for window in windows:
        wid = window['id']
        raw = root / slot / 'captures' / f'{wid}.logits.f32'
        value = write(raw, f'identical-{wid}'.encode())
        files[f'captures/{wid}.logits.f32'] = value
        rows.append({'id': wid, 'domain': window['domain'], 'rows': 2047,
                     'raw_sha256': value['sha256']})
    return {'schema': 'glm53-p8.forced-m1-v2-stage.v1', 'stage': 'full',
            'arm': arm, 'plan_sha256': plan_sha, 'exit_code': 0,
            'cleanup': {'ok': True, 'errors': []}, 'unreadable_artifacts': [],
            'protected_roles_opened': [], 'allocation_restart': False,
            'windows': rows, 'files': files}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    repo = tmp_path / 'capture-repo'
    v2repo = tmp_path / 'analysis-repo'
    capture_root = tmp_path / 'capture-root'
    v2root = tmp_path / 'analysis-root'
    repo.mkdir(); v2repo.mkdir(); capture_root.mkdir(); v2root.mkdir()
    capture_plan_path = repo / 'experiments/capture.json'
    v2_plan_path = v2repo / 'experiments/analysis-v2.json'

    source_map = {}
    for index in range(56):
        path = repo / 'sources' / f'{index}.py'
        write(path, f'source-{index}'.encode())
        source_map[str(path.relative_to(repo))] = s.receipt(path)['sha256']
    v2_sources = {}
    for name in ('glm53_nvfp4/analysis_v2.py', 'tests/test_analysis_v2.py'):
        path = v2repo / name
        write(path, name.encode())
        v2_sources[name] = s.receipt(path)['sha256']

    domains = ('axis1_general', 'axis2_legal', 'axis3_code_agentic',
               'axis4_reasoning_termination')
    inputs, windows = {}, []
    roles = tmp_path / 'inputs/roles.json'
    write(roles, b'roles')
    inputs[str(roles)] = fingerprint(roles)
    for index in range(32):
        wid = f'conditional-fit-{index:04d}'
        token = tmp_path / 'inputs' / f'{wid}.tokens.npy'
        teacher = tmp_path / 'inputs' / f'{wid}.teacher.safetensors'
        write(token, f'token-{index}'.encode())
        write(teacher, f'teacher-{index}'.encode())
        inputs[str(token)] = fingerprint(token)
        inputs[str(teacher)] = fingerprint(teacher)
        windows.append({'id': wid, 'domain': domains[index % 4],
                        'token_path': str(token), 'input_sha256': s.receipt(token)['sha256'],
                        'teacher_path': teacher.name,
                        'teacher_sha256': s.receipt(teacher)['sha256']})
    capture_plan = {'schema': 'glm53-p8.index-order-full-plan.v1',
                    'output': str(capture_root), 'analysis_output': str(v2root),
                    'source_sha256': source_map, 'windows': windows,
                    'input_stats': inputs}
    write(capture_plan_path, capture_plan)
    capture_plan_sha = seal(capture_plan_path)

    stages = []
    for slot in s.SLOTS:
        value = stage(slot, capture_plan_sha, windows, capture_root)
        path = capture_root / slot / 'execution.json'
        write(path, value)
        stages.append({'slot': slot, 'execution_sha256': s.receipt(path)['sha256']})
    exact = {'schema': 'glm53-p8.forced-m1-v2-stage-exact.v1', 'stage': 'full',
             'all_exact': True, 'allocation_restart': False,
             'windows': [{'window_id': row['id'], 'rows': 2047,
                          'vocabulary': 154880, 'exact': True,
                          'unequal_values': 0, 'differing_rows': []}
                         for row in windows]}
    exact_path = capture_root / 'full-exact.json'
    write(exact_path, exact)
    exact_sha = s.receipt(exact_path)['sha256']
    execution = {'schema': 'glm53-p8.index-order-full-execution.v1',
                 'plan_sha256': capture_plan_sha, 'exit_code': 0,
                 'capture_protocol_complete': True, 'full_exact': True,
                 'full_exact_sha256': exact_sha, 'stages': stages,
                 'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
                 'observer_enabled': False, 'allocation_restart': False,
                 'speed_measurement_valid': False, 'final_identity_audit': {'ok': True},
                 'prior': {'backend': True, 'timer': False},
                 'restoration': {'backend': True, 'timer': False, 'errors': []},
                 'restoration_safety': {'ok': True, 'errors': [], 'containers': []}}
    execution_path = capture_root / 'execution.json'
    write(execution_path, execution)
    execution_sha = s.receipt(execution_path)['sha256']

    prerequisite = {'all_exact': True, 'capture_plan_sha256': capture_plan_sha,
                    'capture_execution_sha256': execution_sha,
                    'full_exact_sha256': exact_sha, 'source_count': 56,
                    'windows': 32, 'causal_rows': 65504}
    v2_plan = {'schema': 'glm53-p8.index-order-full-analysis-v2-plan.v1',
               'capture_plan': str(capture_plan_path),
               'capture_plan_sha256': capture_plan_sha,
               'capture_execution_sha256': execution_sha,
               'full_exact_sha256': exact_sha, 'output': str(v2root),
               'windows': 32, 'causal_rows': 65504,
               'gpu_launch_authorized': False, 'protected_roles_opened': [],
               'capture_prerequisite': prerequisite, 'source_sha256': v2_sources}
    write(v2_plan_path, v2_plan)
    v2_plan_sha = seal(v2_plan_path)

    scored = v2root / 'scored'
    contract = {'plan_sha256': capture_plan_sha,
                'source_receipt_sha256': {
                    str(capture_plan_path): capture_plan_sha,
                    str(execution_path): execution_sha, str(exact_path): exact_sha},
                'rows': 65504, 'window_count': 32, 'bootstrap_seed': 20260905,
                'bootstrap_replicates': 20000, 'torch_cpu_threads': 8}
    contract_path = scored / 'analysis-contract.json'
    write(contract_path, contract)
    records = []
    result_files = {contract_path.name: s.receipt(contract_path)['sha256']}
    for index, window in enumerate(windows):
        wid = window['id']
        raw_sha = json.loads((capture_root / 'full-n128/execution.json').read_text())[
            'windows'][index]['raw_sha256']
        value = float(index + 1) / 1000
        row = {'window_id': wid, 'domain': window['domain'], 'rows': 2047,
               'exact_logits': True,
               'candidate_metric_reused_from_bitexact_reference': True,
               'n128_mean_kld': value, 'n64_mean_kld': value,
               'n128_raw_sha256': raw_sha, 'n64_raw_sha256': raw_sha}
        path = scored / f'{wid}.json'
        write(path, row)
        result_files[path.name] = s.receipt(path)['sha256']
        score_bytes = f'scores-{wid}'.encode()
        for arm in ('n128', 'n64'):
            score = scored / f'{wid}.{arm}.npz'
            write(score, score_bytes)
            result_files[score.name] = s.receipt(score)['sha256']
        records.append(row)
    aggregate = s.metric.aggregate(records, True)
    analysis = {'schema': 'glm53-p8.index-order-full-kld.v1', 'status': 'complete',
                'plan_sha256': capture_plan_sha, 'exact_logits': True,
                'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
                'speed_measurement_valid': False, 'allocation_restart': False,
                'files': result_files, 'limits': ['conditional-fit only'], **aggregate}
    analysis_path = scored / 'analysis.json'
    write(analysis_path, analysis)
    analysis_sha = s.receipt(analysis_path)['sha256']
    outer_files = {str(path.relative_to(v2root)): s.receipt(path)
                   for path in sorted(scored.iterdir())}
    outer = {'schema': 'glm53-p8.index-order-full-analysis-v2-execution.v1',
             'plan_sha256': v2_plan_sha, 'exit_code': 0, 'windows_scored': 32,
             'capture_reused': True, 'gpu_launched': False, 'exact_logits': True,
             'protected_roles_opened': [], 'analysis_sha256': analysis_sha,
             'files': outer_files}
    outer_path = v2root / 'execution.json'
    write(outer_path, outer)
    outer_sha = s.receipt(outer_path)['sha256']

    for name, value in {
            'REPO': repo, 'CAPTURE_PLAN': capture_plan_path,
            'CAPTURE_PLAN_SHA256': capture_plan_sha, 'CAPTURE_ROOT': capture_root,
            'CAPTURE_EXECUTION_SHA256': execution_sha, 'FULL_EXACT_SHA256': exact_sha,
            'V2_REPO': v2repo, 'V2_PLAN': v2_plan_path,
            'V2_PLAN_SHA256': v2_plan_sha, 'V2_ROOT': v2root,
            'V2_EXECUTION_SHA256': outer_sha,
            'ANALYSIS_CONTRACT_SHA256': s.receipt(contract_path)['sha256'],
            'ANALYSIS_SHA256': analysis_sha} .items():
        monkeypatch.setattr(s, name, value)
    invocation = {'method': 'mocked-terminal-systemd', 'unit': s.UNIT,
                  'invocation_id': 'a' * 32, 'terminal_success': True}
    monkeypatch.setattr(s, 'successful_invocation', lambda ids, receipt: invocation)
    return {'repo': repo, 'capture_root': capture_root, 'v2root': v2root,
            'capture_plan': capture_plan_path, 'v2_plan': v2_plan_path,
            'destination': tmp_path / 'snapshot', 'windows': windows,
            'invocation': invocation}


def test_retained_systemd_success_requires_exact_terminal_state():
    state = {'LoadState': 'loaded', 'ActiveState': 'inactive', 'MainPID': '0',
             'Result': 'success', 'ExecMainStatus': '0', 'InvocationID': 'a' * 32}

    def command(argv, **kwargs):
        return '\n'.join(f'{key}={value}' for key, value in state.items())

    value = s.successful_invocation([], {'exit_code': 0, 'windows_scored': 32}, command)
    assert value['method'] == 'retained-systemd-state'
    state['MainPID'] = '1'
    with pytest.raises(ValueError, match='terminal success'):
        s.successful_invocation([], {'exit_code': 0, 'windows_scored': 32}, command)


def test_collected_systemd_unit_uses_invocation_correlated_32_progress_events():
    ids = [f'conditional-fit-{index:04d}' for index in range(32)]
    invocation = 'b' * 32
    entries = [{
        'MESSAGE': 'Started ' + s.UNIT + ' - /usr/bin/python3 -u -m '
                   'glm53_nvfp4.p8_index_order_full_analysis_v2 run --plan ' + str(s.V2_PLAN),
    }]
    entries += [{'_SYSTEMD_USER_UNIT': s.UNIT,
                 '_SYSTEMD_INVOCATION_ID': invocation,
                 'MESSAGE': json.dumps({'windows_scored': number, 'window_id': wid})}
                for number, wid in enumerate(ids, 1)]
    journal = b'\n'.join(json.dumps(row).encode() for row in entries)

    def command(argv, **kwargs):
        if argv[0] == 'systemctl':
            return ('LoadState=not-found\nActiveState=inactive\nMainPID=0\n'
                    'Result=success\nExecMainStatus=0\nInvocationID=')
        return journal

    value = s.successful_invocation(ids, {'exit_code': 0, 'windows_scored': 32}, command)
    assert value['invocation_id'] == invocation and value['progress_events'] == 32
    entries[-1]['MESSAGE'] = json.dumps({'windows_scored': 31, 'window_id': ids[-1]})
    journal = b'\n'.join(json.dumps(row).encode() for row in entries)
    with pytest.raises(ValueError, match='correlated'):
        s.successful_invocation(ids, {'exit_code': 0, 'windows_scored': 32}, command)


def test_snapshot_is_create_only_and_copies_only_sanitized_json(campaign):
    report = s.snapshot(campaign['destination'])
    destination = campaign['destination']
    assert report['status'] == 'full32-exact-and-analysis-complete'
    assert report['windows'] == 32 and report['causal_rows'] == 65504
    assert report['n128_n64_all_rows_exact'] is True
    assert report['aggregation_replayed'] is True
    assert report['paired_delta_n64_minus_n128'] == 0.0
    assert report['systemd_invocation'] == campaign['invocation']
    names = {str(path.relative_to(destination)) for path in destination.rglob('*')
             if path.is_file()}
    expected = {'capture-plan.json', 'capture-plan.sha256', 'analysis-v2-plan.json',
                'analysis-v2-plan.sha256', 'receipts/capture-execution.json',
                'receipts/full-n128-execution.json', 'receipts/full-n64-execution.json',
                'receipts/analysis-v2-execution.json', 'full-exact.json',
                'analysis-contract.json', 'analysis.json', 'report.json', 'snapshot.json'}
    expected |= {f'windows/{window["id"]}.json' for window in campaign['windows']}
    assert names == expected
    assert not any(any(marker in name for marker in ('.npz', '.f32', '.npy',
                                                      '.safetensors', '.private.'))
                   for name in names)
    manifest = json.loads((destination / 'snapshot.json').read_text())
    for name, expected_receipt in manifest['outputs'].items():
        assert s.receipt(destination / name) == expected_receipt
    with pytest.raises(ValueError, match='fresh'):
        s.snapshot(destination)


@pytest.mark.parametrize('fault', [
    'capture_source', 'v2_source', 'capture_seal', 'v2_seal', 'capture_execution',
    'stage_receipt', 'raw_logits', 'exact', 'analysis_execution', 'contract',
    'window_json', 'npz', 'analysis', 'input_stat',
])
def test_snapshot_fails_closed_before_destination_on_any_evidence_drift(
        campaign, fault):
    if fault == 'capture_source':
        next((campaign['repo'] / 'sources').iterdir()).write_text('changed')
    elif fault == 'v2_source':
        (s.V2_REPO / 'glm53_nvfp4/analysis_v2.py').write_text('changed')
    elif fault == 'capture_seal':
        s.CAPTURE_PLAN.with_suffix('.sha256').write_text('0' * 64 + ' capture.json')
    elif fault == 'v2_seal':
        s.V2_PLAN.with_suffix('.sha256').write_text('0' * 64 + ' analysis-v2.json')
    elif fault == 'capture_execution':
        s.CAPTURE_EXECUTION_SHA256 = '0' * 64
    elif fault == 'stage_receipt':
        value = json.loads((s.CAPTURE_ROOT / 'execution.json').read_text())
        value['stages'][0]['execution_sha256'] = '0' * 64
        write(s.CAPTURE_ROOT / 'execution.json', value)
        s.CAPTURE_EXECUTION_SHA256 = s.receipt(s.CAPTURE_ROOT / 'execution.json')['sha256']
    elif fault == 'raw_logits':
        path = next((s.CAPTURE_ROOT / 'full-n64/captures').iterdir())
        path.write_bytes(b'changed raw')
    elif fault == 'exact':
        s.FULL_EXACT_SHA256 = '0' * 64
    elif fault == 'analysis_execution':
        s.V2_EXECUTION_SHA256 = '0' * 64
    elif fault == 'contract':
        s.ANALYSIS_CONTRACT_SHA256 = '0' * 64
    elif fault == 'window_json':
        next((s.V2_ROOT / 'scored').glob('conditional-fit-*.json')).write_text('{}')
    elif fault == 'npz':
        next((s.V2_ROOT / 'scored').glob('*.npz')).write_bytes(b'changed score')
    elif fault == 'analysis':
        s.ANALYSIS_SHA256 = '0' * 64
    else:
        path = Path(next(iter(json.loads(s.CAPTURE_PLAN.read_text())['input_stats'])))
        path.write_bytes(path.read_bytes() + b'drift')
    with pytest.raises((ValueError, KeyError)):
        s.snapshot(campaign['destination'])
    assert not campaign['destination'].exists()


@pytest.mark.parametrize('field,value', [
    ('exit_code', 1), ('windows_scored', 31), ('capture_reused', False),
    ('gpu_launched', True), ('exact_logits', False),
    ('protected_roles_opened', ['final']),
])
def test_analysis_outer_receipt_is_strict(campaign, field, value):
    path = s.V2_ROOT / 'execution.json'
    outer = json.loads(path.read_text())
    outer[field] = value
    with pytest.raises(ValueError, match='outer execution'):
        s.validate_analysis(
            json.loads(s.V2_PLAN.read_text()), outer,
            json.loads((s.V2_ROOT / 'scored/analysis-contract.json').read_text()),
            json.loads((s.V2_ROOT / 'scored/analysis.json').read_text()),
            [window['id'] for window in campaign['windows']],
            {window['id']: json.loads((s.CAPTURE_ROOT / 'full-n128/execution.json').read_text())[
                'windows'][index]['raw_sha256']
             for index, window in enumerate(campaign['windows'])},
            s.receipt)


def test_input_stat_validation_does_not_open_teacher_or_token_payloads(campaign, monkeypatch):
    plan = json.loads(s.CAPTURE_PLAN.read_text())
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if str(path) in plan['input_stats']:
            raise AssertionError('teacher/token/role payload opened')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    assert len(s.validate_input_stats(plan)) == 65
