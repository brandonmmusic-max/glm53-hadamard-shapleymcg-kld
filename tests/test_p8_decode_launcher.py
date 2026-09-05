import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

import numpy as np
import pytest

from glm53_nvfp4 import p8_decode_launcher as launcher

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:071da1f9e9b24709e59a0959b97a8d524e82fca2fbf4de9a5dc2113a938c6b60'


def source_recipe():
    return json.loads((REPO / 'evidence/opened/codec-v2/p8-smallm-integrated-v1/container.json').read_text())[0]


def pilot_recipe():
    source = source_recipe()
    source['Image'] = launcher.cold.IMAGES['p8']
    source['Config']['Env'] += ['GLM53_P8_FC1_TILE_N=64', 'GLM53_P8_FUSED_SCRATCH=1']
    return source


def log(arm, complete=False):
    cfg = launcher.ARMS[arm]
    rows = ['Using V2 Model Runner', 'tensor_parallel_size=4', 'decode_context_parallel_size=1',
            'speculative_config=None', 'kv_cache_dtype=nvfp4_ds_mla', 'enforce_eager=False',
            'Breakable CUDA graph enabled', 'Capturing CUDA graphs (FULL): 100%']
    for rank in range(4):
        rows += [f'(Worker_TP{rank}) Graph capturing finished',
                 f'GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank={rank} max_num_reqs=1 real_vocab=154880 expected_outputs=2047',
                 f'GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank={rank} registrations=1 samples=2']
        for layer in range(3, 45):
            rows += [f'GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} small_m_scheduler=true',
                     f'GLM53_P8_M1_DISPATCH layer={layer} rank={rank} fc1_tile_n={cfg["tile_n"]} '
                     f'fused_scratch_zero={str(cfg["fused_scratch_zero"]).lower()}']
    if complete:
        rows += ['GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=conditional-fit-0000 rows=2047 tp_rank=0']
    return '\n'.join(rows)


@pytest.mark.parametrize('arm', ['n128', 'n64'])
def test_same_capture_image_v2_env_no_v1_processor_explicit_maxseqs(tmp_path, arm):
    source = pilot_recipe()
    windows = [{'id': 'conditional-fit-0000'}]
    argv = launcher.clone_argv(source, IMAGE, {'stage': 'canary', 'arm': arm}, tmp_path, windows, 'owner')
    env = dict(argv[i + 1].split('=', 1) for i, arg in enumerate(argv) if arg == '--env')
    assert env['VLLM_USE_V2_MODEL_RUNNER'] == env['GLM53_P8_DECODE_CAPTURE_V2'] == '1'
    assert env['GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS'] == 'conditional-fit-0000'
    assert env['GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS'] == '2047'
    assert env['GLM53_P8_FC1_TILE_N'] == str(launcher.ARMS[arm]['tile_n'])
    assert env['GLM53_P8_FUSED_SCRATCH'] == ('1' if arm == 'n64' else '')
    assert 'GLM53_P8_DECODE_CAPTURE_ALLOW_NONZERO_START' not in env
    assert argv[-3] == IMAGE and '--logits-processors' not in argv[-1]
    tokens = shlex.split(argv[-1])
    assert tokens[tokens.index('--max-num-seqs') + 1] == '1'
    assert tokens[tokens.index('--decode-context-parallel-size') + 1] == '1'
    for bind in source['HostConfig']['Binds']:
        expected = f'{REPO / "runtime_patch"}:/runtime-patch:ro' if bind == launcher.PRIOR_RUNTIME_MOUNT else bind
        assert expected in argv
    assert f'{tmp_path / "captures"}:/p8-captures:rw' in argv


@pytest.mark.parametrize('arm', ['n128', 'n64'])
def test_runtime_four_v2_ready_and_actual_168_m1_receipts(arm):
    assert launcher.runtime_audit(log(arm), arm)['m1_pairs'] == 168
    assert launcher.runtime_audit(log(arm, True), arm, [{'id': 'conditional-fit-0000'}])['complete_window_ids'] == ['conditional-fit-0000']
    for invalid in (log(arm).replace('real_vocab=154880', 'real_vocab=154879', 1),
                    log(arm).replace('Using V2 Model Runner', 'Using V1 Model Runner'),
                    log(arm).replace('expected_outputs=2047', 'expected_outputs=32', 1),
                    log(arm).replace('100%', '0%'),
                    log(arm) + '\nGLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=0 max_num_reqs=1 real_vocab=154880 expected_outputs=2047'):
        with pytest.raises(ValueError):
            launcher.runtime_audit(invalid, arm)
    with pytest.raises(ValueError):
        launcher.runtime_audit(log(arm), arm, [{'id': 'conditional-fit-0000'}])


def test_authenticated_built_image_receipt_cpu_only(tmp_path):
    builder = launcher.builder
    sources = {name: launcher.pilot.sha(builder.CONTEXT / name) for name in builder.SOURCE_NAMES}
    installed = {p: builder.SAMPLER_PATCHED_SHA for p in builder.SAMPLERS}
    installed.update({p: builder.WARMUP_PATCHED_SHA for p in builder.WARMUPS})
    installed.update({builder.PACKAGE + n: h for n, h in sources.items() if n.endswith('.py')})
    installed[builder.INHERITED_FC2] = builder.INHERITED_FC2_SHA
    value = {'schema': 'glm53-p8.decode-capture-image.v2', 'status': 'complete',
             'image_id': IMAGE, 'parent_image_id': launcher.cold.IMAGES['p8'],
             'gpu_used': False, 'speed_measurement_valid': False,
             'builder_sha256': launcher.pilot.sha(REPO / 'scripts/build_p8_decode_capture_image.py'),
             'source_sha256': sources, 'image_source_sha256': installed}
    path = tmp_path / 'image.json'
    path.write_text(json.dumps(value))
    receipt = launcher.verify_image_receipt(path, IMAGE)
    assert receipt['image_id'] == IMAGE and receipt['parent_image_id'] == launcher.cold.IMAGES['p8']
    with pytest.raises(ValueError):
        launcher.verify_image_receipt(path, launcher.cold.IMAGES['p8'])
    with pytest.raises(ValueError):
        launcher.verify_image_receipt(REPO / 'evidence/opened/codec-v2/p8-decode-capture-image-v1/receipt.json', IMAGE)
    for invalid in (builder.WARMUP_ORIGINAL_SHA, 'b' * 64):
        value['image_source_sha256'][builder.WARMUPS[0]] = invalid
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            launcher.verify_image_receipt(path, IMAGE)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_rank', 'wrong_count'])
def test_warmup_closure_required_before_eval_and_in_final_audit(mutation):
    marker = 'GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=0 registrations=1 samples=2'
    source = log('n64', True)
    changed = {'missing': '', 'duplicate': marker + '\n' + marker,
               'wrong_rank': marker.replace('tp_rank=0', 'tp_rank=4'),
               'wrong_count': marker.replace('samples=2', 'samples=3')}[mutation]
    source = source.replace(marker, changed)
    for windows in (None, [{'id': 'conditional-fit-0000'}]):
        with pytest.raises(ValueError, match='warmup scopes'):
            launcher.runtime_audit(source, 'n64', windows)


def test_missing_or_duplicate_runtime_mount_rejected(tmp_path):
    source = pilot_recipe()
    source['HostConfig']['Binds'] = [b for b in source['HostConfig']['Binds'] if b != launcher.PRIOR_RUNTIME_MOUNT]
    with pytest.raises(ValueError, match='runtime mount'):
        launcher.clone_argv(source, IMAGE, {'stage': 'canary', 'arm': 'n64'}, tmp_path, [], 'owner')
    source['HostConfig']['Binds'] += [launcher.PRIOR_RUNTIME_MOUNT] * 2
    with pytest.raises(ValueError, match='runtime mount'):
        launcher.clone_argv(source, IMAGE, {'stage': 'canary', 'arm': 'n64'}, tmp_path, [], 'owner')


def test_request_loop_runs_thermal_ticks_and_preserves_exact_response(monkeypatch):
    tokens = np.arange(2048, dtype=np.int64)
    request = launcher.protocol.completion_request(tokens, 'conditional-fit-0000', 'test-model')
    response = {'model': 'test-model', 'choices': [{'token_ids': tokens[1:].tolist(), 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 2047}}
    seen, ticks = [], []
    def urlopen(query, timeout):
        seen.append(json.loads(query.data))
        return io.StringIO(json.dumps(response))
    monkeypatch.setattr(launcher, 'urlopen', urlopen)
    actual = launcher.request_capture(request, lambda: ticks.append(1), 5)
    assert actual == response and seen == [request] and ticks
    assert launcher.protocol.verify_response(actual, request)['prediction_positions'] == 2047


def test_canary_is_first_window_full_stage_is_all32():
    windows = [{'id': f'conditional-fit-{i:04d}'} for i in range(32)]
    plan = {'windows': windows}
    assert launcher.stage_windows(plan, 'canary') == windows[:1]
    assert launcher.stage_windows(plan, 'full') == windows
    assert launcher.ORDER == [{'stage': s, 'arm': a} for s in ('canary', 'full') for a in ('n128', 'n64')]
    assert launcher.FIXED['rows_per_window'] == 2047


def test_historical_cold_prerequisite_replays_original_worktree(tmp_path, monkeypatch):
    repo = tmp_path / 'old-repo'
    repo.mkdir()
    source = repo / 'verifier.py'
    source.write_text('# immutable original verifier')
    raw = tmp_path / 'raw'
    raw.mkdir()
    plan_path = repo / 'cold.json'
    plan_path.write_text(json.dumps({'output': str(raw), 'source_sha256': {'verifier.py': launcher.pilot.sha(source)}}))
    analysis = {'speed_gate_pass': True}
    (raw / 'analysis.json').write_text(json.dumps(analysis))
    (raw / 'execution.json').write_text(json.dumps({'exit_code': 0, 'completed_slots': 10}))
    monkeypatch.setattr(launcher, 'PRIOR_REPO', repo)
    monkeypatch.setattr(launcher, 'COLD_PLAN', plan_path)
    monkeypatch.setattr(launcher, 'COLD_PLAN_SHA', launcher.pilot.sha(plan_path))
    calls = []
    def command(args):
        calls.append(args)
        stdout = 'ActiveState=inactive\nResult=success\n' if args[0] == 'systemctl' else json.dumps(analysis)
        return subprocess.CompletedProcess(args, 0, stdout, '')
    monkeypatch.setattr(launcher.pilot, 'command', command)
    assert launcher.cold_prerequisite(plan_path) == (raw, analysis)
    assert calls[-1][:4] == [launcher.sys.executable, '-I', '-B', '-c']
    assert str(repo) in calls[-1][-1]
    assert 'analyze(' in calls[-1][-1] and 'run(' not in calls[-1][-1]
    with pytest.raises(ValueError, match='path or identity'):
        launcher.cold_prerequisite(tmp_path / 'foreign.json')
    original = plan_path.read_text()
    plan_path.write_text(original + '\n')
    with pytest.raises(ValueError, match='path or identity'):
        launcher.cold_prerequisite(plan_path)
    plan_path.write_text(original)
    (raw / 'analysis.json').write_text('{}')
    with pytest.raises(ValueError, match='does not replay'):
        launcher.cold_prerequisite(plan_path)
    (raw / 'analysis.json').write_text(json.dumps(analysis))
    source.write_text('# changed verifier')
    with pytest.raises(ValueError, match='verifier source'):
        launcher.cold_prerequisite(plan_path)


def test_compare_uses_existing_protocol_and_rejects_native_dtype_mismatch(tmp_path, monkeypatch):
    tokens = tmp_path / 'tokens.npy'
    np.save(tokens, np.arange(2048, dtype=np.int64))
    plan = {'windows': [{'id': 'conditional-fit-0000', 'domain': 'axis1_general', 'token_path': str(tokens)}]}
    values = np.zeros((2, 3), dtype='<f4')
    dtypes = ['torch.bfloat16', 'torch.bfloat16']
    calls = []
    def load(root, window, token_ids):
        index = 0 if 'n128' in str(root) else 1
        return values, {'original_logit_dtype': dtypes[index], 'original_logit_width': 3}
    monkeypatch.setattr(launcher.protocol, 'load_capture', load)
    monkeypatch.setattr(launcher.protocol, 'exact_logits', lambda a, b: calls.append((a, b)) or {'exact': True})
    assert launcher.compare_stage(plan, tmp_path, 'canary')['all_exact'] and len(calls) == 1
    dtypes[1] = 'torch.float32'
    with pytest.raises(ValueError, match='representation'):
        launcher.compare_stage(plan, tmp_path, 'canary')


def test_capture_owner_normalization_is_explicit_regular_files_only(tmp_path):
    cid, image, name, owner = 'a' * 64, IMAGE, 'owned-server', 'plan:canary-n128'
    window = {'id': 'conditional-fit-0000'}
    allowed = tmp_path / 'conditional-fit-0000.logits.f32'
    allowed.write_bytes(b'unchanged')
    unrelated = tmp_path / 'not-allowlisted.logits.f32'
    unrelated.write_bytes(b'untouched')
    container = {'Id': cid, 'Name': '/' + name, 'Image': image, 'State': {'Running': True},
                 'Config': {'Labels': {launcher.cold.LABEL: owner}}}
    calls = []
    def command(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps([container]), '')
    receipt = launcher.normalize_capture_ownership(tmp_path, [window], cid, image, name, owner,
                                                   {'uid': os.getuid(), 'gid': os.getgid()}, command)
    assert len(receipt['artifacts']) == 1 and allowed.read_bytes() == b'unchanged'
    assert calls[-1] == ['docker', 'exec', '--user', '0:0', cid, 'chown', '-h',
                         f'{os.getuid()}:{os.getgid()}', '--', '/p8-captures/' + allowed.name]
    assert 'prior_stat' in receipt['artifacts'][0]
    allowed.unlink()
    allowed.symlink_to(unrelated)
    with pytest.raises(ValueError, match='symlink'):
        launcher.normalize_capture_ownership(tmp_path, [window], cid, image, name, owner,
                                             {'uid': os.getuid(), 'gid': os.getgid()}, command)


@pytest.mark.parametrize('running', [True, False])
def test_live_or_foreign_capture_container_blocks_production_restore(running):
    cid, entry = 'a' * 64, launcher.ORDER[0]
    container = {'Id': cid, 'Name': '/' + launcher.PREFIX + '-' + launcher.slot_name(entry),
                 'Image': IMAGE, 'Config': {'Labels': {launcher.cold.LABEL: 'plan:' + launcher.slot_name(entry)}},
                 'State': {'Running': running}}
    def command(argv):
        return subprocess.CompletedProcess(argv, 0, cid + '\n' if argv[1] == 'ps' else json.dumps([container]), '')
    assert launcher.restoration_safety('plan', IMAGE, command)['ok'] is (not running)
    container['Config']['Labels'][launcher.cold.LABEL] = 'foreign'
    assert not launcher.restoration_safety('plan', IMAGE, command)['ok']


def test_plan_pins_approved_image_teacher_inputs_and_seals_without_launch(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    repo = tmp_path / 'repo'
    root.mkdir(); repo.mkdir()
    (repo / 'source.py').write_text('# source\n')
    image_receipt, roles, cold_plan = [tmp_path / name for name in ('image.json', 'roles.json', 'cold.json')]
    for path in (image_receipt, roles, cold_plan):
        path.write_text('{}')
    cold_root = tmp_path / 'cold'
    cold_root.mkdir()
    for name in ('execution.json', 'analysis.json'):
        (cold_root / name).write_text('{}')
    teacher_root = tmp_path / 'teacher'
    teacher_root.mkdir()
    windows = [{'id': f'conditional-fit-{i:04d}'} for i in range(32)]
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(launcher, 'REPO', repo)
    monkeypatch.setattr(launcher, 'SOURCE_FILES', {'source.py'})
    monkeypatch.setattr(launcher.protocol, 'ROLES_SHA256', launcher.pilot.sha(roles))
    monkeypatch.setattr(launcher, 'verify_image_receipt', lambda path, image: {'image_id': image})
    monkeypatch.setattr(launcher, 'cold_prerequisite', lambda path: (cold_root, {'speed_gate_pass': False}))
    monkeypatch.setattr(launcher, 'approved_inputs', lambda roles, teacher: (windows, {}))
    path, output = tmp_path / 'plan.json', root / 'capture-v1'
    plan = launcher.make_plan(path, output, IMAGE, image_receipt, roles, teacher_root, cold_plan)
    assert launcher.authenticate(path)[0] == plan
    assert plan['capture_image'] == IMAGE and plan['canary_window_id'] == windows[0]['id']
    assert plan['cold_speed_gate_pass_observed'] is False and plan['allocation_restart'] is False
    assert plan['raw_capture_bytes'] == 66 * 2047 * 154880 * 4 and not output.exists()
    with pytest.raises(ValueError, match='fresh'):
        launcher.make_plan(path, output, IMAGE, image_receipt, roles, teacher_root, cold_plan)
    for field, value in (('raw_capture_bytes', 0), ('output', str(tmp_path / 'wrong-parent'))):
        changed = {**plan, field: value}
        path.write_text(json.dumps(changed))
        path.with_suffix('.sha256').write_text(launcher.pilot.sha(path))
        with pytest.raises(ValueError, match='size or output parent'):
            launcher.authenticate(path)
    path.write_text(json.dumps(plan))
    path.with_suffix('.sha256').write_text(launcher.pilot.sha(path))
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='seal'):
        launcher.authenticate(path)


def test_stage_boundary_rehashes_sources_and_checks_payload_stats(tmp_path, monkeypatch):
    source, cold_plan = tmp_path / 'source.py', tmp_path / 'cold.json'
    source.write_text('# original')
    cold_plan.write_text(json.dumps({'weight_audit': str(tmp_path / 'weights.json'), 'weight_audit_sha256': 'pinned'}))
    plan = {'source_sha256': {'source.py': launcher.pilot.sha(source)},
            'cold_plan': str(cold_plan), 'cold_plan_sha256': launcher.pilot.sha(cold_plan), 'input_stats': {}}
    monkeypatch.setattr(launcher, 'REPO', tmp_path)
    calls = []
    monkeypatch.setattr(launcher.cold, 'weight_receipt', lambda path, digest: calls.append((path, digest)))
    launcher.verify_stage_identities(plan)
    assert calls == [(tmp_path / 'weights.json', 'pinned')]
    source.write_text('# changed')
    with pytest.raises(ValueError, match='source identity'):
        launcher.verify_stage_identities(plan)
    source.write_text('# original')
    def drift(path, digest):
        raise ValueError('payload stat changed')
    monkeypatch.setattr(launcher.cold, 'weight_receipt', drift)
    with pytest.raises(ValueError, match='payload stat'):
        launcher.verify_stage_identities(plan)


def test_interrupts_are_ignored_during_stage_cleanup_and_handlers_restored(tmp_path, monkeypatch):
    saved = {s: signal.getsignal(s) for s in launcher.INTERRUPT_SIGNALS}
    checked = []
    def identities(plan):
        raise ValueError('preflight drift; must not create a container')
    def cleanup(*args):
        for signum in launcher.INTERRUPT_SIGNALS:
            assert signal.getsignal(signum) == signal.SIG_IGN
            os.kill(os.getpid(), signum)
        checked.append(True)
        return True, []
    monkeypatch.setattr(launcher, 'verify_stage_identities', identities)
    monkeypatch.setattr(launcher.cold, 'cleanup_owned', cleanup)
    plan = {'windows': [{'id': 'conditional-fit-0000'}], 'capture_image': IMAGE}
    out = tmp_path / 'slot'
    result = launcher.capture_stage(plan, launcher.ORDER[0], {}, out, 'seal', [])
    assert checked == [True] and result['exit_code'] == 1 and result['cleanup']['ok']
    assert (out / 'execution.json').is_file()
    assert {s: signal.getsignal(s) for s in launcher.INTERRUPT_SIGNALS} == saved


def test_canary_failure_stops_full_stage_and_restore_ignores_second_interrupt(tmp_path, monkeypatch):
    output, plan_path = tmp_path / 'captures', tmp_path / 'plan.json'
    plan_path.write_text('{}')
    (tmp_path / 'container-final.private.json').write_text('{}')
    plan = {'output': str(output), 'capture_image': IMAGE, 'raw_capture_bytes': launcher.RAW_CAPTURE_BYTES}
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher.cold, 'PILOT', tmp_path)
    monkeypatch.setattr(launcher, 'authenticate', lambda path: (plan, {'image_source_sha256': {}}))
    monkeypatch.setattr(launcher, 'verify_stage_identities', lambda plan: None)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 10**15})())
    monkeypatch.setattr(launcher, 'open', lambda path, mode: (tmp_path / 'test.lock').open(mode), raising=False)
    monkeypatch.setattr(launcher.fcntl, 'flock', lambda *args: None)
    monkeypatch.setattr(launcher.pilot, 'active', lambda *args: False)
    monkeypatch.setattr(launcher.cold, 'inventory', lambda: [])
    def command(argv):
        text = ''
        if argv[0] == 'systemctl':
            text = 'ActiveState=inactive\nResult=success\n'
        elif argv[:3] == ['docker', 'image', 'inspect']:
            text = IMAGE
        elif 'rev-parse' in argv:
            text = 'a' * 40
        return subprocess.CompletedProcess(argv, 0, text, '')
    monkeypatch.setattr(launcher.pilot, 'command', command)
    stages, restored = [], []
    def capture(plan, entry, container, out, *args):
        stages.append(entry)
        out.mkdir()
        (out / 'execution.json').write_text('{}')
        return {'exit_code': 0}
    monkeypatch.setattr(launcher, 'capture_stage', capture)
    monkeypatch.setattr(launcher, 'compare_stage', lambda *args: {'all_exact': False})
    monkeypatch.setattr(launcher, 'restoration_safety', lambda *args: {'ok': True})
    saved = {s: signal.getsignal(s) for s in launcher.INTERRUPT_SIGNALS}
    def restore(prior, safety):
        for signum in launcher.INTERRUPT_SIGNALS:
            assert signal.getsignal(signum) == signal.SIG_IGN
            os.kill(os.getpid(), signum)
        restored.append(True)
        return {**prior, 'errors': []}
    monkeypatch.setattr(launcher.cold, 'restore_if_safe', restore)
    with pytest.raises(RuntimeError, match='partial evidence'):
        launcher.run(plan_path)
    assert stages == launcher.ORDER[:2] and restored == [True]
    receipt = json.loads((output / 'execution.json').read_text())
    assert receipt['exit_code'] == 1 and receipt['final_identity_audit']['ok']
    assert {s: signal.getsignal(s) for s in launcher.INTERRUPT_SIGNALS} == saved
