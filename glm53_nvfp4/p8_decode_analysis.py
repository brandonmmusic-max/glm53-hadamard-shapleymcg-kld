"""CPU-only, full-panel KLD analysis for the sealed V2 forced-M1 replay.

Exact-logit closure is the decision gate; KLD and its paired interval describe
the result without replacing a failed exact gate with a post-hoc tolerance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from . import p8_decode_protocol as protocol
from . import p8_decode_launcher as launcher
from .paired_role_analysis import bca_mean_interval
from .role_eval import _load_teacher, ALIGNMENT_BAND, METRIC_CODE_SHA256

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260905
CPU_THREADS = 8


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('invalid finite window estimands')
    if np.all(values == values[0]):
        return {'ci95_bca': None, 'ci_status': 'undefined: identical window values; no sampling interval inferred'}
    bounds = bca_mean_interval(values, values[indices].mean(axis=1))
    if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError('invalid BCa interval')
    return {'ci95_bca': list(bounds), 'ci_status': 'window bootstrap, not independent quantization repetitions'}


def aggregate(records, exact):
    if (len(records) != 32 or len({r['window_id'] for r in records}) != 32
            or Counter(r['domain'] for r in records) != Counter({d: 8 for d in protocol.DOMAINS})
            or any(r['rows'] != protocol.ROWS for r in records)):
        raise ValueError('complete balanced conditional-fit32 inventory required')
    a = np.array([r['n128_mean_kld'] for r in records], dtype=np.float64)
    b = np.array([r['n64_mean_kld'] for r in records], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite KLD')
    if exact and not np.array_equal(a, b):
        raise ValueError('bit-exact logits cannot have different KLD under the same metric')
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 32, size=(BOOTSTRAP_B, 32))
    delta = b - a
    return {
        'windows': 32, 'causal_rows': 32 * protocol.ROWS, 'window_records': records,
        'reference_n128': {'mean_kld': float(a.mean()), **interval(a, indices)},
        'candidate_n64': {'mean_kld': float(b.mean()), **interval(b, indices)},
        'paired_candidate_minus_reference': {'mean_delta_kld': float(delta.mean()),
            'relative_kld_reduction': float(1 - b.mean() / a.mean()) if a.mean() > 0 else None,
            **interval(delta, indices)},
        'bootstrap': {'unit': 'window', 'replicates': BOOTSTRAP_B, 'seed': BOOTSTRAP_SEED,
                      'paired_indices': True, 'aggregation': 'equal-window mean'},
        'decision': 'exact-closure-pass' if exact else 'exact-closure-fail',
        'kld_does_not_override_exact_failure': True,
    }


def verify_execution(plan, plan_path):
    root = Path(plan['output'])
    execution = json.loads((root / 'execution.json').read_text())
    if (execution.get('schema') != 'glm53-p8.forced-m1-v2-execution.v1'
            or execution.get('plan_sha256') != protocol.sha(plan_path)
            or execution.get('capture_image') != plan['capture_image']
            or execution.get('exit_code') != 0 or execution.get('capture_protocol_complete') is not True
            or execution.get('allocation_restart') is not False
            or execution.get('opened_roles') != ['conditional-fit']
            or execution.get('protected_roles_opened') != []
            or execution.get('restoration_safety', {}).get('ok') is not True
            or execution.get('restoration', {}).get('errors')
            or any(execution.get('restoration', {}).get(k) != v for k, v in execution['prior'].items())):
        raise ValueError('full capture execution/restoration is incomplete or unauthenticated')
    expected = [launcher.slot_name(e) for e in launcher.ORDER]
    if [entry['slot'] for entry in execution['stages']] != expected:
        raise ValueError('four fresh capture stages in frozen order required')
    ids = set()
    previous_end = None
    for slot in execution['stages']:
        folder = root / slot['slot']
        if protocol.sha(folder / 'execution.json') != slot['execution_sha256']:
            raise ValueError('capture stage receipt changed')
        stage = json.loads((folder / 'execution.json').read_text())
        stage_name, arm = slot['slot'].split('-')
        if (stage.get('exit_code') != 0 or stage.get('cleanup', {}).get('ok') is not True
                or stage.get('stage') != stage_name or stage.get('arm') != arm
                or stage.get('capture_image') != plan['capture_image']
                or stage.get('plan_sha256') != protocol.sha(plan_path)
                or stage.get('unreadable_artifacts') != []
                or stage.get('protected_roles_opened') != [] or stage.get('allocation_restart') is not False
                or not re.fullmatch(r'[0-9a-f]{64}', stage.get('container_id', ''))
                or stage['container_id'] in ids):
            raise ValueError('capture stage identity or success differs')
        ids.add(stage['container_id'])
        begin, end = (datetime.fromisoformat(stage[key]) for key in ('started_at', 'finished_at'))
        if end <= begin or (previous_end is not None and begin < previous_end):
            raise ValueError('capture stage chronology or overlap differs')
        previous_end = end
        windows = launcher.stage_windows(plan, stage_name)
        mandatory = {'cooldown.jsonl', 'thermal.jsonl', 'server-final.private.log',
                     'container-final.private.json', 'runtime-final-audit.json'}
        for window in windows:
            wid = window['id']
            mandatory.update({f'captures/{wid}.capture.json', f'captures/{wid}.logits.f32',
                              f'requests/{wid}.request.json', f'requests/{wid}.response.json'})
        if not mandatory <= stage['files'].keys():
            raise ValueError('mandatory capture files are absent from the hashed inventory')
        for relative, expected_file in stage['files'].items():
            source = folder / relative
            if (source != source.resolve() or not source.is_relative_to(folder)
                    or source.stat().st_size != expected_file['bytes']
                    or protocol.sha(source) != expected_file['sha256']):
                raise ValueError('capture stage file inventory changed')
        cooling = [json.loads(line)['temperatures'] for line in (folder / 'cooldown.jsonl').read_text().splitlines()]
        thermal = [json.loads(line)['temperatures'] for line in (folder / 'thermal.jsonl').read_text().splitlines()]
        if (not cooling or not thermal or len(cooling[-1]) != 4
                or not all(math.isfinite(v) and 0 <= v <= 75 for v in cooling[-1])
                or any(len(row) != 4 or not all(math.isfinite(v) and v >= 0 for v in row) for row in cooling)
                or any(len(row) != 4 or not all(math.isfinite(v) and 0 <= v < 90 for v in row) for row in thermal)):
            raise ValueError('capture thermal/start gate failed')
        final = json.loads((folder / 'container-final.private.json').read_text())
        if (final['Id'] != stage['container_id'] or final['Image'] != plan['capture_image']
                or final['Name'] != f'/{launcher.PREFIX}-{slot["slot"]}'
                or final['State']['Running'] is not False
                or final['Config']['Labels'].get(launcher.cold.LABEL) != f'{protocol.sha(plan_path)}:{slot["slot"]}'):
            raise ValueError('final owned capture container differs')
        created = datetime.fromisoformat(final['Created'].replace('Z', '+00:00'))
        started = datetime.fromisoformat(final['State']['StartedAt'].replace('Z', '+00:00'))
        stopped = datetime.fromisoformat(final['State']['FinishedAt'].replace('Z', '+00:00'))
        if not begin <= created <= started < stopped <= end:
            raise ValueError('capture process was not a fresh start inside the stage')
        if [w['id'] for w in stage['windows']] != [w['id'] for w in windows]:
            raise ValueError('capture stage lost or reordered windows')
        audit = launcher.runtime_audit((folder / 'server-final.private.log').read_text(), arm, windows)
        if audit != json.loads((folder / 'runtime-final-audit.json').read_text()):
            raise ValueError('native V2/FULL runtime proof does not replay')
        for window in windows:
            tokens = np.load(window['token_path'], allow_pickle=False)
            request = protocol.completion_request(tokens, window['id'], launcher.model_name(arm))
            request_root = folder / 'requests'
            if request != json.loads((request_root / (window['id'] + '.request.json')).read_text()):
                raise ValueError('request does not match the frozen complete causal history')
            response = json.loads((request_root / (window['id'] + '.response.json')).read_text())
            protocol.verify_response(response, request)
    for name in ('canary', 'full'):
        recomputed = launcher.compare_stage(plan, root, name)
        if recomputed != json.loads((root / f'{name}-exact.json').read_text()):
            raise ValueError('stored exact-logit comparison does not replay')
        if name == 'canary' and not recomputed['all_exact']:
            raise ValueError('full capture followed a failed canary')
    if execution.get('full_exact') != recomputed['all_exact']:
        raise ValueError('full exact decision differs from execution receipt')
    return execution, recomputed


def score_capture_pair(teacher, a, b, tokens, exact):
    scores_a = protocol.score_aligned(teacher, a, tokens)
    scores_b = scores_a if exact else protocol.score_aligned(teacher, b, tokens)
    alignment = float(np.mean(scores_a.teacher_top1 == scores_a.realized_token))
    if not ALIGNMENT_BAND[0] <= alignment <= ALIGNMENT_BAND[1]:
        raise ValueError('teacher realized-token alignment guard failed')
    return scores_a, scores_b, alignment


def verify_scoring_pair(a, b, am, bm, closure):
    """Recompute current identity before any reuse of reference KLD scores."""
    for key in ('original_logit_dtype', 'original_logit_width'):
        if am[key] != bm[key] or am[key] != closure[key]:
            raise ValueError('native logit representation changed before scoring')
    current = protocol.exact_logits(a, b)
    if any(current[key] != closure[key] for key in current):
        raise ValueError('logit closure changed between verification and scoring')
    return current


def bind_capture_to_stage(root, arm, window_id, metadata, execution):
    slot = f'full-{arm}'
    expected = next(entry['execution_sha256'] for entry in execution['stages'] if entry['slot'] == slot)
    raw_receipt = (root / slot / 'execution.json').read_bytes()
    if hashlib.sha256(raw_receipt).hexdigest() != expected:
        raise ValueError('capture stage changed before scoring')
    stage = json.loads(raw_receipt)
    entry = next(row for row in stage['windows'] if row['id'] == window_id)
    meta_path = root / slot / 'captures' / f'{window_id}.capture.json'
    if (metadata['raw_sha256'] != entry['raw_sha256']
            or protocol.sha(meta_path) != entry['capture_sha256']):
        raise ValueError('reloaded capture differs from the verified stage snapshot')


def run(plan_path, output):
    import torch
    torch.set_num_threads(CPU_THREADS)
    plan, _ = launcher.authenticate(plan_path)
    if output != output.resolve() or output.exists():
        raise ValueError('fresh canonical analysis output required')
    execution, exact = verify_execution(plan, plan_path)
    receipt_paths = [plan_path, *(Path(plan['output']) / name for name in
                                 ('execution.json', 'canary-exact.json', 'full-exact.json'))]
    receipt_hashes = {str(path): protocol.sha(path) for path in receipt_paths}
    output.mkdir(parents=True, mode=0o700)
    started = time.time_ns()
    write_json(output / 'analysis-contract.json', {
        'plan_sha256': protocol.sha(plan_path), 'execution_sha256': protocol.sha(Path(plan['output']) / 'execution.json'),
        'metric_code_sha256': METRIC_CODE_SHA256, 'metric': 'KL(teacher || student), nats, full vocabulary',
        'direction': 'candidate N64 minus reference N128', 'rows': 65504, 'window_count': 32,
        'decision': 'exact-logit closure; no post-hoc KLD tolerance',
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_B,
        'torch_cpu_threads': CPU_THREADS,
        'source_receipt_sha256': receipt_hashes,
        'original_head_precision': 'recorded in each capture; stored FP32 does not change head arithmetic',
    })
    records = []
    try:
        for window, closure in zip(plan['windows'], exact['windows'], strict=True):
            window_id = window['id']
            if closure['window_id'] != window_id:
                raise ValueError('exact and KLD window inventories differ')
            tokens = np.load(window['token_path'], allow_pickle=False)
            root = Path(plan['output'])
            a, am = protocol.load_capture(root / 'full-n128/captures', window_id, tokens)
            b, bm = protocol.load_capture(root / 'full-n64/captures', window_id, tokens)
            bind_capture_to_stage(root, 'n128', window_id, am, execution)
            bind_capture_to_stage(root, 'n64', window_id, bm, execution)
            current_closure = verify_scoring_pair(a, b, am, bm, closure)
            teacher_path = Path(plan['teacher_root']) / window['teacher_path']
            if protocol.sha(teacher_path) != window['teacher_sha256']:
                raise ValueError('teacher changed before scoring')
            teacher = _load_teacher(teacher_path, window)
            sa, sb, alignment = score_capture_pair(teacher, a, b, tokens, current_closure['exact'])
            for arm, metadata in (('n128', am), ('n64', bm)):
                raw_path = root / f'full-{arm}' / 'captures' / f'{window_id}.logits.f32'
                if protocol.sha(raw_path) != metadata['raw_sha256']:
                    raise ValueError('raw capture changed during scoring')
                bind_capture_to_stage(root, arm, window_id, metadata, execution)
            for arm, scores in (('n128', sa), ('n64', sb)):
                with (output / f'{window_id}.{arm}.npz').open('xb') as handle:
                    np.savez(handle, **{field.name: getattr(scores, field.name) for field in fields(scores)})
            record = {'window_id': window_id, 'domain': window['domain'], 'rows': protocol.ROWS,
                'n128_mean_kld': float(sa.kld.mean()), 'n64_mean_kld': float(sb.kld.mean()),
                'n128_one_token_prefill_kld': float(sa.kld[0]), 'n64_one_token_prefill_kld': float(sb.kld[0]),
                'n128_true_decode_mean_kld': float(sa.kld[1:].mean()),
                'n64_true_decode_mean_kld': float(sb.kld[1:].mean()),
                'teacher_realized_token_alignment': alignment, 'exact_logits': closure['exact'],
                'candidate_metric_reused_from_bitexact_reference': closure['exact'],
                'teacher_sha256': window['teacher_sha256'], 'n128_raw_sha256': am['raw_sha256'],
                'n64_raw_sha256': bm['raw_sha256'], 'original_logit_dtype': am['original_logit_dtype']}
            write_json(output / f'{window_id}.json', record)
            records.append(record)
            print(json.dumps({'window_id': window_id, 'windows_scored': len(records),
                              'n128_kld': record['n128_mean_kld'], 'n64_kld': record['n64_mean_kld']}), flush=True)
            del teacher, a, b, sa, sb
        launcher.verify_input_stats(plan)
        if any(protocol.sha(Path(path)) != expected for path, expected in receipt_hashes.items()):
            raise ValueError('plan, execution or exact-comparison receipt changed during scoring')
        result = {'schema': 'glm53-p8.forced-m1-v2-kld.v1', 'status': 'complete',
            'plan_sha256': protocol.sha(plan_path), 'capture_image': plan['capture_image'],
            'metric_code_sha256': METRIC_CODE_SHA256, 'metric': 'KL(teacher || student), nats',
            'torch_cpu_threads': CPU_THREADS,
            'source_receipt_sha256': receipt_hashes,
            'exact_logits': exact['all_exact'], **aggregate(records, exact['all_exact']),
            'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
            'speed_measurement_valid': False, 'allocation_restart': False,
            'isa_cost': 'P8 E4M3 mxf8f6f4 has twice NVFP4 MMA issue count; not P4',
            'limits': ['Development conditional-fit evidence, not protected-selection qualification.',
                       'Both arms are the same P8 checkpoint and topology; not a matched NVFP4 quality comparison.',
                       'Window intervals do not establish independent model or quantization replication.'],
            'started_unix_ns': started, 'completed_unix_ns': time.time_ns()}
        result['files'] = {p.name: protocol.sha(p) for p in output.iterdir() if p.is_file()}
        write_json(output / 'analysis.json', result)
        return result
    except BaseException as error:
        write_json(output / 'analysis.failed.json', {'status': 'failed', 'error_type': type(error).__name__,
            'error': str(error), 'windows_scored': len(records), 'started_unix_ns': started,
            'completed_unix_ns': time.time_ns(), 'allocation_restart': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
