"""CPU KLD for the corrected-index two-stage full32 campaign.

The external canary is authenticated, never copied into synthetic stages.
Arithmetic closure and absolute teacher KLD are distinct reported outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from . import p8_index_order_full as launcher
from . import p8_decode_analysis as metric
from . import p8_decode_protocol as protocol

SLOTS = ['full-n128', 'full-n64']


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timezone-aware chronology required')
    return result


def panel_inventory(windows):
    if (len(windows) != 32 or len({w['id'] for w in windows}) != 32
            or Counter(w['domain'] for w in windows) != Counter({d: 8 for d in protocol.DOMAINS})
            or any(not re.fullmatch(r'conditional-fit-\d{4}', w['id']) for w in windows)):
        raise ValueError('exact balanced conditional-fit32 panel required')


def thermal_gate(folder):
    cooling = [json.loads(line)['temperatures'] for line in (folder / 'cooldown.jsonl').read_text().splitlines()]
    thermal = [json.loads(line)['temperatures'] for line in (folder / 'thermal.jsonl').read_text().splitlines()]
    def valid(row, maximum):
        return len(row) == 4 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= maximum for v in row)
    if (not cooling or not thermal or not valid(cooling[-1], 75)
            or any(not valid(row, float('inf')) for row in cooling)
            or any(not valid(row, 90) or max(row) >= 90 for row in thermal)):
        raise ValueError('thermal or startup gate failed')


def verify_execution(plan, plan_path):
    """Verify real two-stage receipts without mutating the old four-stage API."""
    launcher.verify_identities(plan)
    panel_inventory(plan['windows'])
    root, digest = Path(plan['output']), protocol.sha(plan_path)
    execution = json.loads((root / 'execution.json').read_text())
    if (execution.get('schema') != 'glm53-p8.index-order-full-execution.v1'
            or execution.get('plan_sha256') != digest or execution.get('capture_image') != plan['capture_image']
            or type(execution.get('exit_code')) is not int or execution['exit_code'] != 0
            or execution.get('capture_protocol_complete') is not True
            or execution.get('allocation_restart') is not False
            or execution.get('opened_roles') != ['conditional-fit']
            or execution.get('protected_roles_opened') != []
            or execution.get('observer_enabled') is not False
            or execution.get('speed_measurement_valid') is not False
            or execution.get('final_identity_audit') != {'ok': True}
            or execution.get('restoration_safety') != {'ok': True, 'errors': [], 'containers': []}
            or set(execution.get('prior', {})) != {'backend', 'timer'}
            or any(type(v) is not bool for v in execution['prior'].values())
            or execution.get('restoration') != {**execution['prior'], 'errors': []}):
        raise ValueError('full capture or restoration unauthenticated')
    if [s['slot'] for s in execution['stages']] != SLOTS:
        raise ValueError('exactly two fresh full stages required; external canary is separate')
    root_begin, root_end = [timestamp(execution[k]) for k in ('started_at', 'finished_at')]
    if not timestamp(plan['created_at']) <= root_begin < root_end:
        raise ValueError('root chronology invalid')
    previous_end, ids = root_begin, set()
    for item in execution['stages']:
        slot, arm = item['slot'], item['slot'].split('-')[1]
        folder = root / slot
        if protocol.sha(folder / 'execution.json') != item['execution_sha256']:
            raise ValueError('stage receipt changed')
        stage = json.loads((folder / 'execution.json').read_text())
        if (stage.get('schema') != 'glm53-p8.forced-m1-v2-stage.v1'
                or type(stage.get('exit_code')) is not int or stage['exit_code'] != 0
                or stage.get('cleanup') != {'ok': True, 'errors': []}
                or stage.get('stage') != 'full' or stage.get('arm') != arm
                or stage.get('capture_image') != plan['capture_image'] or stage.get('plan_sha256') != digest
                or stage.get('unreadable_artifacts') != [] or stage.get('protected_roles_opened') != []
                or stage.get('allocation_restart') is not False
                or not re.fullmatch('[a-f0-9]{64}', stage.get('container_id', ''))
                or stage['container_id'] in ids):
            raise ValueError('stage identity or success differs')
        ids.add(stage['container_id'])
        begin, end = [timestamp(stage[k]) for k in ('started_at', 'finished_at')]
        if not previous_end <= begin < end <= root_end:
            raise ValueError('capture stage overlap or chronology invalid')
        previous_end = end
        mandatory = {'cooldown.jsonl', 'thermal.jsonl', 'server-ready.private.log', 'runtime-ready-audit.json',
                     'server-final.private.log', 'container-final.private.json', 'runtime-final-audit.json'}
        for w in plan['windows']:
            mandatory.update({f'captures/{w["id"]}.capture.json', f'captures/{w["id"]}.logits.f32',
                              f'requests/{w["id"]}.request.json', f'requests/{w["id"]}.response.json'})
        if not mandatory <= stage['files'].keys():
            raise ValueError('mandatory stage files missing')
        actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != folder / 'execution.json'}
        if actual != set(stage['files']):
            raise ValueError('unreceipted or missing stage artifacts')
        for relative, expected in stage['files'].items():
            source = folder / relative
            if (source != source.resolve() or not source.is_relative_to(folder)
                    or source.stat().st_size != expected['bytes'] or protocol.sha(source) != expected['sha256']):
                raise ValueError('stage artifact changed')
        thermal_gate(folder)
        final = json.loads((folder / 'container-final.private.json').read_text())
        if (final['Id'] != stage['container_id'] or final['Image'] != plan['capture_image']
                or final['Name'] != f'/{launcher.PREFIX}-{slot}' or final['State']['Running'] is not False
                or final['Config']['Labels'].get(launcher.base.cold.LABEL) != f'{digest}:{slot}'):
            raise ValueError('stopped owned container identity differs')
        created = timestamp(final['Created'])
        started = timestamp(final['State']['StartedAt'])
        stopped = timestamp(final['State']['FinishedAt'])
        if not begin <= created <= started < stopped <= end:
            raise ValueError('container not a fresh process within stage')
        if [w['id'] for w in stage['windows']] != [w['id'] for w in plan['windows']]:
            raise ValueError('capture panel missing or reordered')
        for phase, completed in (('ready', None), ('final', plan['windows'])):
            audit = launcher.runtime_audit((folder / f'server-{phase}.private.log').read_text(), arm, completed)
            if audit != json.loads((folder / f'runtime-{phase}-audit.json').read_text()):
                raise ValueError('corrected-index FULL/M1 runtime proof does not replay')
        for window, row in zip(plan['windows'], stage['windows'], strict=True):
            if protocol.sha(Path(window['token_path'])) != window['input_sha256']:
                raise ValueError('token input changed')
            tokens = np.load(window['token_path'], allow_pickle=False)
            request = protocol.completion_request(tokens, window['id'], launcher.base.model_name(arm))
            request_root = folder / 'requests'
            if request != json.loads((request_root / f'{window["id"]}.request.json').read_text()):
                raise ValueError('request does not match approved causal sequence')
            response = json.loads((request_root / f'{window["id"]}.response.json').read_text())
            if (protocol.verify_response(response, request) != row['response_audit']
                    or row['rows'] != 2047 or row['domain'] != window['domain']
                    or row['capture_sha256'] != stage['files'][f'captures/{window["id"]}.capture.json']['sha256']
                    or row['raw_sha256'] != stage['files'][f'captures/{window["id"]}.logits.f32']['sha256']):
                raise ValueError('stage response/window proof differs')
    exact_path = root / 'full-exact.json'
    if protocol.sha(exact_path) != execution.get('full_exact_sha256'):
        raise ValueError('full comparison hash chain differs')
    exact = launcher.compare_full(plan)
    if exact != json.loads(exact_path.read_text()):
        raise ValueError('full exact comparison does not replay')
    if (type(execution.get('full_exact')) is not bool or execution['full_exact'] is not exact['all_exact']
            or len(exact['windows']) != 32
            or [w['window_id'] for w in exact['windows']] != [w['id'] for w in plan['windows']]
            or any(w['rows'] != 2047 or w['vocabulary'] != 154880 or type(w.get('exact')) is not bool
                   for w in exact['windows'])
            or exact['all_exact'] is not all(w['exact'] for w in exact['windows'])):
        raise ValueError('full closure decision or dimensions differ')
    return execution, exact


def run(plan_path, output):
    import torch
    torch.set_num_threads(metric.CPU_THREADS)
    plan = launcher.authenticate(plan_path)
    if output != output.resolve() or output.exists():
        raise ValueError('fresh canonical analysis output required')
    execution, exact = verify_execution(plan, plan_path)
    root = Path(plan['output'])
    paths = [plan_path, root / 'execution.json', root / 'full-exact.json']
    receipts = {str(p): protocol.sha(p) for p in paths}
    output.mkdir(mode=0o700, parents=True)
    started, records = time.time_ns(), []
    metric.write_json(output / 'analysis-contract.json', {
        'plan_sha256': protocol.sha(plan_path), 'source_receipt_sha256': receipts,
        'metric_code_sha256': metric.METRIC_CODE_SHA256, 'metric': 'KL(teacher || student), nats, full vocabulary',
        'rows': 65504, 'window_count': 32, 'torch_cpu_threads': metric.CPU_THREADS,
        'bootstrap_replicates': metric.BOOTSTRAP_B, 'bootstrap_seed': metric.BOOTSTRAP_SEED,
        'direction': 'N64 minus N128, same corrected P8 checkpoint',
        'decision': 'Bitwise closure and absolute KLD reported separately; KLD cannot override failed closure',
        'tail_policy': 'Existing incomplete-KPool-tail omission unchanged; no final quality qualification',
        'external_canary': 'Authenticated by the full plan; no canary rerun or synthesized stage',
    })
    try:
        for window, closure in zip(plan['windows'], exact['windows'], strict=True):
            wid = window['id']
            if closure['window_id'] != wid:
                raise ValueError('closure and scoring windows differ')
            tokens = np.load(window['token_path'], allow_pickle=False)
            a, am = protocol.load_capture(root / 'full-n128/captures', wid, tokens)
            b, bm = protocol.load_capture(root / 'full-n64/captures', wid, tokens)
            metric.bind_capture_to_stage(root, 'n128', wid, am, execution)
            metric.bind_capture_to_stage(root, 'n64', wid, bm, execution)
            current = metric.verify_scoring_pair(a, b, am, bm, closure)
            teacher_path = Path(plan['teacher_root']) / window['teacher_path']
            if protocol.sha(teacher_path) != window['teacher_sha256']:
                raise ValueError('teacher changed before scoring')
            teacher = metric._load_teacher(teacher_path, window)
            sa, sb, alignment = metric.score_capture_pair(teacher, a, b, tokens, current['exact'])
            if protocol.sha(teacher_path) != window['teacher_sha256']:
                raise ValueError('teacher changed during scoring')
            for arm, metadata, scores in (('n128', am, sa), ('n64', bm, sb)):
                if protocol.sha(root / f'full-{arm}/captures/{wid}.logits.f32') != metadata['raw_sha256']:
                    raise ValueError('capture changed during scoring')
                metric.bind_capture_to_stage(root, arm, wid, metadata, execution)
                with (output / f'{wid}.{arm}.npz').open('xb') as handle:
                    np.savez(handle, **{field.name: getattr(scores, field.name) for field in fields(scores)})
            row = {'window_id': wid, 'domain': window['domain'], 'rows': 2047,
                   'n128_mean_kld': float(sa.kld.mean()), 'n64_mean_kld': float(sb.kld.mean()),
                   'n128_one_token_prefill_kld': float(sa.kld[0]), 'n64_one_token_prefill_kld': float(sb.kld[0]),
                   'n128_true_decode_mean_kld': float(sa.kld[1:].mean()),
                   'n64_true_decode_mean_kld': float(sb.kld[1:].mean()),
                   'teacher_realized_token_alignment': alignment, 'exact_logits': current['exact'],
                   'candidate_metric_reused_from_bitexact_reference': current['exact'],
                   'teacher_sha256': window['teacher_sha256'], 'n128_raw_sha256': am['raw_sha256'],
                   'n64_raw_sha256': bm['raw_sha256'], 'original_logit_dtype': am['original_logit_dtype']}
            metric.write_json(output / f'{wid}.json', row)
            records.append(row)
            print(json.dumps({'windows_scored': len(records), 'window_id': wid,
                              'n128_kld': row['n128_mean_kld'], 'n64_kld': row['n64_mean_kld']}), flush=True)
            del teacher, a, b, sa, sb
        launcher.verify_identities(plan)
        if any(protocol.sha(Path(p)) != sha for p, sha in receipts.items()):
            raise ValueError('analysis receipt input changed')
        result = {'schema': 'glm53-p8.index-order-full-kld.v1', 'status': 'complete',
                  'plan_sha256': protocol.sha(plan_path), 'source_receipt_sha256': receipts,
                  'capture_image': plan['capture_image'], 'metric_code_sha256': metric.METRIC_CODE_SHA256,
                  'metric': 'KL(teacher || student), nats', 'torch_cpu_threads': metric.CPU_THREADS,
                  'exact_logits': exact['all_exact'], **metric.aggregate(records, exact['all_exact']),
                  'opened_roles': ['conditional-fit'], 'protected_roles_opened': [],
                  'speed_measurement_valid': False, 'allocation_restart': False,
                  'isa_cost': 'P8 E4M3 mxf8f6f4 issues twice NVFP4 MMA count; not P4',
                  'limits': ['Development conditional-fit evidence, not protected qualification.',
                             'N128 and N64 share P8 checkpoint/topology; no matched NVFP4 quality comparison.',
                             'Incomplete-KPool-tail omission remains unchanged and unqualified.',
                             'Window intervals are not independent quantization replications.'],
                  'started_unix_ns': started, 'completed_unix_ns': time.time_ns()}
        result['files'] = {p.name: protocol.sha(p) for p in output.iterdir() if p.is_file()}
        metric.write_json(output / 'analysis.json', result)
        return result
    except BaseException as error:
        metric.write_json(output / 'analysis.failed.json', {'status': 'failed', 'error_type': type(error).__name__,
            'error': str(error), 'windows_scored': len(records), 'started_unix_ns': started,
            'completed_unix_ns': time.time_ns(), 'allocation_restart': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
