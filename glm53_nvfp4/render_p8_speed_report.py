"""Render only a complete, receipt-linked frozen five-pair speed analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def validate(analysis: dict, audit: dict) -> None:
    if audit.get('status') != 'pass' or audit.get('missing'):
        raise ValueError('complete receipt audit required')
    rows = audit.get('runs', [])
    if (len(rows) != 10 or len({row['container_id'] for row in rows}) != 10
            or {(row['round'], row['arm']) for row in rows}
            != {(r, a) for r in range(1, 6) for a in ('p8', 'exl3')}):
        raise ValueError('ten distinct cold containers and exact round inventory required')
    if analysis['plan']['sha256'] != audit['plan_sha256']:
        raise ValueError('plan identities differ')
    indexed = {(row['round'], row['arm']): row for row in rows}
    for arm in ('p8', 'exl3'):
        runs = analysis[arm]['runs']
        if len(runs) != 5:
            raise ValueError('five runs per arm required')
        for i, run in enumerate(runs, 1):
            if run['sha256'] != indexed[i, arm]['benchmark_sha256']:
                raise ValueError('analysis and receipt run identities differ')
        for context in ('32768', '65536'):
            for metric, field in (('prefill', 'server_tps'), ('decode', 'aggregate_tps')):
                median = statistics.median(r[metric][context][field] for r in runs)
                if median != analysis[arm]['median'][metric][context][field]:
                    raise ValueError('reported median differs from frozen run values')
    expected = all(analysis['p8']['median'][metric]['32768'][field]
                   > analysis['exl3']['median'][metric]['32768'][field]
                   for metric, field in (('prefill', 'server_tps'), ('decode', 'aggregate_tps')))
    if (analysis['decision']['passed'] is not expected
            or analysis['decision']['allocation_game'] != ('advance-to-power-sized-design' if expected else 'stop')):
        raise ValueError('decision does not implement the declared two-median rule')


def render(analysis: dict, audit: dict) -> str:
    validate(analysis, audit)
    decision = 'PASS' if analysis['decision']['passed'] else 'FAIL — allocation game stopped'
    lines = ['# Uniform native P8 versus EXL3: five cold runs', '',
             f'Decision: **{decision}**.', '',
             'This is a target-only system comparison: P8 TP4/no-EP/DCP1 versus',
             'EXL3 TP4/EP4/DCP4. It is not matched-topology codec causality or',
             'a comparison against production MTP5. Compiled caches remain warm;',
             'each cold run uses a new server container and worker processes.', '',
             'P8 is E4M3 `mxf8f6f4` m16n8k32: twice the NVFP4 MMA issue count',
             'for equal K, not native P4/NVFP4 speed class.', '',
             '## Frozen primary decision', '',
             analysis['decision']['rule'], '',
             '| Arm | 32K prefill median tokens/s | C1 32K decode median tokens/s |',
             '|---|---:|---:|']
    for arm in ('p8', 'exl3'):
        median = analysis[arm]['median']
        lines.append(f'| {arm.upper()} | {median["prefill"]["32768"]["server_tps"]:.2f} | '
                     f'{median["decode"]["32768"]["aggregate_tps"]:.4f} |')
    lines.extend(['', '## Every run, including secondary 64K cells', '',
                  'Prefill uses Prometheus-validated server input tokens/s. Decode uses',
                  'C1 continuous-usage output tokens/s. Prompt counts are actual, not the',
                  'nominal context labels. These rounded values are for display only.', '',
                  '| Round | Arm | Actual prompt tokens 32K / 64K | Prefill 32K / 64K | Decode 32K / 64K | Peak GPU C |',
                  '|---:|---|---:|---:|---:|---:|'])
    indexed = {(row['round'], row['arm']): row for row in audit['runs']}
    for i in range(5):
        for arm in ('p8', 'exl3'):
            r = analysis[arm]['runs'][i]
            p, d = r['prefill'], r['decode']
            lines.append(f'| {i+1} | {arm.upper()} | {p["32768"]["prompt_tokens"]} / {p["65536"]["prompt_tokens"]} | '
                         f'{p["32768"]["server_tps"]:.2f} / {p["65536"]["server_tps"]:.2f} | '
                         f'{d["32768"]["aggregate_tps"]:.4f} / {d["65536"]["aggregate_tps"]:.4f} | '
                         f'{indexed[i+1, arm]["measured_max_temp_c"]:.0f} |')
    lines.extend(['', '## Evidence and limits', '',
                  'The receipt audit verifies ten distinct containers, chronological run order,',
                  'within-arm configuration stability, graph/backend/dispatch evidence, hashes,',
                  'and thermal limits. The five repeats measure serving-process variation for',
                  'one prepared checkpoint, not independent quantization-pipeline replicates.', '',
                  'Full-model KLD was measured separately at 0.04002139481676188 on the',
                  '32 conditional-fit windows, in eager FP8-MLA-KV serving. This graph/NVFP4-KV',
                  'speed test does not establish quality parity for its different serving regime.',
                  'The earlier 24.96% contextual KLD decrease is not a matched NVFP4 win.', '',
                  f'Frozen plan SHA-256: `{analysis["plan"]["sha256"]}`.', '',
                  'Reproduction inputs: `analysis.json`, `speed-receipt-audit.json`, per-run',
                  '`benchmark.json` and receipts in the terminal speed-v2a3 evidence snapshot.',
                  'Failures and amended attempts remain in `P8_SPEED_ATTEMPTS.md`.', '',
                  'Existing ExLlamaV3, KQuant, QSRT and w4a8_trellis attribution applies.', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = render(json.loads(args.analysis.read_text()), json.loads(args.audit.read_text()))
    with args.output.open('x') as handle:
        handle.write(report)
    print(json.dumps({'report': str(args.output),
                      'sha256': hashlib.sha256(report.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
