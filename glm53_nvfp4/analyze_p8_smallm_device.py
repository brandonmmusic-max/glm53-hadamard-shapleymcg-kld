"""Validate identical-workload P8 M1 closure and developmental timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell_map(payload: dict) -> dict[int, dict]:
    return {int(cell['tokens']): cell for cell in payload['cells']}


def analyze(baseline: dict, candidate: dict, baseline_speed: dict,
            candidate_speed: dict, baseline_fallback: dict,
            candidate_fallback: dict) -> dict:
    payloads = (baseline, candidate, baseline_speed, candidate_speed,
                baseline_fallback, candidate_fallback)
    identities = {(p['sidecar']['sha256'], p['dense']['sha256'], p['rank'], p['layer'])
                  for p in payloads}
    if len(identities) != 1:
        raise ValueError('payload identities differ')
    for payload in payloads:
        if payload['decision'] != 'pass' or not payload['deterministic_output']:
            raise ValueError('a device arm failed its declared closure')
    if baseline['small_m_scheduler'] or candidate_speed['small_m_scheduler'] is not True:
        raise ValueError('baseline/candidate labels differ')
    if _cell_map(baseline)[1]['output_sha256'] != _cell_map(candidate)[1]['output_sha256']:
        raise ValueError('M1 output differs')
    for tokens in (2, 3):
        if (_cell_map(baseline_fallback)[tokens]['output_sha256']
                != _cell_map(candidate_fallback)[tokens]['output_sha256']):
            raise ValueError(f'M{tokens} fallback output differs')
    base_timing = _cell_map(baseline_speed)[1]['timing']
    cand_timing = _cell_map(candidate_speed)[1]['timing']
    if (base_timing['warmups'], base_timing['repeats']) != (20, 100):
        raise ValueError('baseline timing inventory differs')
    if (cand_timing['warmups'], cand_timing['repeats']) != (20, 100):
        raise ValueError('candidate timing inventory differs')
    base_ms, cand_ms = base_timing['median_ms'], cand_timing['median_ms']
    return {
        'schema': 'glm53-p8-smallm-device-analysis.v1',
        'status': 'pass',
        'scope': ('single-GPU developmental kernel timing and arithmetic closure; '
                  'not CUDA-graph serving throughput or KLD qualification'),
        'm1_bit_exact': True,
        'm2_m3_fallback_bit_exact': True,
        'five_run_bitwise_determinism': True,
        'baseline_median_ms': base_ms,
        'candidate_median_ms': cand_ms,
        'device_time_reduction_percent': 100.0 * (base_ms - cand_ms) / base_ms,
        'device_time_speedup': base_ms / cand_ms,
        'timing_warmups': 20,
        'timing_repeats': 100,
        'sidecar_sha256': baseline['sidecar']['sha256'],
        'dense_sha256': baseline['dense']['sha256'],
        'rank': baseline['rank'],
        'layer': baseline['layer'],
        'protected_roles_opened': [],
        'ldlq': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'baseline_speed', 'candidate_speed',
                 'baseline_fallback', 'candidate_fallback'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(*(json.loads(getattr(args, name).read_text()) for name in (
        'baseline', 'candidate', 'baseline_speed', 'candidate_speed',
        'baseline_fallback', 'candidate_fallback')))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
