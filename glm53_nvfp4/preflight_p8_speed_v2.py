"""Validate the disclosed product-topology amendment before speed execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(plan: dict, prior: dict, prior_sha: str, failed_log: str) -> None:
    if plan['prior_plan']['sha256'] != prior_sha:
        raise ValueError('prior speed plan identity differs')
    if 'checkpoint=2, runtime=4' not in failed_log:
        raise ValueError('missing preserved topology rejection')
    expected = {'candidate': {'tp': 4, 'ep': False, 'dcp': 1},
                'baseline': {'tp': 4, 'ep': True, 'dcp': 4}}
    for arm, topology in expected.items():
        if plan[arm]['topology'] != topology:
            raise ValueError(f'{arm} topology differs from product trial')
        excluded = {'topology', 'image_id'} if arm == 'candidate' else {'topology'}
        if {k: v for k, v in plan[arm].items() if k not in excluded} != {
            k: v for k, v in prior[arm].items() if k not in excluded
        }:
            raise ValueError(f'{arm} artifact configuration changed')
    if plan['candidate']['image_id'] != 'sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8':
        raise ValueError('P8 image differs from image amendment 2')
    for key in ('benchmark', 'analysis'):
        if plan[key] != prior[key]:
            raise ValueError(f'{key} protocol changed')
    for key in ('pass', 'secondary'):
        if plan['decision_before_result'][key] != prior['decision_before_result'][key]:
            raise ValueError('numerical decision changed')
    for key, value in prior['common_serving_regime'].items():
        if key != 'topology' and plan['common_serving_regime'].get(key) != value:
            raise ValueError(f'common serving field changed: {key}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prior-plan', type=Path, required=True)
    parser.add_argument('--failed-log', type=Path, required=True)
    args = parser.parse_args()
    prior_bytes = args.prior_plan.read_bytes()
    verify(json.loads(args.plan.read_text()), json.loads(prior_bytes),
           hashlib.sha256(prior_bytes).hexdigest(), args.failed_log.read_text())
    print(json.dumps({'status': 'pass', 'claim': 'different-topology target-only product comparison',
                      'failed_log_sha256': hashlib.sha256(args.failed_log.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
