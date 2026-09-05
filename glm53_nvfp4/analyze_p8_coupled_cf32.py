"""Fixed three-arm development analysis; no capture or protected-role access."""
from collections import Counter

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval


ARMS = ('stock', 'identity_p8', 'coupled_p8')
CONDITIONS = ('attention', 'kv_dtype', 'moe_backend', 'activation_precision', 'bpw')


def analyze(windows, arms):
    """Consume normalized scored rows after capture/teacher installation checks.

    Each arm has `conditions` and `windows`, with one row per manifest ID:
    window_id, domain, mean_kld, true_decode_mean_kld. This function proves
    pairing/aggregation only, not provenance, causal alignment or GPU dispatch.
    """
    ids = [w['id'] for w in windows]
    domains = {w['id']: w['domain'] for w in windows}
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('requires exactly 32 unique manifest windows')
    if sorted(Counter(domains.values()).values()) != [8, 8, 8, 8]:
        raise ValueError('requires four domains with eight windows each')
    if set(arms) != set(ARMS):
        raise ValueError('requires exactly stock, identity_p8, coupled_p8')
    values, all_rows, conditions = {}, {}, {}
    for arm in ARMS:
        entry = arms[arm]
        conditions[arm] = entry['conditions']
        if any(key not in conditions[arm] or conditions[arm][key] is None
               for key in CONDITIONS):
            raise ValueError(f'{arm}: missing runtime/precision/rate label')
        rows = entry['windows']
        by_id = {r['window_id']: r for r in rows}
        if len(rows) != 32 or set(by_id) != set(ids):
            raise ValueError(f'{arm}: duplicate, missing or unexpected window')
        if any(by_id[key]['domain'] != domains[key] for key in ids):
            raise ValueError(f'{arm}: domain mismatch')
        values[arm] = np.array([by_id[key]['true_decode_mean_kld'] for key in ids], dtype=np.float64)
        all_rows[arm] = np.array([by_id[key]['mean_kld'] for key in ids], dtype=np.float64)
        if any(not np.isfinite(v).all() or (v < 0).any()
               for v in (values[arm], all_rows[arm])):
            raise ValueError(f'{arm}: nonfinite or negative KLD')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 32, size=(BOOTSTRAP_B, 32))
    comparisons = {}
    for control in ('identity_p8', 'stock'):
        delta = values['coupled_p8'] - values[control]
        bootstrap = delta[indices].mean(axis=1)
        constant = bool(np.all(delta == delta[0]))
        interval = ((float(delta[0]), float(delta[0])) if constant
                    else bca_mean_interval(delta, bootstrap))
        control_mean = float(values[control].mean())
        comparisons[control] = {
            'candidate_conditions': conditions['coupled_p8'],
            'control_conditions': conditions[control],
            'mean_delta_kld': float(delta.mean()),
            'relative_improvement_percent': (float(-100 * delta.mean() / control_mean)
                                             if control_mean > 0 else None),
            'paired_bca95': list(interval),
            'degenerate_delta_distribution': constant,
            'paired_window_wins': int((delta < 0).sum()),
            'per_window_delta': [{'window_id': key, 'delta_kld': float(delta[i])}
                                 for i, key in enumerate(ids)],
            'per_domain_mean_delta': {
                domain: float(delta[[domains[key] == domain for key in ids]].mean())
                for domain in sorted(set(domains.values()))},
        }
    return {
        'schema': 'glm53.p8-coupled-cf32-paired-development.v1',
        'role': 'conditional-fit', 'windows': 32,
        'decision': 'pass' if comparisons['identity_p8']['mean_delta_kld'] < 0 else 'fail',
        'decision_rule': 'coupled mean true-decode KLD below identity P8; BCa nonblocking',
        'stock_win': comparisons['stock']['mean_delta_kld'] < 0,
        'bootstrap': {'unit': 'window', 'replicates': BOOTSTRAP_B, 'seed': BOOTSTRAP_SEED},
        'arms': {arm: {'conditions': conditions[arm],
                       'true_decode_mean_kld': float(values[arm].mean()),
                       'including_prefill_mean_kld': float(all_rows[arm].mean())}
                 for arm in ARMS},
        'comparisons': comparisons,
        'claim_boundary': 'Already-opened CF32 development evidence, not final qualification; input provenance and runtime installation require separate receipts',
        'isa_cost': 'P8 E4M3 mxf8f6f4 has twice the NVFP4 MMA issue count',
    }
