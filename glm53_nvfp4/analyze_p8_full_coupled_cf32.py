"""Paired full-model CF32 analysis: coupled_full candidate versus identity_full control.

Same estimator as the three-layer pilot analysis (equal-window mean true-decode
KLD, paired window bootstrap with the pinned seed and replicate count, BCa
interval) applied to the two full-model arms.  No capture or protected-role
access happens here; callers supply normalized scored rows.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval

ARMS = ("coupled_full", "identity_full")
CANDIDATE, CONTROL = ARMS
CONDITIONS = ("attention", "kv_dtype", "moe_backend", "activation_precision", "bpw",
              "layers", "image_id")
SCHEMA = "glm53.p8-full-coupled-cf32-paired-development.v1"


def analyze(windows: list[dict], arms: dict) -> dict:
    ids = [w["id"] for w in windows]
    domains = {w["id"]: w["domain"] for w in windows}
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("requires exactly 32 unique manifest windows")
    if sorted(Counter(domains.values()).values()) != [8, 8, 8, 8]:
        raise ValueError("requires four domains with eight windows each")
    if set(arms) != set(ARMS):
        raise ValueError("requires exactly coupled_full and identity_full")
    values, all_rows, conditions = {}, {}, {}
    for arm in ARMS:
        entry = arms[arm]
        conditions[arm] = entry["conditions"]
        missing = [key for key in CONDITIONS if key not in conditions[arm] or conditions[arm][key] is None]
        if missing:
            raise ValueError(f"{arm}: missing runtime/precision/rate label {missing}")
        rows = entry["windows"]
        by_id = {r["window_id"]: r for r in rows}
        if len(rows) != 32 or set(by_id) != set(ids):
            raise ValueError(f"{arm}: duplicate, missing or unexpected window")
        if any(by_id[key]["domain"] != domains[key] for key in ids):
            raise ValueError(f"{arm}: domain mismatch")
        values[arm] = np.array([by_id[key]["true_decode_mean_kld"] for key in ids], dtype=np.float64)
        all_rows[arm] = np.array([by_id[key]["mean_kld"] for key in ids], dtype=np.float64)
        if any(not np.isfinite(v).all() or (v < 0).any() for v in (values[arm], all_rows[arm])):
            raise ValueError(f"{arm}: nonfinite or negative KLD")
    for key in ("attention", "kv_dtype", "activation_precision", "image_id", "layers"):
        if conditions[CANDIDATE][key] != conditions[CONTROL][key]:
            raise ValueError(f"arms differ on the matched condition {key}; the pair is not matched")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 32, size=(BOOTSTRAP_B, 32))
    delta = values[CANDIDATE] - values[CONTROL]
    bootstrap = delta[indices].mean(axis=1)
    constant = bool(np.all(delta == delta[0]))
    interval = (float(delta[0]), float(delta[0])) if constant else bca_mean_interval(delta, bootstrap)
    control_mean = float(values[CONTROL].mean())
    comparison = {
        "candidate_conditions": conditions[CANDIDATE],
        "control_conditions": conditions[CONTROL],
        "mean_delta_kld": float(delta.mean()),
        "relative_improvement_percent": float(-100 * delta.mean() / control_mean) if control_mean > 0 else None,
        "paired_bca95": list(interval),
        "degenerate_delta_distribution": constant,
        "paired_window_wins": int((delta < 0).sum()),
        "per_window_delta": [{"window_id": key, "delta_kld": float(delta[i])} for i, key in enumerate(ids)],
        "per_domain_mean_delta": {
            domain: float(delta[[domains[key] == domain for key in ids]].mean())
            for domain in sorted(set(domains.values()))
        },
        "per_domain_mean_kld": {
            arm: {domain: float(values[arm][[domains[key] == domain for key in ids]].mean())
                  for domain in sorted(set(domains.values()))}
            for arm in ARMS
        },
    }
    strict = comparison["mean_delta_kld"] < 0 and interval[1] < 0
    return {
        "schema": SCHEMA,
        "role": "conditional-fit",
        "windows": 32,
        "decision": "pass" if comparison["mean_delta_kld"] < 0 else "fail",
        "decision_rule": "coupled_full mean true-decode KLD below identity_full on the same image and runtime; BCa reported, nonblocking",
        "strict_claim": strict,
        "strict_claim_rule": "mean paired delta negative and paired BCa upper bound below zero",
        "bootstrap": {"unit": "window", "replicates": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "arms": {arm: {"conditions": conditions[arm],
                       "true_decode_mean_kld": float(values[arm].mean()),
                       "including_prefill_mean_kld": float(all_rows[arm].mean()),
                       "true_decode_window_bca95": list(bca_mean_interval(
                           values[arm], values[arm][indices].mean(axis=1)))}
                 for arm in ARMS},
        "comparison": comparison,
        "claim_boundary": "Already-opened CF32 development evidence, not final qualification; input provenance and runtime installation require separate receipts",
        "isa_cost": "P8 E4M3 mxf8f6f4 has twice the NVFP4 MMA issue count; no speed claim",
    }
