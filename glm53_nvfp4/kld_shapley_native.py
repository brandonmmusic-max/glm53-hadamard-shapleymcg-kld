"""Direct teacher-KLD Shapley design and analysis for native P8 allocations.

This module deliberately does not consume local NMSE as an allocation value.
Every marginal is computed from matched end-to-end teacher-KLD windows.  The
default game upgrades whole routed layers from a K4 procedural-MCG P8 stream
to scalar MXFP8 E4M3 while keeping the same native ``mxf8f6f4`` MMA family.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


ROUTED_LAYERS = tuple(range(3, 45))


def coalition_id(layers: Iterable[int]) -> str:
    payload = ",".join(map(str, sorted(layers))).encode()
    return "kld-c-" + hashlib.sha256(payload).hexdigest()[:16]


def allocation_slots(
    units: int,
    *,
    base_bpw: float = 4.25,
    upgrade_bpw: float = 8.25,
    budget_bpw: float = 6.0,
) -> int:
    """Largest equal-size whole-layer upgrade count within the rate budget."""
    if units <= 0:
        raise ValueError("units must be positive")
    if not (0 < base_bpw < budget_bpw < upgrade_bpw):
        raise ValueError("rates must satisfy 0 < base < budget < upgrade")
    return int(math.floor((budget_bpw - base_bpw) * units / (upgrade_bpw - base_bpw)))


def build_design(
    seed: int,
    *,
    layers: Iterable[int] = ROUTED_LAYERS,
    base_bpw: float = 4.25,
    upgrade_bpw: float = 8.25,
    budget_bpw: float = 6.0,
    game: str = "p8-k4-to-scalar-mxfp8",
    role: str = "conditional-fit",
    upgrade_slots_override: int | None = None,
) -> dict:
    units = tuple(int(layer) for layer in layers)
    if not units or len(set(units)) != len(units):
        raise ValueError("layers must be non-empty and unique")
    if any(layer not in ROUTED_LAYERS for layer in units):
        raise ValueError("layer is outside GLM routed layers 3..44")
    if upgrade_slots_override is None:
        slots = allocation_slots(
            len(units), base_bpw=base_bpw, upgrade_bpw=upgrade_bpw, budget_bpw=budget_bpw
        )
    else:
        slots = int(upgrade_slots_override)
        if not 0 <= slots <= len(units):
            raise ValueError("upgrade slot override is outside the unit count")
    rng = np.random.default_rng(seed)
    first = list(map(int, rng.permutation(units)))
    orders = [first, list(reversed(first))]
    coalitions: dict[str, list[int]] = {}
    paths = []
    for permutation_index, order in enumerate(orders):
        current: set[int] = set()
        ids = [coalition_id(current)]
        coalitions[ids[0]] = []
        for layer in order:
            current.add(layer)
            cid = coalition_id(current)
            coalitions[cid] = sorted(current)
            ids.append(cid)
        paths.append(
            {
                "permutation_index": permutation_index,
                "order": order,
                "coalition_ids": ids,
            }
        )
    realized_bpw = base_bpw + slots / len(units) * (upgrade_bpw - base_bpw)
    if game == "matched-gptq-to-p8-interaction-pilot":
        tiers = {
            "base": {
                "name": "decoded full-Hessian GPTQ NVFP4 control",
                "stored_bpw": base_bpw,
                "weights": "exact decoded reconstruction of the matched 4.5-bpw NVFP4 control",
                "compute": "BF16 overlay pilot only",
            },
            "upgrade": {
                "name": "decoded K4 procedural-MCG P8 candidate",
                "stored_bpw": upgrade_bpw,
                "weights": "exact decoded reconstruction of the 4.25-bpw P8 stream",
                "compute": "BF16 overlay pilot only",
            },
        }
        isa = "This pilot estimates interaction in matched BF16 overlays and makes no kernel-speed claim; the physical P8 endpoint uses mxf8f6f4 at twice NVFP4 MMA issue count."
        next_gate = "use the interaction result to validate or reject the estimator, then measure the native P8-to-MXFP8 6-bpw game"
        final_rate_gate = "pilot only; no 6-bpw allocation or physical-rate claim is authorized"
    else:
        tiers = {
            "base": {
                "name": "K4 procedural-MCG P8",
                "stored_bpw": base_bpw,
                "weights": "four trellis edge bits per weight plus one UE8M0 byte per K32",
                "compute": "decode to fully-scaled E4M3 in the MMA warp; identity SFB; mxf8f6f4",
            },
            "upgrade": {
                "name": "scalar MXFP8 E4M3",
                "stored_bpw": upgrade_bpw,
                "weights": "one E4M3 byte per weight plus one UE8M0 byte per K32",
                "compute": "native E4M3 mxf8f6f4",
            },
        }
        isa = "P8 and scalar MXFP8 use mxf8f6f4 at twice the MMA issue count of NVFP4; this is the quality product, not the P4 speed product."
        next_gate = "build and measure the selected allocation as one endpoint; Shapley values alone are not a quality claim"
        final_rate_gate = "exact physical receipt, including every boundary/policy byte, must be <= 6.0 bpw"
    return {
        "schema": "glm53-p8.direct-kld-native-shapley-design.v1",
        "seed": seed,
        "game": game,
        "role": role,
        "estimator": "one seeded permutation and its antithetic reverse",
        "unit": "whole routed MoE layer",
        "layers": list(units),
        "permutations": paths,
        "coalitions": coalitions,
        "unique_coalitions": len(coalitions),
        "value": "equal-window end-to-end teacher-KLD reduction when the layer is upgraded in its measured coalition context",
        "value_exclusions": [
            "routed-output NMSE is not an allocation value",
            "weight NMSE is not an allocation value",
            "LDLQ and BlockLDLQ are not used",
        ],
        "tiers": tiers,
        "allocation": {
            "budget_bpw": budget_bpw,
            "upgrade_slots": slots,
            "base_slots": len(units) - slots,
            "realized_weight_bpw_before_boundary_metadata": realized_bpw,
            "headroom_bpw_for_boundary_metadata": budget_bpw - realized_bpw,
            "final_gate": final_rate_gate,
        },
        "isa_cost_statement": isa,
        "next_gate": next_gate,
        "development_gate": "window and domain intervals are reported controls; ranking uses preregistered mean direct KLD marginal",
        "strict_claim_gate": "a final endpoint beats its matched control only if mean paired KLD delta is negative and the paired BCa upper bound is below zero",
        "protected_boundary": "design and adaptive execution must not open selection, confirmation, final, or the reserved 28 confirmation logits",
        "ldlq": False,
    }


def _load_run(path: Path, expected_role: str) -> tuple[dict[str, float], dict[str, str], dict]:
    run = json.loads(path.read_text())
    if run.get("status") != "complete" or run.get("role") != expected_role:
        raise RuntimeError(f"incomplete or wrong-role KLD run: {path}")
    windows = run.get("windows")
    if not isinstance(windows, dict) or not windows:
        raise RuntimeError(f"run lacks window-level KLD: {path}")
    values: dict[str, float] = {}
    domains: dict[str, str] = {}
    for window_id, row in windows.items():
        value = float(row["mean_kld"])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite KLD for {window_id}: {path}")
        values[window_id] = value
        domains[window_id] = str(row["domain"])
    return values, domains, run


def analyze(
    design_path: Path,
    run_root: Path,
    *,
    run_prefix: str,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int = 20260966,
) -> dict:
    design = json.loads(design_path.read_text())
    if design.get("schema") != "glm53-p8.direct-kld-native-shapley-design.v1":
        raise RuntimeError("unsupported direct-KLD Shapley design")
    if design.get("ldlq") is not False:
        raise RuntimeError("design does not preserve the no-LDLQ boundary")

    runs: dict[str, dict[str, float]] = {}
    evidence = []
    expected_ids: list[str] | None = None
    expected_domains: dict[str, str] | None = None
    for cid in design["coalitions"]:
        path = run_root / f"run-{run_prefix}-{cid}.json"
        values, domains, run = _load_run(path, design["role"])
        ids = sorted(values)
        if expected_ids is None:
            expected_ids, expected_domains = ids, domains
        elif ids != expected_ids or domains != expected_domains:
            raise RuntimeError(f"coalition window/domain mismatch: {path}")
        runs[cid] = values
        evidence.append(
            {
                "coalition_id": cid,
                "layers": design["coalitions"][cid],
                "run": str(path.resolve()),
                "run_sha256": sha256_file(path),
                "mean_kld": float(np.mean(list(values.values()))),
                "config_id": run.get("config_id"),
            }
        )
    assert expected_ids is not None and expected_domains is not None

    samples: dict[int, list[np.ndarray]] = {int(layer): [] for layer in design["layers"]}
    contextual_marginals: dict[int, list[dict]] = {
        int(layer): [] for layer in design["layers"]
    }
    path_efficiency = []
    running_rankings = []
    for permutation in design["permutations"]:
        ids = permutation["coalition_ids"]
        empty = np.asarray([runs[ids[0]][wid] for wid in expected_ids], dtype=np.float64)
        full = np.asarray([runs[ids[-1]][wid] for wid in expected_ids], dtype=np.float64)
        path_sum = np.zeros_like(empty)
        for index, layer in enumerate(permutation["order"]):
            before = np.asarray([runs[ids[index]][wid] for wid in expected_ids], dtype=np.float64)
            after = np.asarray([runs[ids[index + 1]][wid] for wid in expected_ids], dtype=np.float64)
            marginal = before - after
            samples[int(layer)].append(marginal)
            contextual_marginals[int(layer)].append(
                {
                    "permutation_index": permutation["permutation_index"],
                    "before_coalition_id": ids[index],
                    "after_coalition_id": ids[index + 1],
                    "mean_marginal_kld_reduction": float(marginal.mean()),
                    "per_window_marginal_kld_reduction": {
                        wid: float(value)
                        for wid, value in zip(expected_ids, marginal, strict=True)
                    },
                }
            )
            path_sum += marginal
        residual = (empty - full) - path_sum
        path_efficiency.append(
            {
                "permutation_index": permutation["permutation_index"],
                "empty_mean_kld": float(empty.mean()),
                "full_mean_kld": float(full.mean()),
                "mean_total_reduction": float((empty - full).mean()),
                "max_abs_window_efficiency_residual": float(np.abs(residual).max()),
            }
        )
        current = {
            layer: float(np.mean(np.stack(arrays, axis=0)))
            for layer, arrays in samples.items()
            if arrays
        }
        running_rankings.append(
            sorted(current, key=lambda layer: (-current[layer], layer))[
                : design["allocation"]["upgrade_slots"]
            ]
        )

    rng = np.random.default_rng(bootstrap_seed)
    choices = rng.integers(
        0, len(expected_ids), size=(bootstrap_replicates, len(expected_ids))
    )
    rows = []
    for layer in map(int, design["layers"]):
        per_window = np.mean(np.stack(samples[layer], axis=0), axis=0)
        bootstrap = per_window[choices].mean(axis=1)
        ci = bca_mean_interval(per_window, bootstrap)
        rows.append(
            {
                "layer": layer,
                "mean_marginal_kld_reduction": float(per_window.mean()),
                "delta_ci95_bca": [float(ci[0]), float(ci[1])],
                "window_wins": int((per_window > 0).sum()),
                "windows": len(expected_ids),
                "permutation_samples": len(samples[layer]),
                "contextual_marginals": contextual_marginals[layer],
                "per_window_marginal_kld_reduction": {
                    wid: float(value) for wid, value in zip(expected_ids, per_window, strict=True)
                },
                "domain_mean_marginal_kld_reduction": {
                    domain: float(
                        np.mean(
                            [
                                per_window[i]
                                for i, wid in enumerate(expected_ids)
                                if expected_domains[wid] == domain
                            ]
                        )
                    )
                    for domain in sorted(set(expected_domains.values()))
                },
            }
        )
    slots = int(design["allocation"]["upgrade_slots"])
    ranked = sorted(rows, key=lambda row: (-row["mean_marginal_kld_reduction"], row["layer"]))
    selected = sorted(row["layer"] for row in ranked[:slots])
    first = set(running_rankings[0]) if running_rankings else set()
    final = set(running_rankings[-1]) if running_rankings else set()
    overlap = len(first & final)
    empty_id = design["permutations"][0]["coalition_ids"][0]
    full_id = design["permutations"][0]["coalition_ids"][-1]
    empty_values = np.asarray([runs[empty_id][wid] for wid in expected_ids])
    full_values = np.asarray([runs[full_id][wid] for wid in expected_ids])
    coalition_comparisons = []
    for cid, layers in design["coalitions"].items():
        candidate = np.asarray([runs[cid][wid] for wid in expected_ids])
        delta = candidate - empty_values
        ci = bca_mean_interval(delta, delta[choices].mean(axis=1))
        coalition_comparisons.append(
            {
                "coalition_id": cid,
                "layers": layers,
                "mean_kld": float(candidate.mean()),
                "mean_delta_kld_vs_empty": float(delta.mean()),
                "relative_arithmetic_mean_improvement_vs_empty": float(
                    1.0 - candidate.mean() / empty_values.mean()
                ),
                "delta_ci95_bca": [float(ci[0]), float(ci[1])],
                "window_wins_vs_empty": int((delta < 0).sum()),
            }
        )
    coalition_comparisons.sort(
        key=lambda row: (row["mean_kld"], row["coalition_id"])
    )
    observed_best = next(
        row for row in coalition_comparisons if row["coalition_id"] != empty_id
    )
    return {
        "schema": "glm53-p8.direct-kld-native-shapley-analysis.v1",
        "design": str(design_path.resolve()),
        "design_sha256": sha256_file(design_path),
        "run_prefix": run_prefix,
        "role": design["role"],
        "windows": len(expected_ids),
        "baseline_empty_mean_kld": float(empty_values.mean()),
        "full_upgrade_mean_kld": float(full_values.mean()),
        "full_game_mean_kld_reduction": float((empty_values - full_values).mean()),
        "observed_coalition_comparisons": coalition_comparisons,
        "observed_best_nonempty_coalition": {
            **observed_best,
            "claim_boundary": "adaptive opened-role diagnostic; not qualification",
        },
        "per_layer": rows,
        "selected_upgrade_layers": selected,
        "selected_base_layers": sorted(set(design["layers"]) - set(selected)),
        "allocation": design["allocation"],
        "convergence_control": {
            "selected_after_each_permutation": running_rankings,
            "first_vs_final_overlap": overlap,
            "first_vs_final_jaccard": (
                overlap / len(first | final) if first or final else 1.0
            ),
        },
        "path_efficiency": path_efficiency,
        "bootstrap": {
            "method": "paired BCa over windows",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval_is_nonblocking_for_development": True,
        },
        "next_gate": "build and measure the selected allocation as one endpoint; Shapley values alone are not a quality claim",
        "strict_claim_gate": design["strict_claim_gate"],
        "ldlq_used": False,
        "evidence": evidence,
    }
