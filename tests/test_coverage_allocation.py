import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from glm53_nvfp4 import coverage_allocation as cov


def _brute_force_shapley(a: np.ndarray, c: float) -> np.ndarray:
    """Shapley by definition: average marginal contribution over all orderings."""
    n = a.shape[0]
    phi = np.zeros(n)
    orders = list(itertools.permutations(range(n)))
    for order in orders:
        present = np.zeros(n, dtype=bool)
        before = 0.0
        for player in order:
            present[player] = True
            after = cov.coverage_value(a, present, c)
            phi[player] += after - before
            before = after
    return phi / len(orders)


def test_path_integral_matches_the_definition_of_the_shapley_value():
    a = np.array([0.30, 0.10, 0.55, 0.02, 0.44])
    c = 0.037
    phi = cov.shapley_values(a, c)
    brute = _brute_force_shapley(a, c)
    assert np.allclose(phi, brute, rtol=1e-10, atol=1e-14)


def test_shares_sum_exactly_to_the_grand_coalition_value():
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = int(rng.integers(2, 30))
        a = rng.random(n) * 0.9 + 0.001
        c = float(rng.random() * 0.1 + 0.001)
        phi = cov.shapley_values(a, c)
        total = cov.grand_coalition(a, c)
        assert abs(phi.sum() - total) <= 1e-12 * max(1.0, abs(total))
        assert np.all(phi >= 0)
    # A saturating layer takes its share without breaking efficiency.
    a = np.array([1.0, 0.4, 0.1])
    phi = cov.shapley_values(a, 0.05)
    assert abs(phi.sum() - cov.grand_coalition(a, 0.05)) < 1e-12


def test_fit_recovers_planted_break_rates_from_about_l_configurations():
    rng = np.random.default_rng(5)
    n = 12
    a_true = rng.random(n) * 0.4 + 0.02
    c_true = 0.06
    rows = [np.eye(n, dtype=bool)[i] for i in range(n)]          # singletons
    rows.append(np.ones(n, dtype=bool))                           # grand coalition
    for _ in range(6):                                            # a few random subsets
        rows.append(rng.random(n) < 0.5)
    design = np.array(rows)
    measured = cov.coverage_values(a_true, design, c_true)
    fit = cov.fit_coverage(design, measured)
    assert fit["r_squared"] > 0.9999
    # The data pins down the fitted values, the first-order products c*a_i and the Shapley
    # shares; c and a individually are only jointly identified (f(S) ~ c*sum a_i when the
    # break-rates are small), so the parameters themselves get the looser tolerance.
    assert np.allclose(fit["predicted"], measured, atol=1e-6)
    assert np.allclose(fit["first_order_coefficients"], c_true * a_true, rtol=5e-3)
    phi = cov.shapley_values(fit["break_rates"], fit["c"])
    assert np.allclose(phi, cov.shapley_values(a_true, c_true), rtol=5e-3)
    held_out = np.random.default_rng(99).random((200, n)) < 0.4
    assert np.allclose(cov.coverage_values(fit["break_rates"], held_out, fit["c"]),
                       cov.coverage_values(a_true, held_out, c_true), atol=1e-5)
    assert np.allclose(fit["break_rates"], a_true, rtol=0.15)
    # L+1 configurations is the documented minimum; fewer must fail closed.
    with pytest.raises(ValueError, match="at least"):
        cov.fit_coverage(design[:n], measured[:n])
    with pytest.raises(ValueError, match="non-negative"):
        cov.fit_coverage(design, -measured)


def test_certificate_is_zero_for_one_layer_and_grows_with_saturation():
    single = cov.certificate(np.array([0.3]), 0.05, samples=20_000)
    assert single["order_ge2_share"] < 1e-9, "a one-layer game has no interactions"
    mild = cov.certificate(np.array([0.01] * 12), 0.05, samples=60_000)["order_ge2_share"]
    heavy = cov.certificate(np.array([0.6] * 12), 0.05, samples=60_000)["order_ge2_share"]
    assert mild < 1e-3 < heavy, "near-linear coverage is additive; saturated coverage is not"
    assert heavy < 1.0


def test_analyse_end_to_end_and_cli(tmp_path: Path):
    rng = np.random.default_rng(3)
    layers = list(range(3, 15))
    n = len(layers)
    a_true = rng.random(n) * 0.3 + 0.01
    rows = [np.eye(n, dtype=bool)[i] for i in range(n)] + [np.ones(n, dtype=bool)]
    design = np.array(rows)
    measured = cov.coverage_values(a_true, design, 0.04)
    result = cov.analyse(layers, design, measured)
    assert result["shapley_closure_error"] < 1e-9
    assert abs(result["shapley_sum"] - result["grand_coalition_value"]) < 1e-12
    assert sorted(result["ranking_worst_first"]) == layers
    assert result["ranking_worst_first"][0] == layers[int(np.argmax(a_true))]
    payload = {"layers": layers, "configurations": [
        {"set": [layers[i] for i in np.flatnonzero(row)], "loss_increase": float(value), "label": f"cfg{k}"}
        for k, (row, value) in enumerate(zip(design, measured))]}
    src = tmp_path / "m.json"
    src.write_text(json.dumps(payload))
    out = tmp_path / "out.json"
    assert cov.main(["--measurements", str(src), "--output", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["schema"] == cov.SCHEMA
    assert math.isclose(sum(written["shapley"].values()), written["grand_coalition_value"], rel_tol=1e-9)
    assert len(written["break_rates"]) == n
