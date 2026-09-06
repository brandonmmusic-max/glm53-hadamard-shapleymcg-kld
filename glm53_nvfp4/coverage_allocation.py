"""Coverage-model attribution of a measured KLD change over layer sets.

Follows Hill, "Saturation Makes Quantization Error Additive: A Coverage Model with a
Certificate" (arXiv:2607.12266).  The characteristic function is the measured loss
increase of a quantized layer set, modelled as

    f(S) = c * (1 - prod_{i in S} (1 - a_i))                        (coverage model)

with one break-rate ``a_i`` per layer plus the scale ``c``: L+1 parameters fitted by
joint least squares from about L measured configurations, not from an enumeration of
the 2^L coalitions.

Attribution.  The Harsanyi dividends of the coverage game are exact::

    m(T) = c * (-1)^{|T|+1} * prod_{i in T} a_i

so the Shapley value phi_i = sum_{T ni i} m(T)/|T| collapses to a one-dimensional
integral (derived here; it is the Aumann-Shapley path integral of the same game)::

    phi_i = c * a_i * integral_0^1 prod_{j != i} (1 - t * a_j) dt

and these sum **exactly** to the grand-coalition value::

    sum_i phi_i = c * (1 - prod_i (1 - a_i)) = f(N)

because c * sum_i a_i * prod_{j!=i}(1 - t a_j) is d/dt of -c * prod_j (1 - t a_j).
That identity is the efficiency the allocation needs: every layer's share is a real
part of the one measured number, and the shares leave nothing over.

Certificate.  The coverage model is not additive; the paper reports the order->=2
share, the part of the variance of f no additive model can explain, as a certificate
attached to every result.  ``certificate`` computes it directly: under a deployment
distribution over layer sets it takes the best additive (first-order) fit in the
least-squares sense and returns its residual variance divided by the variance of f.
A small share means the per-layer numbers can be trusted on their own; a large share
means interactions matter and only the measured joint configuration is safe to quote.

Nothing here estimates a loss; it only decomposes losses that were measured.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "glm53.coverage-allocation.v1"
GAUSS_POINTS = 64


def coverage_value(a: np.ndarray, subset: np.ndarray, c: float) -> float:
    """f(S) for a boolean membership vector."""
    return float(c * (1.0 - np.prod(1.0 - a[subset])))


def coverage_values(a: np.ndarray, design: np.ndarray, c: float) -> np.ndarray:
    """f(S) for a design matrix of boolean rows."""
    kept = np.where(design, 1.0 - a[None, :], 1.0)
    return c * (1.0 - kept.prod(axis=1))


def fit_coverage(design: np.ndarray, measured: np.ndarray, *, max_rate: float = 0.999,
                 iterations: int = 400, seed: int = 0) -> dict:
    """Joint least squares for (c, a) from measured configurations.

    ``design`` is (n_configs, n_layers) boolean, ``measured`` the loss increase of each
    configuration.  Optimised in log space for the break-rates with a projected
    Gauss-Newton style refinement; ``c`` is solved exactly at each step because the model
    is linear in ``c``.
    """
    design = np.asarray(design, dtype=bool)
    measured = np.asarray(measured, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != measured.shape[0]:
        raise ValueError("design and measured disagree in shape")
    if design.shape[0] < design.shape[1] + 1:
        raise ValueError(f"need at least {design.shape[1] + 1} configurations to fit "
                         f"{design.shape[1]} break-rates and the scale, got {design.shape[0]}")
    if np.any(measured < 0):
        raise ValueError("coverage model expects a non-negative loss increase; flip the sign of a benefit game")
    n_layers = design.shape[1]
    rng = np.random.default_rng(seed)

    def solve_c(a: np.ndarray) -> tuple[float, float]:
        basis = coverage_values(a, design, 1.0)
        denom = float(basis @ basis)
        if denom <= 0:
            return 0.0, float(measured @ measured)
        c = float(basis @ measured / denom)
        residual = measured - c * basis
        return c, float(residual @ residual)

    # Start from the single-layer marginals where they exist, else a common rate.
    singles = {int(np.flatnonzero(row)[0]): value for row, value in zip(design, measured) if row.sum() == 1}
    scale = max(measured.max(), 1e-12)
    a = np.full(n_layers, 0.5, dtype=np.float64)
    for index, value in singles.items():
        a[index] = min(max(value / scale, 1e-6), max_rate)
    best_c, best_loss = solve_c(a)
    step = 0.35
    for iteration in range(iterations):
        improved = False
        for index in range(n_layers):
            current = a[index]
            for trial in (current * (1 - step), current * (1 + step), current + step * (max_rate - current)):
                candidate = float(min(max(trial, 1e-9), max_rate))
                if candidate == current:
                    continue
                a[index] = candidate
                c, loss = solve_c(a)
                if loss < best_loss - 1e-18:
                    best_loss, best_c, improved = loss, c, True
                else:
                    a[index] = current
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
            if iteration % 40 == 39:
                a = np.clip(a * rng.uniform(0.95, 1.05, n_layers), 1e-9, max_rate)
                best_c, best_loss = solve_c(a)
    # Coordinate descent lands near the optimum but the model is weakly identified: in the
    # small-break-rate regime f(S) ~ c * sum_{i in S} a_i, so (c, a) and (c/k, k*a) fit almost
    # equally well.  Polish with a proper nonlinear least squares over log-odds of a, which
    # follows the flat direction instead of crawling across it.
    def residuals(theta: np.ndarray) -> np.ndarray:
        rates = max_rate / (1.0 + np.exp(-theta[1:]))
        return theta[0] * coverage_values(rates, design, 1.0) - measured

    theta0 = np.concatenate([[best_c], np.log(np.clip(a / max_rate, 1e-9, 1 - 1e-9) /
                                              (1.0 - np.clip(a / max_rate, 1e-9, 1 - 1e-9)))])
    try:
        from scipy.optimize import least_squares

        fitted = least_squares(residuals, theta0, method="lm", max_nfev=20_000)
        loss = float(fitted.cost * 2.0)
        if loss < best_loss:
            best_loss = loss
            best_c = float(fitted.x[0])
            a = max_rate / (1.0 + np.exp(-fitted.x[1:]))
    except Exception:  # pragma: no cover - scipy absent or refused to converge
        pass
    predicted = best_c * coverage_values(a, design, 1.0)
    total_variance = float(((measured - measured.mean()) ** 2).sum())
    return {"c": best_c, "break_rates": a.copy(), "residual_sum_of_squares": best_loss,
            "r_squared": float(1.0 - best_loss / total_variance) if total_variance > 0 else None,
            "first_order_coefficients": best_c * a,
            "predicted": predicted, "measured": measured.copy(), "configurations": int(design.shape[0]),
            "identifiability_note": "c and the break-rates are only jointly identified; the products c*a_i "
                                    "(first-order coefficients), the fitted values and the Shapley shares are "
                                    "what the data pins down"}


def shapley_values(a: np.ndarray, c: float, *, points: int = GAUSS_POINTS) -> np.ndarray:
    """Exact Shapley values of the coverage game, by the one-dimensional path integral."""
    a = np.asarray(a, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(points)
    t = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    factors = 1.0 - t[:, None] * a[None, :]           # (points, layers)
    # prod_{j != i} = full product divided by own factor; guard the exact-1.0 break-rate case.
    logs = np.log(np.maximum(factors, 1e-300))
    total = logs.sum(axis=1, keepdims=True)
    others = np.exp(total - logs)
    zero = factors <= 1e-300
    if zero.any():
        for row, col in zip(*np.nonzero(zero)):
            mask = np.ones(a.shape[0], dtype=bool)
            mask[col] = False
            others[row, col] = float(np.prod(factors[row][mask]))
    return c * a * (w[:, None] * others).sum(axis=0)


def grand_coalition(a: np.ndarray, c: float) -> float:
    return float(c * (1.0 - np.prod(1.0 - np.asarray(a, dtype=np.float64))))


def certificate(a: np.ndarray, c: float, *, density: float | np.ndarray = 0.5,
                samples: int = 200_000, seed: int = 20260906) -> dict:
    """Order->=2 share: the variance of f no additive model can explain.

    Layers are included independently with probability ``density`` (scalar or per layer).
    Returns the residual variance of the best least-squares additive fit divided by the
    variance of f, plus that additive fit's coefficients.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    p = np.full(n, float(density)) if np.isscalar(density) else np.asarray(density, dtype=np.float64)
    rng = np.random.default_rng(seed)
    design = rng.random((samples, n)) < p[None, :]
    values = coverage_values(a, design, c)
    x = np.column_stack([np.ones(samples), design.astype(np.float64)])
    coefficients, *_ = np.linalg.lstsq(x, values, rcond=None)
    residual = values - x @ coefficients
    variance = float(values.var())
    share = float((residual ** 2).mean() / variance) if variance > 0 else 0.0
    return {"order_ge2_share": share, "additive_r_squared": 1.0 - share, "variance_of_f": variance,
            "additive_intercept": float(coefficients[0]), "additive_coefficients": coefficients[1:].tolist(),
            "density": p.tolist(), "samples": int(samples)}


def analyse(layers: list[int], design: np.ndarray, measured: np.ndarray, *, density: float = 0.5,
            labels: list[str] | None = None) -> dict:
    fit = fit_coverage(design, measured)
    a, c = fit["break_rates"], fit["c"]
    phi = shapley_values(a, c)
    total = grand_coalition(a, c)
    closure = abs(float(phi.sum()) - total)
    if closure > 1e-9 * max(1.0, abs(total)):
        raise RuntimeError("coverage Shapley values do not close to the grand-coalition value")
    cert = certificate(a, c, density=density)
    order = np.argsort(-phi)
    return {
        "schema": SCHEMA,
        "model": "f(S) = c * (1 - prod_{i in S} (1 - a_i)); Hill arXiv:2607.12266",
        "layers": list(layers),
        "configurations": fit["configurations"],
        "configuration_labels": labels,
        "scale_c": c,
        "break_rates": {str(layer): float(value) for layer, value in zip(layers, a)},
        "fit_r_squared": fit["r_squared"],
        "fit_residual_sum_of_squares": fit["residual_sum_of_squares"],
        "fit_predicted": fit["predicted"].tolist(),
        "fit_measured": fit["measured"].tolist(),
        "shapley": {str(layer): float(value) for layer, value in zip(layers, phi)},
        "shapley_sum": float(phi.sum()),
        "grand_coalition_value": total,
        "shapley_closure_error": closure,
        "ranking_worst_first": [int(layers[i]) for i in order],
        "certificate": cert,
        "efficiency_note": "sum_i shapley_i equals f(N) exactly by construction; the certificate reports the "
                           "order->=2 variance share that no per-layer number can carry",
    }


def _load_measurements(path: Path) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    raw = json.loads(Path(path).read_text())
    layers = [int(x) for x in raw["layers"]]
    index = {layer: i for i, layer in enumerate(layers)}
    rows, values, labels = [], [], []
    for entry in raw["configurations"]:
        row = np.zeros(len(layers), dtype=bool)
        for layer in entry["set"]:
            row[index[int(layer)]] = True
        rows.append(row)
        values.append(float(entry["loss_increase"]))
        labels.append(str(entry.get("label", "")))
    return layers, np.array(rows), np.array(values, dtype=np.float64), labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", type=Path, required=True,
                        help='JSON: {"layers": [...], "configurations": [{"set": [...], "loss_increase": x, "label": "..."}]}')
    parser.add_argument("--density", type=float, default=0.5, help="per-layer inclusion probability for the certificate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    layers, design, measured, labels = _load_measurements(args.measurements)
    result = analyse(layers, design, measured, density=args.density, labels=labels)
    result["measurements_path"] = str(Path(args.measurements).resolve())
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"layers": len(layers), "configurations": result["configurations"],
                      "fit_r_squared": result["fit_r_squared"],
                      "order_ge2_share": result["certificate"]["order_ge2_share"],
                      "grand_coalition_value": result["grand_coalition_value"],
                      "worst_first": result["ranking_worst_first"][:8]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
