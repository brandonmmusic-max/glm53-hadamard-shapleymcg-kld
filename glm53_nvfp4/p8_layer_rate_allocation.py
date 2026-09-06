"""Exact same-size layer-rate allocation from per-layer K3/K4/K5 damage receipts.

Players are the 42 routed layers; each chooses a stored trellis rate in
{3, 4, 5} with coupling kept.  Payload bytes are exactly linear in the rate
(one bit per weight per rate step, 905,969,664 bytes per layer), so the byte
constraint "no larger than the identity K4 checkpoint files" reduces to::

    (#K5 - #K3) * 905,969,664 + coupled_metadata_and_header_bytes <= 0

which for GLM-5.3-Flash means at least one more K3 layer than K5 layers.
The objective is the additive fit-role routed-output damage from
``p8_layer_rate_damage`` receipts.  Under an additive objective the Shapley
value of changing one layer's rate equals its marginal damage change, so the
dynamic program below is exact for the stated game; the end-to-end KLD of the
installed allocation is the only quality claim.

Two passes use this module:

* ``--assume-ratio 3=R3 --assume-ratio 5=R5`` (preliminary): every layer has a
  measured K4 damage; K3/K5 damages that were not measured are filled with
  ``damage_K4 * R`` and the result is a *candidate* list (the preliminary
  assignment widened by ``--candidate-margin`` along the K4-damage ranking),
  never an installable allocation.
* measured (final): K3/K5 damages exist only for the candidate layers that were
  actually encoded and scored; every other layer is eligible for K4 only.  The
  output records per layer which rates were measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = tuple(range(3, 45))
RATES = (3, 4, 5)
WEIGHTS_PER_LAYER = 288 * 3 * 4096 * 2048
RATE_STEP_BYTES = WEIGHTS_PER_LAYER // 8  # one stored bit per weight
IDENTITY_FILE_BYTES = 161_715_707_328  # measured uniform-p8-all42-v1 sidecar files
COUPLED_METADATA_BYTES_PER_LAYER = 4 * 901_408
HEADER_BYTES_PER_RANK_APPROX = 3_272
SCHEMA = "glm53.p8-layer-rate-allocation.v1"
DAMAGE_SCHEMA = "glm53.p8-layer-rate-damage.v1"


def payload_bytes(bits: int) -> int:
    return WEIGHTS_PER_LAYER * (4 * bits + 1) // 32


def file_bytes_estimate(bits: int) -> int:
    return payload_bytes(bits) + COUPLED_METADATA_BYTES_PER_LAYER + 4 * HEADER_BYTES_PER_RANK_APPROX


def read_damage_receipt(path: Path, layer: int, *, metric: str) -> dict[int, float]:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != DAMAGE_SCHEMA or value.get("layer") != layer:
        raise ValueError(f"damage receipt schema/layer differs: {path}")
    rates = {int(k): float(v[metric]) for k, v in value["rates"].items()}
    if 4 not in rates:
        raise ValueError(f"layer {layer} receipt lacks the K4 damage: {path}")
    if not set(rates) <= set(RATES):
        raise ValueError(f"layer {layer} receipt carries an unknown rate: {path}")
    return rates


def load_damage(receipts: dict[int, Path], *, metric: str = "damage_sum",
                require_all_rates: bool = True) -> dict[int, dict[int, float]]:
    damage: dict[int, dict[int, float]] = {}
    for layer in LAYERS:
        if layer not in receipts:
            raise ValueError(f"missing damage receipt for layer {layer}")
        rates = read_damage_receipt(receipts[layer], layer, metric=metric)
        if require_all_rates and set(rates) != set(RATES):
            raise ValueError(f"layer {layer} receipt lacks all three rates")
        damage[layer] = rates
    return damage


def locate_receipts(receipt_dir: Path, k4_only_dir: Path | None) -> dict[int, Path]:
    """Prefer the candidate receipt (K4 + encoded candidate rates); fall back to the K4-only receipt."""
    receipts: dict[int, Path] = {}
    for layer in LAYERS:
        primary = Path(receipt_dir) / f"damage-layer-{layer:03d}.json"
        fallback = Path(k4_only_dir) / f"damage-layer-{layer:03d}.json" if k4_only_dir else None
        if primary.is_file():
            receipts[layer] = primary
        elif fallback is not None and fallback.is_file():
            receipts[layer] = fallback
        else:
            raise ValueError(f"no damage receipt for layer {layer} under {receipt_dir}"
                             + (f" or {k4_only_dir}" if k4_only_dir else ""))
    return receipts


def apply_assumed_ratios(damage: dict[int, dict[int, float]], ratios: dict[int, float]) -> dict[int, list[int]]:
    """Fill unmeasured K3/K5 damages as K4 damage times an assumed ratio; return which rates were estimated."""
    estimated: dict[int, list[int]] = {}
    for layer, rates in damage.items():
        for rate, ratio in sorted(ratios.items()):
            if rate not in rates:
                rates[rate] = rates[4] * ratio
                estimated.setdefault(layer, []).append(rate)
    return estimated


def solve(damage: dict[int, dict[int, float]], *, budget_bytes: int = IDENTITY_FILE_BYTES,
          metadata_bytes_per_layer: int = COUPLED_METADATA_BYTES_PER_LAYER + 4 * HEADER_BYTES_PER_RANK_APPROX) -> dict:
    """Exact DP over the net rate offset d = #K5 - #K3 minimizing total damage.

    A layer is eligible only for the rates present in its damage entry; a layer
    with only K4 measured stays K4.
    """
    layers = sorted(damage)
    n = len(layers)
    for layer in layers:
        if 4 not in damage[layer]:
            raise ValueError(f"layer {layer} has no K4 damage")
    base_bytes = n * (payload_bytes(4) + metadata_bytes_per_layer)
    # Feasible offsets: base_bytes + d * RATE_STEP_BYTES <= budget_bytes.
    max_offset = (budget_bytes - base_bytes) // RATE_STEP_BYTES
    if budget_bytes - base_bytes < 0 and (budget_bytes - base_bytes) % RATE_STEP_BYTES:
        max_offset = -((base_bytes - budget_bytes + RATE_STEP_BYTES - 1) // RATE_STEP_BYTES)
    offset_lo, offset_hi = -n, n
    width = offset_hi - offset_lo + 1
    INF = float("inf")
    best = [INF] * width
    choice: list[list[tuple[int, int] | None]] = []
    best[-offset_lo] = 0.0
    for layer in layers:
        next_best = [INF] * width
        step_choice: list[tuple[int, int] | None] = [None] * width
        for index in range(width):
            if best[index] == INF:
                continue
            offset = index + offset_lo
            for rate in RATES:
                if rate not in damage[layer]:
                    continue
                new_offset = offset + (rate - 4)
                if not offset_lo <= new_offset <= offset_hi:
                    continue
                candidate = best[index] + damage[layer][rate]
                j = new_offset - offset_lo
                if candidate < next_best[j]:
                    next_best[j] = candidate
                    step_choice[j] = (index, rate)
        best = next_best
        choice.append(step_choice)
    feasible = [(best[i], i + offset_lo) for i in range(width) if best[i] < INF and i + offset_lo <= max_offset]
    if not feasible:
        raise ValueError("no feasible allocation under the byte budget")
    total, offset = min(feasible)
    assignment: dict[int, int] = {}
    index = offset - offset_lo
    for layer, step_choice in zip(reversed(layers), reversed(choice)):
        previous, rate = step_choice[index]
        assignment[layer] = rate
        index = previous
    uniform = sum(damage[layer][4] for layer in layers)
    marginals = {layer: {"downgrade_cost": (damage[layer][3] - damage[layer][4]) if 3 in damage[layer] else None,
                         "upgrade_gain": (damage[layer][4] - damage[layer][5]) if 5 in damage[layer] else None}
                 for layer in layers}
    total_bytes = sum(payload_bytes(assignment[layer]) + metadata_bytes_per_layer for layer in layers)
    return {
        "schema": SCHEMA,
        "game": "layers 3..44 choose K3/K4/K5 with the coupled boundary; additive fit-role routed-output damage; exact DP over #K5-#K3",
        "budget_bytes": budget_bytes,
        "base_uniform_k4_bytes": base_bytes,
        "max_net_offset_k5_minus_k3": max_offset,
        "selected_net_offset": offset,
        "assignment": {str(layer): assignment[layer] for layer in layers},
        "counts": {str(rate): sum(1 for layer in layers if assignment[layer] == rate) for rate in RATES},
        "eligible_rates": {str(layer): sorted(damage[layer]) for layer in layers},
        "total_damage": total,
        "uniform_k4_damage": uniform,
        "predicted_relative_damage_change": (total - uniform) / uniform if uniform > 0 else None,
        "estimated_total_file_bytes": total_bytes,
        "bytes_under_budget": budget_bytes - total_bytes,
        "marginals": {str(layer): marginals[layer] for layer in layers},
        "shapley_note": "additive proxy objective: each layer's Shapley value for its rate change equals its marginal damage change; per-expert psi shares live in the damage receipts",
    }


def candidate_sets(damage: dict[int, dict[int, float]], assignment: dict[str, int], *, margin: int,
                   min_candidates: int) -> dict[str, list[int]]:
    """Widen a preliminary assignment along the K4-damage ranking.

    K5 candidates are the highest-K4-damage layers (upgrades pay off where the
    layer already hurts most); K3 candidates are the lowest-K4-damage layers.
    The sets are disjoint; the K5 set wins any overlap.
    """
    ranked = sorted(damage, key=lambda layer: (-damage[layer][4], layer))
    n5 = sum(1 for v in assignment.values() if v == 5)
    n3 = sum(1 for v in assignment.values() if v == 3)
    want5 = min(len(ranked), max(n5 + margin, min_candidates))
    want3 = min(len(ranked), max(n3 + margin, min_candidates + 1))
    k5 = ranked[:want5]
    k3 = [layer for layer in reversed(ranked) if layer not in set(k5)][:want3]
    return {"5": sorted(k5), "3": sorted(k3)}


def parse_ratios(items: list[str]) -> dict[int, float]:
    ratios: dict[int, float] = {}
    for item in items:
        rate, sep, value = item.partition("=")
        if not sep or int(rate) not in (3, 5) or float(value) <= 0:
            raise SystemExit(f"--assume-ratio expects 3=R or 5=R with R>0, got {item!r}")
        ratios[int(rate)] = float(value)
    return ratios


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--receipt-dir", type=Path, required=True,
                        help="directory holding damage-layer-LLL.json (K4 plus any measured candidate rates)")
    parser.add_argument("--k4-only-dir", type=Path,
                        help="fallback directory holding K4-only damage-layer-LLL.json for layers absent from --receipt-dir")
    parser.add_argument("--metric", default="damage_sum", choices=("damage_sum", "damage_mean_per_token", "relative_to_routed_output"))
    parser.add_argument("--budget-bytes", type=int, default=IDENTITY_FILE_BYTES)
    parser.add_argument("--assume-ratio", action="append", default=[],
                        help="preliminary pass only: fill unmeasured K3/K5 damage as K4 damage times this ratio (3=R3, 5=R5)")
    parser.add_argument("--candidate-margin", type=int, default=3)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    receipts = locate_receipts(args.receipt_dir, args.k4_only_dir)
    damage = load_damage(receipts, metric=args.metric, require_all_rates=False)
    measured = {str(layer): sorted(rates) for layer, rates in damage.items()}
    ratios = parse_ratios(args.assume_ratio)
    estimated = apply_assumed_ratios(damage, ratios) if ratios else {}
    result = solve(damage, budget_bytes=args.budget_bytes)
    result["metric"] = args.metric
    result["receipts"] = {str(layer): str(path.resolve()) for layer, path in receipts.items()}
    result["measured_rates"] = measured
    if ratios:
        result["mode"] = "preliminary-estimated"
        result["assumed_ratios"] = {str(k): v for k, v in sorted(ratios.items())}
        result["estimated_rates"] = {str(layer): rates for layer, rates in sorted(estimated.items())}
        result["candidates"] = candidate_sets(damage, result["assignment"], margin=args.candidate_margin,
                                              min_candidates=args.min_candidates)
        result["installable"] = False
        result["note"] = ("preliminary: K3/K5 damages marked in estimated_rates are K4 damage times an assumed ratio; "
                          "use only to choose which layers to encode and score exactly")
    else:
        result["mode"] = "measured"
        result["installable"] = True
        result["note"] = "layers whose eligible_rates lack a rate were never scored at that rate and stay K4 by construction"
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    summary = {"mode": result["mode"], "counts": result["counts"],
               "predicted_relative_damage_change": result["predicted_relative_damage_change"],
               "bytes_under_budget": result["bytes_under_budget"]}
    if ratios:
        summary["candidates"] = result["candidates"]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
