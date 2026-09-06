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


def payload_bytes(bits: int) -> int:
    return WEIGHTS_PER_LAYER * (4 * bits + 1) // 32


def file_bytes_estimate(bits: int) -> int:
    return payload_bytes(bits) + COUPLED_METADATA_BYTES_PER_LAYER + 4 * HEADER_BYTES_PER_RANK_APPROX


def load_damage(receipts: dict[int, Path], *, metric: str = "damage_sum") -> dict[int, dict[int, float]]:
    damage: dict[int, dict[int, float]] = {}
    for layer in LAYERS:
        if layer not in receipts:
            raise ValueError(f"missing damage receipt for layer {layer}")
        value = json.loads(Path(receipts[layer]).read_text())
        if value.get("schema") != "glm53.p8-layer-rate-damage.v1" or value.get("layer") != layer:
            raise ValueError(f"damage receipt schema/layer differs: {receipts[layer]}")
        rates = value["rates"]
        if set(rates) != {"3", "4", "5"}:
            raise ValueError(f"layer {layer} receipt lacks all three rates")
        damage[layer] = {int(k): float(v[metric]) for k, v in rates.items()}
    return damage


def solve(damage: dict[int, dict[int, float]], *, budget_bytes: int = IDENTITY_FILE_BYTES,
          metadata_bytes_per_layer: int = COUPLED_METADATA_BYTES_PER_LAYER + 4 * HEADER_BYTES_PER_RANK_APPROX) -> dict:
    """Exact DP over the net rate offset d = #K5 - #K3 minimizing total damage."""
    layers = sorted(damage)
    n = len(layers)
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
    marginals = {layer: {"downgrade_cost": damage[layer][3] - damage[layer][4],
                         "upgrade_gain": damage[layer][4] - damage[layer][5]} for layer in layers}
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
        "total_damage": total,
        "uniform_k4_damage": uniform,
        "predicted_relative_damage_change": (total - uniform) / uniform if uniform > 0 else None,
        "estimated_total_file_bytes": total_bytes,
        "bytes_under_budget": budget_bytes - total_bytes,
        "marginals": {str(layer): marginals[layer] for layer in layers},
        "shapley_note": "additive proxy objective: each layer's Shapley value for its rate change equals its marginal damage change; per-expert psi shares live in the damage receipts",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path, required=True,
                        help="directory holding damage-layer-LLL.json for every layer 3..44")
    parser.add_argument("--metric", default="damage_sum", choices=("damage_sum", "damage_mean_per_token", "relative_to_routed_output"))
    parser.add_argument("--budget-bytes", type=int, default=IDENTITY_FILE_BYTES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    receipts = {layer: args.receipt_dir / f"damage-layer-{layer:03d}.json" for layer in LAYERS}
    damage = load_damage(receipts, metric=args.metric)
    result = solve(damage, budget_bytes=args.budget_bytes)
    result["metric"] = args.metric
    result["receipts"] = {str(layer): str(path.resolve()) for layer, path in receipts.items()}
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": result["counts"], "predicted_relative_damage_change": result["predicted_relative_damage_change"],
                      "bytes_under_budget": result["bytes_under_budget"]}))


if __name__ == "__main__":
    main()
