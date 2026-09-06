import json
from pathlib import Path

import pytest
import torch

from glm53_nvfp4 import p8_layer_rate_allocation as alloc
from glm53_nvfp4.p8_layer_rate_damage import shapley_split


def test_rate_step_and_budget_arithmetic_match_the_checkpoint():
    assert alloc.payload_bytes(4) * 42 == 161_715_585_024
    assert alloc.RATE_STEP_BYTES == 905_969_664
    assert alloc.payload_bytes(5) - alloc.payload_bytes(4) == alloc.RATE_STEP_BYTES
    assert alloc.payload_bytes(4) - alloc.payload_bytes(3) == alloc.RATE_STEP_BYTES


def _damage(seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    damage = {}
    for layer in alloc.LAYERS:
        k4 = float(torch.rand(1, generator=gen).item() + 0.5)
        k3 = k4 * float(1.5 + torch.rand(1, generator=gen).item())
        k5 = k4 * float(0.3 + 0.4 * torch.rand(1, generator=gen).item())
        damage[layer] = {3: k3, 4: k4, 5: k5}
    return damage


def test_solver_respects_same_size_budget_and_beats_uniform():
    damage = _damage()
    result = alloc.solve(damage)
    counts = result["counts"]
    assert counts["3"] >= counts["5"] + 1, "coupled metadata must be paid by at least one extra K3 layer"
    assert result["bytes_under_budget"] >= 0
    assert result["total_damage"] <= result["uniform_k4_damage"]
    assert sum(counts.values()) == 42
    assert set(result["assignment"]) == {str(layer) for layer in alloc.LAYERS}
    # Brute-force check on a reduced problem: 6 layers, exhaustive search.
    small = {layer: damage[layer] for layer in list(alloc.LAYERS)[:6]}
    budget = 6 * (alloc.payload_bytes(4) + alloc.COUPLED_METADATA_BYTES_PER_LAYER + 4 * alloc.HEADER_BYTES_PER_RANK_APPROX) - 1
    exact = alloc.solve(small, budget_bytes=budget)
    import itertools
    best = None
    for rates in itertools.product(alloc.RATES, repeat=6):
        offset = sum(r - 4 for r in rates)
        if offset > -1:
            continue
        total = sum(small[layer][r] for layer, r in zip(small, rates))
        if best is None or total < best:
            best = total
    assert abs(exact["total_damage"] - best) < 1e-12


def test_solver_raises_when_budget_is_infeasible():
    damage = _damage()
    with pytest.raises(ValueError, match="no feasible"):
        alloc.solve(damage, budget_bytes=1)


def test_load_damage_rejects_missing_rates(tmp_path: Path):
    receipts = {}
    for layer in alloc.LAYERS:
        path = tmp_path / f"damage-layer-{layer:03d}.json"
        rates = {"3": {"damage_sum": 3.0}, "4": {"damage_sum": 2.0}, "5": {"damage_sum": 1.0}}
        if layer == 10:
            rates.pop("5")
        path.write_text(json.dumps({"schema": "glm53.p8-layer-rate-damage.v1", "layer": layer, "rates": rates}))
        receipts[layer] = path
    with pytest.raises(ValueError, match="lacks all three rates"):
        alloc.load_damage(receipts)


def test_shapley_split_closes_to_half_squared_damage():
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(5, 8, 16, generator=gen)
    ids = torch.randint(0, 288, (5, 8), generator=gen)
    total, psi = shapley_split(z, ids)
    assert torch.allclose(total, z.double().sum(dim=1))
    assert abs(psi.sum().item() - 0.5 * total.square().sum().item()) < 1e-9
