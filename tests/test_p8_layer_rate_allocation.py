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
    partial = alloc.load_damage(receipts, require_all_rates=False)
    assert partial[10] == {3: 3.0, 4: 2.0}


def test_solver_keeps_unscored_layers_at_k4():
    damage = _damage()
    full = alloc.solve(damage)
    # Only the layers the full solution moved stay eligible; everything else is K4-only.
    moved = {int(k) for k, v in full["assignment"].items() if v != 4}
    partial = {layer: (dict(rates) if layer in moved else {4: rates[4]}) for layer, rates in damage.items()}
    result = alloc.solve(partial)
    assert result["assignment"] == full["assignment"]
    assert all(result["eligible_rates"][str(layer)] == [4] for layer in alloc.LAYERS if layer not in moved)
    assert result["marginals"][str(next(iter(set(alloc.LAYERS) - moved)))] == {"downgrade_cost": None, "upgrade_gain": None}
    # Without any K3-eligible layer the same-size budget cannot be met: the coupled metadata
    # must be paid by at least one K3 layer, so the solver fails closed instead of installing K5.
    k5_only = {layer: {4: rates[4], 5: rates[5]} for layer, rates in damage.items()}
    with pytest.raises(ValueError, match="no feasible"):
        alloc.solve(k5_only)
    # One K3-eligible layer is enough, and K5 can never be assigned to a layer scored only at K4.
    k5_only[3] = dict(damage[3])
    result = alloc.solve(k5_only)
    assert result["assignment"]["3"] == 3
    assert all(v in (4, 5) for k, v in result["assignment"].items() if k != "3")


def _write_receipts(directory: Path, damage, rates_per_layer):
    directory.mkdir(parents=True, exist_ok=True)
    for layer, rates in rates_per_layer.items():
        (directory / f"damage-layer-{layer:03d}.json").write_text(json.dumps({
            "schema": "glm53.p8-layer-rate-damage.v1", "layer": layer,
            "rates": {str(r): {"damage_sum": damage[layer][r]} for r in rates}}))


def test_preliminary_pass_estimates_and_widens_candidates(tmp_path: Path):
    damage = _damage()
    k4_only = tmp_path / "k4-only"
    _write_receipts(k4_only, damage, {layer: (4,) for layer in alloc.LAYERS})
    out = tmp_path / "prelim.json"
    alloc.main(["--receipt-dir", str(tmp_path / "candidates"), "--k4-only-dir", str(k4_only), "--output", str(out),
                "--assume-ratio", "3=3.76", "--assume-ratio", "5=0.285", "--candidate-margin", "2", "--min-candidates", "3"])
    prelim = json.loads(out.read_text())
    assert prelim["mode"] == "preliminary-estimated" and prelim["installable"] is False
    assert prelim["estimated_rates"]["10"] == [3, 5] and prelim["measured_rates"]["10"] == [4]
    cands = prelim["candidates"]
    assert not set(cands["3"]) & set(cands["5"])
    ranked = sorted(alloc.LAYERS, key=lambda layer: -damage[layer][4])
    assert set(cands["5"]) == set(ranked[:len(cands["5"])]), "K5 candidates are the highest-K4-damage layers"
    assert set(cands["3"]) == set(ranked[-len(cands["3"]):]), "K3 candidates are the lowest-K4-damage layers"
    assert len(cands["5"]) >= max(prelim["counts"]["5"] + 2, 3) and len(cands["3"]) >= max(prelim["counts"]["3"] + 2, 4)
    # Final pass: candidate receipts carry the measured rate, everything else falls back to K4-only.
    candidates = tmp_path / "candidates"
    _write_receipts(candidates, damage, {**{layer: (4, 5) for layer in cands["5"]}, **{layer: (3, 4) for layer in cands["3"]}})
    final = tmp_path / "final.json"
    alloc.main(["--receipt-dir", str(candidates), "--k4-only-dir", str(k4_only), "--output", str(final)])
    result = json.loads(final.read_text())
    assert result["mode"] == "measured" and result["installable"] is True
    assert all(result["assignment"][str(layer)] == 4 for layer in alloc.LAYERS if layer not in set(cands["3"]) | set(cands["5"]))
    assert all(result["assignment"][str(layer)] in (4, 5) for layer in cands["5"])
    assert all(result["assignment"][str(layer)] in (3, 4) for layer in cands["3"])
    assert result["counts"]["3"] >= result["counts"]["5"] + 1 and result["bytes_under_budget"] >= 0
    assert result["receipts"]["10"].endswith("k4-only/damage-layer-010.json") or 10 in set(cands["3"]) | set(cands["5"])


def test_shapley_split_closes_to_half_squared_damage():
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(5, 8, 16, generator=gen)
    ids = torch.randint(0, 288, (5, 8), generator=gen)
    total, psi = shapley_split(z, ids)
    assert torch.allclose(total, z.double().sum(dim=1))
    assert abs(psi.sum().item() - 0.5 * total.square().sum().item()) < 1e-9
