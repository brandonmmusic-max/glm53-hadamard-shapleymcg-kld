import json
from pathlib import Path

import numpy as np

from glm53_nvfp4.kld_shapley_native import allocation_slots, analyze, build_design


def test_native6_rate_and_antithetic_design():
    design = build_design(17)
    assert design == build_design(17)
    assert allocation_slots(42) == 18
    assert design["allocation"]["upgrade_slots"] == 18
    assert design["allocation"]["base_slots"] == 24
    assert np.isclose(
        design["allocation"]["realized_weight_bpw_before_boundary_metadata"],
        5.964285714285714,
    )
    assert design["allocation"]["headroom_bpw_for_boundary_metadata"] > 0
    assert design["unique_coalitions"] == 84
    assert design["permutations"][1]["order"] == list(
        reversed(design["permutations"][0]["order"])
    )
    assert design["ldlq"] is False


def test_direct_kld_analysis_recovers_additive_ranking(tmp_path: Path):
    layers = [3, 19, 20, 22]
    design = build_design(
        5,
        layers=layers,
        base_bpw=4.25,
        upgrade_bpw=8.25,
        budget_bpw=6.0,
        game="matched-p8-vs-gptq-pilot",
    )
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    effects = {3: 0.004, 19: -0.001, 20: 0.003, 22: 0.001}
    windows = {
        "w-general": ("general", 0.04),
        "w-legal": ("legal", 0.05),
        "w-code": ("code", 0.03),
        "w-reasoning": ("reasoning", 0.06),
    }
    for cid, coalition in design["coalitions"].items():
        payload = {
            "status": "complete",
            "role": "conditional-fit",
            "config_id": "matched-test",
            "windows": {
                wid: {
                    "domain": domain,
                    "mean_kld": baseline - sum(effects[layer] for layer in coalition),
                }
                for wid, (domain, baseline) in windows.items()
            },
        }
        payload["mean_of_window_means"] = float(
            np.mean([row["mean_kld"] for row in payload["windows"].values()])
        )
        (tmp_path / f"run-pilot-{cid}.json").write_text(json.dumps(payload))

    result = analyze(
        design_path,
        tmp_path,
        run_prefix="pilot",
        bootstrap_replicates=500,
        bootstrap_seed=9,
    )
    assert result["selected_upgrade_layers"] == [3]
    means = {
        row["layer"]: row["mean_marginal_kld_reduction"]
        for row in result["per_layer"]
    }
    for layer, expected in effects.items():
        assert np.isclose(means[layer], expected)
    for row in result["per_layer"]:
        assert len(row["contextual_marginals"]) == 2
        assert all(
            np.isclose(context["mean_marginal_kld_reduction"], effects[row["layer"]])
            for context in row["contextual_marginals"]
        )
    assert all(
        row["max_abs_window_efficiency_residual"] < 1e-15
        for row in result["path_efficiency"]
    )
    assert result["ldlq_used"] is False
    best = result["observed_best_nonempty_coalition"]
    assert best["layers"] == [3, 19, 20, 22]
    assert np.isclose(best["relative_arithmetic_mean_improvement_vs_empty"], 7 / 45)
    empty = next(
        row
        for row in result["observed_coalition_comparisons"]
        if not row["layers"]
    )
    assert np.isclose(empty["mean_delta_kld_vs_empty"], 0.0)


def test_direct_kld_analysis_rejects_window_mismatch(tmp_path: Path):
    design = build_design(3, layers=[3, 20])
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    for index, cid in enumerate(design["coalitions"]):
        window_id = "wrong" if index == 1 else "w0"
        (tmp_path / f"run-pilot-{cid}.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "role": "conditional-fit",
                    "windows": {
                        window_id: {"domain": "general", "mean_kld": 0.1}
                    },
                }
            )
        )
    try:
        analyze(design_path, tmp_path, run_prefix="pilot", bootstrap_replicates=10)
    except RuntimeError as exc:
        assert "window/domain mismatch" in str(exc)
    else:
        raise AssertionError("window mismatch was accepted")
