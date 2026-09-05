import pytest

from glm53_nvfp4.analyze_shared_mid_butterfly_train import analyze


def test_analysis_selects_lowest_grid_arm_and_not_stock():
    rows = [
        {"id": f"w{i}", "domain": f"d{i % 4}"}
        for i in range(32)
    ]
    execution = {
        "run_order": ["stock", "zero", "negative", "positive"],
        "grid_arms": ["zero", "negative", "positive"],
    }

    def manifest(value):
        return {
            "status": "complete",
            "role": "fit",
            "windows": {row["id"]: {"mean_kld": value} for row in rows},
        }

    result = analyze(
        execution,
        {"roles": {"fit": rows}},
        {
            "stock": manifest(0.04),
            "zero": manifest(0.039),
            "negative": manifest(0.035),
            "positive": manifest(0.038),
        },
    )
    assert result["selected_arm"] == "negative"
    assert result["adaptive_decision"] == "select-nonzero-for-tune"
    assert result["arms"]["negative"]["relative_improvement_vs_stock"] == pytest.approx(0.125)
    assert result["protected_roles_opened"] == []
