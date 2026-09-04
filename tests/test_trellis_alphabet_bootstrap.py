import pytest

from glm53_nvfp4.evaluate_trellis_alphabet import _bootstrap_experts


def test_expert_bootstrap_is_not_computed_for_one_expert() -> None:
    result = _bootstrap_experts(
        [{"candidate_error": 0.8, "control_error": 1.0, "energy": 2.0}]
    )
    assert result["status"] == "not-computed"
    assert result["expert_count"] == 1


def test_expert_bootstrap_uses_at_least_sixteen_experts() -> None:
    rows = [
        {"candidate_error": 0.8, "control_error": 1.0, "energy": 2.0}
        for _ in range(16)
    ]
    result = _bootstrap_experts(rows, replicates=100, seed=7)
    assert result["status"] == "computed"
    assert result["expert_count"] == 16
    assert result["candidate_minus_control_nmse_ci95_bca"] == pytest.approx([-0.1, -0.1])
    assert result["upper_below_zero"]
