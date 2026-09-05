import numpy as np

from glm53_nvfp4.analyze_uniform_p8_fullmodel import analyze


def test_uniform_fullmodel_analysis_preserves_windows_and_domains():
    rows = [
        {"id": "w0", "domain": "general"},
        {"id": "w1", "domain": "general"},
        {"id": "w2", "domain": "legal"},
        {"id": "w3", "domain": "legal"},
    ]
    candidate = {"w0": 0.08, "w1": 0.09, "w2": 0.10, "w3": 0.11}
    control = {"w0": 0.10, "w1": 0.10, "w2": 0.10, "w3": 0.10}
    result = analyze(
        plan={"bootstrap": {"replicates": 5000, "seed": 7}},
        role_rows=rows,
        candidate=candidate,
        control=control,
    )
    assert result["status"] == "measurement-complete"
    assert result["windows"] == 4
    assert np.isclose(result["candidate_mean_kld"], 0.095)
    assert np.isclose(result["mean_delta_kld"], -0.005)
    assert result["candidate_window_wins"] == 2
    assert result["domains"]["general"]["windows"] == 2
    assert result["domains"]["legal"]["windows"] == 2
    assert [row["window_id"] for row in result["window_records"]] == [
        "w0",
        "w1",
        "w2",
        "w3",
    ]


def test_uniform_fullmodel_analysis_rejects_non_exact_role():
    rows = [{"id": "w0", "domain": "general"}]
    try:
        analyze(
            plan={"bootstrap": {"replicates": 100, "seed": 7}},
            role_rows=rows,
            candidate={"w0": 0.1, "extra": 0.2},
            control={"w0": 0.1},
        )
    except RuntimeError as error:
        assert "exact preregistered role" in str(error)
    else:
        raise AssertionError("non-exact role was accepted")
