from glm53_nvfp4.analyze_shared_mid_butterfly_gate import analyze


def _manifest(rows, base, domain_adjust=None):
    domain_adjust = domain_adjust or {}
    return {
        "status": "complete",
        "role": "fit",
        "windows": {
            row["id"]: {"mean_kld": base + domain_adjust.get(row["domain"], 0.0)}
            for row in rows
        },
    }


def test_tune_gate_applies_mean_and_domain_thresholds():
    rows = [{"id": f"w{i}", "domain": f"d{i % 4}"} for i in range(32)]
    execution = {
        "runtime_role": "fit",
        "analysis_role": "fit/tune32",
        "stage": "tune32",
        "selected_arm": "p00625",
        "threshold": {"maximum_stock_delta": -0.0004, "maximum_domain_delta": 0.0005},
    }
    result = analyze(
        execution,
        {"roles": {"fit": rows}},
        {
            "stock": _manifest(rows, 0.04),
            "zero": _manifest(rows, 0.041),
            "candidate": _manifest(rows, 0.039),
        },
    )
    assert result["decision"] == "pass"
    assert all(result["checks"].values())


def test_tune_gate_fails_one_bad_domain_even_if_global_mean_wins():
    rows = [{"id": f"w{i}", "domain": f"d{i % 4}"} for i in range(32)]
    execution = {
        "runtime_role": "fit",
        "analysis_role": "fit/tune32",
        "stage": "tune32",
        "selected_arm": "p00625",
        "threshold": {"maximum_stock_delta": -0.0004, "maximum_domain_delta": 0.0005},
    }
    result = analyze(
        execution,
        {"roles": {"fit": rows}},
        {
            "stock": _manifest(rows, 0.04),
            "zero": _manifest(rows, 0.041),
            "candidate": _manifest(rows, 0.038, {"d0": 0.003}),
        },
    )
    assert result["decision"] == "fail"
    assert result["checks"]["maximum_domain_delta"] is False
