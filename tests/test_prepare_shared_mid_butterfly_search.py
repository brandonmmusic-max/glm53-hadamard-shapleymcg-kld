import argparse
import json

from glm53_nvfp4.prepare_shared_mid_butterfly_search import build


def _write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def test_build_records_every_input_and_preserves_decision_fields(tmp_path):
    prior_v6 = _write_json(tmp_path / "v6.json", {"decision": "fail"})
    prior_v7 = _write_json(tmp_path / "v7.json", {"decision": "fail"})
    train_rows = [{"id": f"train-{i}"} for i in range(32)]
    tune_rows = [{"id": f"tune-{i}"} for i in range(32)]
    train = _write_json(
        tmp_path / "train.json",
        {"counts": {"fit": 32}, "roles": {"fit": train_rows}},
    )
    tune = _write_json(
        tmp_path / "tune.json",
        {"counts": {"fit": 32}, "roles": {"fit": tune_rows}},
    )
    input_names = (
        "source_index",
        "capture_manifest",
        "calibration_roles",
        "conditional_fit_roles",
        "materializer",
        "rotation_builder",
        "candidate_builder",
        "build_finalizer",
        "layer_validator",
        "build_runner",
        "run_kld",
        "role_eval",
        "paired_analysis",
        "runtime_manifest",
    )
    values = {}
    for name in input_names:
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        values[name] = path
    args = argparse.Namespace(
        prior_v6_analysis=prior_v6,
        prior_v7_analysis=prior_v7,
        train_roles=train,
        tune_roles=tune,
        prior_plan=None,
        amendment_reason="unused",
        **values,
    )

    plan = build(args)

    assert plan["decision_before_result"] is True
    assert plan["protected_roles_opened"] == []
    assert set(plan["inputs"]) == set(input_names) | {
        "train_roles",
        "tune_roles",
        "plan_builder",
    }
    for record in plan["inputs"].values():
        assert record is not None
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64
