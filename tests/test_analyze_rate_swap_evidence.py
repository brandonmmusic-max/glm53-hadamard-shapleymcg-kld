import argparse
import json
from pathlib import Path

from glm53_nvfp4 import analyze_rate_swap_evidence as ev


def _analysis(path: Path, kld: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arms": {"coupled_full": {
        "true_decode_mean_kld": kld, "including_prefill_mean_kld": kld + 0.001,
        "true_decode_window_bca95": [kld - 0.002, kld + 0.002],
        "per_domain_true_decode_mean_kld": {"axis1_general": kld},
        "conditions": {"image_id": "sha256:abc"}}}}))


def _damage(directory: Path, layer: int, k4: float, k3: float | None = None, k5: float | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    rates = {"4": {"damage_sum": k4}}
    if k3 is not None:
        rates["3"] = {"damage_sum": k3}
    if k5 is not None:
        rates["5"] = {"damage_sum": k5}
    (directory / f"damage-layer-{layer:03d}.json").write_text(json.dumps(
        {"schema": "glm53.p8-layer-rate-damage.v1", "layer": layer, "rates": rates}))


def _args(tmp_path: Path, **over):
    base = {name: None for name in ev.ARMS}
    base.update({"allocation": None, "damage_dir": None, "k4_only_dir": None, "output": tmp_path / "out.json"})
    base.update(over)
    return argparse.Namespace(**{f"{k}_analysis" if k in ev.ARMS else k: v for k, v in base.items()})


def test_verdict_and_additivity_from_four_measured_arms(tmp_path: Path):
    _analysis(tmp_path / "uniform.json", 0.0400)
    _analysis(tmp_path / "k3.json", 0.0460)      # downgrade costs +0.006
    _analysis(tmp_path / "k5.json", 0.0380)      # upgrade gains -0.002
    _analysis(tmp_path / "alloc.json", 0.0432)   # combined +0.0032, not the solo sum of +0.004
    result = ev.build(_args(tmp_path, uniform=tmp_path / "uniform.json", k3_only=tmp_path / "k3.json",
                            k5_only=tmp_path / "k5.json", allocated=tmp_path / "alloc.json"))
    d = result["kld_delta_vs_uniform"]
    assert abs(d["k3_only"] - 0.006) < 1e-12 and abs(d["k5_only"] + 0.002) < 1e-12
    g = result["group_effects"]
    assert abs(g["downgrade_cost_kld"] - 0.006) < 1e-12
    assert abs(g["upgrade_benefit_kld"] - 0.002) < 1e-12
    assert abs(g["additivity_retention"] - 0.8) < 1e-9
    assert result["verdict"]["same_size_swap_pays"] is False
    assert "did not lower" in result["verdict"]["statement"]


def test_swap_that_pays_is_reported_as_such(tmp_path: Path):
    _analysis(tmp_path / "uniform.json", 0.0400)
    _analysis(tmp_path / "alloc.json", 0.0388)
    result = ev.build(_args(tmp_path, uniform=tmp_path / "uniform.json", allocated=tmp_path / "alloc.json"))
    assert result["verdict"]["same_size_swap_pays"] is True
    assert abs(result["verdict"]["relative_change"] + 0.03) < 1e-9
    assert result["kld_delta_vs_uniform"]["k3_only"] is None
    assert "group_effects" not in result


def test_local_proxy_calibration_uses_measured_damage_and_falls_back(tmp_path: Path):
    _analysis(tmp_path / "uniform.json", 0.0400)
    _analysis(tmp_path / "k3.json", 0.0460)
    _analysis(tmp_path / "k5.json", 0.0380)
    _analysis(tmp_path / "alloc.json", 0.0432)
    cand, k4only = tmp_path / "cand", tmp_path / "k4only"
    _damage(cand, 5, 1.0, k3=3.5)
    _damage(cand, 9, 2.0, k5=0.6)
    for layer in (7, 8):
        _damage(k4only, layer, 1.5)
    allocation = tmp_path / "alloc-plan.json"
    allocation.write_text(json.dumps({"assignment": {"5": 3, "7": 4, "8": 4, "9": 5},
                                      "counts": {"3": 1, "4": 2, "5": 1}, "selected_net_offset": 0,
                                      "estimated_total_file_bytes": 1, "bytes_under_budget": 2}))
    result = ev.build(_args(tmp_path, uniform=tmp_path / "uniform.json", k3_only=tmp_path / "k3.json",
                            k5_only=tmp_path / "k5.json", allocated=tmp_path / "alloc.json",
                            allocation=allocation, damage_dir=cand, k4_only_dir=k4only))
    p = result["local_proxy"]["predicted_routed_output_damage_delta"]
    assert abs(p["k3_only"] - 2.5) < 1e-12 and abs(p["k5_only"] + 1.4) < 1e-12 and abs(p["allocated"] - 1.1) < 1e-12
    assert result["local_proxy"]["sign_agreement"] == {"k3_only": True, "k5_only": True, "allocated": True}
    assert result["allocation"]["k3_layers"] == [5] and result["allocation"]["k5_layers"] == [9]
    # A layer whose candidate rate was never scored makes the prediction unavailable, not wrong.
    allocation.write_text(json.dumps({"assignment": {"5": 3, "7": 3, "8": 4, "9": 5}}))
    result = ev.build(_args(tmp_path, uniform=tmp_path / "uniform.json", allocated=tmp_path / "alloc.json",
                            allocation=allocation, damage_dir=cand, k4_only_dir=k4only))
    assert result["local_proxy"]["predicted_routed_output_damage_delta"]["k3_only"] is None


def test_cli_writes_a_receipt(tmp_path: Path):
    _analysis(tmp_path / "uniform.json", 0.04)
    _analysis(tmp_path / "alloc.json", 0.041)
    out = tmp_path / "evidence.json"
    assert ev.main(["--uniform-analysis", str(tmp_path / "uniform.json"),
                    "--allocated-analysis", str(tmp_path / "alloc.json"), "--output", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["schema"] == ev.SCHEMA and written["verdict"]["same_size_swap_pays"] is False
