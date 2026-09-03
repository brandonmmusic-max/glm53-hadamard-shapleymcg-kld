import json
import sys
from pathlib import Path

from glm53_nvfp4 import shapley_analysis
from glm53_nvfp4.shapley_design import build


def test_shapley_analysis_selects_highest_marginals(tmp_path: Path, monkeypatch):
    design = build(9)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    run_root = tmp_path / "runs"
    run_root.mkdir()
    for cid, layers in design["coalitions"].items():
        # Additive game: lower KLD by layer number / 1e6.
        mean = 1.0 - sum(layers) / 1_000_000
        (run_root / f"run-shapley-{cid}.json").write_text(
            json.dumps({"status": "complete", "role": "conditional-fit", "mean_of_window_means": mean})
        )
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(sys, "argv", ["shapley_analysis", "--design", str(design_path), "--run-root", str(run_root), "--output", str(output)])
    shapley_analysis.main()
    result = json.loads(output.read_text())
    assert result["selected_mxfp6_layers"] == list(range(10, 45))
    assert abs(result["efficiency_remainder"]) < 1e-12


def test_shapley_analysis_accepts_isolated_run_prefix(tmp_path: Path, monkeypatch):
    design = build(11)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    run_root = tmp_path / "runs"
    run_root.mkdir()
    for cid, layers in design["coalitions"].items():
        mean = 2.0 - len(layers) / 1000
        (run_root / f"run-qwen-v10-{cid}.json").write_text(
            json.dumps({"status": "complete", "role": "conditional-fit", "mean_of_window_means": mean})
        )
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["shapley_analysis", "--design", str(design_path), "--run-root", str(run_root),
         "--run-prefix", "qwen-v10", "--output", str(output)],
    )
    shapley_analysis.main()
    assert json.loads(output.read_text())["run_prefix"] == "qwen-v10"
