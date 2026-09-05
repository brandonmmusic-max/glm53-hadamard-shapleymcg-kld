import json
import sys
from pathlib import Path

from glm53_nvfp4 import build_kld_shapley_overlay_matrix
from glm53_nvfp4.kld_shapley_native import build_design


def _model(path: Path, weight_map: dict[str, str], quantized_layers=(3, 20)) -> None:
    path.mkdir()
    for shard in set(weight_map.values()):
        (path / shard).write_bytes(shard.encode())
    targets = [f"model.language_model.layers.{layer}.mlp.experts" for layer in quantized_layers]
    quantization_config = {
        "config_groups": {"nvfp4": {"targets": targets}},
        "quantized_layers": {target: {"quant_algo": "NVFP4"} for target in targets},
    }
    (path / "config.json").write_text(
        json.dumps({"quantization_config": quantization_config}) + "\n"
    )
    (path / "hf_quant_config.json").write_text(json.dumps(quantization_config) + "\n")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def test_overlay_matrix_replaces_complete_layer_entry_sets(tmp_path: Path, monkeypatch):
    carrier = tmp_path / "carrier"
    base3 = tmp_path / "base3"
    base20 = tmp_path / "base20"
    cand3 = tmp_path / "cand3"
    cand20 = tmp_path / "cand20"
    prefix = "model.layers.{}.mlp.experts.0.gate_proj"
    carrier_map = {
        f"{prefix.format(layer)}.weight": "stock.safetensors"
        for layer in (3, 20)
    }
    carrier_map.update(
        {
            f"{prefix.format(layer)}.weight_scale": "stock.safetensors"
            for layer in (3, 20)
        }
    )
    _model(carrier, carrier_map)
    _model(base3, {f"{prefix.format(3)}.weight": "base3.safetensors"})
    _model(base20, {f"{prefix.format(20)}.weight": "base20.safetensors"})
    _model(cand3, {f"{prefix.format(3)}.weight": "cand3.safetensors"})
    _model(cand20, {f"{prefix.format(20)}.weight": "cand20.safetensors"})
    design = build_design(
        2,
        layers=[3, 20],
        base_bpw=4.5,
        upgrade_bpw=4.25,
        budget_bpw=4.5,
        game="matched-gptq-to-p8-interaction-pilot",
        upgrade_slots_override=2,
    )
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    output = tmp_path / "matrix"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix",
            "--design", str(design_path),
            "--carrier", str(carrier),
            "--base-overlay", f"3={base3}",
            "--base-overlay", f"20={base20}",
            "--candidate-overlay", f"3={cand3}",
            "--candidate-overlay", f"20={cand20}",
            "--output-root", str(output),
            "--manifest", str(manifest),
        ],
    )
    build_kld_shapley_overlay_matrix.main()
    result = json.loads(manifest.read_text())
    full = next(row for row in result["coalitions"] if row["layers"] == [3, 20])
    index = json.loads((Path(full["model"]) / "model.safetensors.index.json").read_text())
    assert index["weight_map"][f"{prefix.format(3)}.weight"] == "cand3.safetensors"
    assert index["weight_map"][f"{prefix.format(20)}.weight"] == "cand20.safetensors"
    assert not any(name.endswith("weight_scale") for name in index["weight_map"])
    config = json.loads((Path(full["model"]) / "config.json").read_text())
    hf_config = json.loads((Path(full["model"]) / "hf_quant_config.json").read_text())
    for quant in (config["quantization_config"], hf_config):
        assert quant["config_groups"]["nvfp4"]["targets"] == []
        assert quant["quantized_layers"] == {}
    assert full["generated_config_sha256"].keys() == {
        "config.json", "hf_quant_config.json"
    }
    assert result["ldlq"] is False


def test_overlay_matrix_rejects_broken_required_shard(tmp_path: Path, monkeypatch):
    carrier = tmp_path / "carrier"
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    key = "model.layers.3.mlp.experts.0.gate_proj.weight"
    _model(carrier, {key: "stock.safetensors"}, quantized_layers=(3,))
    _model(base, {key: "base.safetensors"}, quantized_layers=())
    _model(candidate, {key: "candidate.safetensors"}, quantized_layers=())
    (candidate / "candidate.safetensors").unlink()
    (candidate / "candidate.safetensors").symlink_to(tmp_path / "missing.safetensors")
    design = build_design(1, layers=[3], upgrade_slots_override=1)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix", "--design", str(design_path), "--carrier", str(carrier),
            "--base-overlay", f"3={base}", "--candidate-overlay", f"3={candidate}",
            "--output-root", str(tmp_path / "matrix"),
            "--manifest", str(tmp_path / "manifest.json"),
        ],
    )
    try:
        build_kld_shapley_overlay_matrix.main()
    except RuntimeError as exc:
        assert "missing shard source files" in str(exc)
    else:
        raise AssertionError("broken required shard was accepted")
