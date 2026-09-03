import json

from glm53_nvfp4.w4a16_control import build


def test_w4a16_control_removes_static_group_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    qc = {
        "quantized_layers": {
            "model.language_model.layers.3.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
        },
        "config_groups": {
            "nvfp4": {
                "targets": ["model.language_model.layers.3.mlp.experts"],
                "weights": {"type": "float", "num_bits": 4, "dynamic": False},
                "input_activations": {"type": "float", "num_bits": 4, "dynamic": False},
            }
        },
    }
    (source / "config.json").write_text(json.dumps({"quantization_config": qc}))
    (source / "hf_quant_config.json").write_text(json.dumps(qc))
    output = tmp_path / "output"
    result = build(source, output, {3})
    actual = json.loads((output / "config.json").read_text())["quantization_config"]
    assert actual["quantized_layers"]["model.language_model.layers.3.mlp.experts"]["quant_algo"] == "W4A16_NVFP4"
    assert actual["config_groups"]["nvfp4"]["targets"] == []
    assert result["weight_payload_unchanged"] is True
