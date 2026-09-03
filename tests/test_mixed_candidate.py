import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from glm53_nvfp4.mixed_candidate import build


def test_mixed_candidate_counts_payload_and_patches_configs(tmp_path: Path):
    carrier = tmp_path / "carrier"
    carrier.mkdir()
    tensors = {}
    weight_map = {}
    for layer in range(3, 45):
        base = f"model.language_model.layers.{layer}.mlp.experts.0.gate_proj"
        tensors[f"{base}.weight"] = torch.zeros((1, 2), dtype=torch.uint8)
        tensors[f"{base}.weight_scale"] = torch.zeros((1, 1), dtype=torch.float8_e4m3fn)
        tensors[f"{base}.weight_scale_2"] = torch.ones((), dtype=torch.float32)
        tensors[f"{base}.input_scale"] = torch.ones((), dtype=torch.float32)
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            weight_map[f"{base}.{suffix}"] = "carrier.safetensors"
    save_file(tensors, carrier / "carrier.safetensors")
    (carrier / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    qc = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": [],
        "config_groups": {"group_nvfp4_routed_experts": {"targets": []}},
        "quantized_layers": {},
    }
    (carrier / "config.json").write_text(json.dumps({"quantization_config": qc}))
    (carrier / "hf_quant_config.json").write_text(json.dumps(qc))

    chunk = tmp_path / "mxfp6-layer-003.safetensors"
    prefix = "model.language_model.layers.3.mlp.experts.0.gate_proj"
    save_file(
        {
            f"{prefix}.weight": torch.zeros((1, 3), dtype=torch.uint8),
            f"{prefix}.weight_scale": torch.zeros((1, 1), dtype=torch.uint8),
            f"{prefix}.weight_scale_2": torch.ones((), dtype=torch.float32),
            f"{prefix}.input_scale": torch.ones((), dtype=torch.float32),
        },
        chunk,
        metadata={"layer": "3"},
    )
    output = tmp_path / "mixed"
    receipt = build(carrier, output, [chunk], {3})
    assert receipt["logical_elements"] == 42 * 4
    assert receipt["payload_bytes"] == 42 * 11 + 1
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["weight_format"] == "e2m3"
    assert config["activation_format"] == "e4m3"
    assert receipt["source_format"] == "mxfp6_w6a8"
    assert config["quantized_layers"]["model.language_model.layers.3.mlp.experts"]["quant_algo"] == "MXFP6"
    assert config["quantized_layers"]["model.language_model.layers.4.mlp.experts"]["quant_algo"] == "NVFP4"
