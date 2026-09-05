from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from glm53_nvfp4.bf16_layer_candidate import build


def _carrier(root):
    root.mkdir()
    name = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    scale = f"{name}_scale"
    input_scale = name.replace(".weight", ".input_scale")
    save_file(
        {
            name: torch.zeros(2, 2, dtype=torch.uint8),
            scale: torch.ones(2, 1, dtype=torch.float8_e4m3fn),
            input_scale: torch.ones(()),
        },
        str(root / "base.safetensors"),
    )
    weight_map = {key: "base.safetensors" for key in (name, scale, input_scale)}
    qc = {
        "quantized_layers": {"model.language_model.layers.3.mlp.experts": {}},
        "config_groups": {
            "routed": {"targets": ["model.language_model.layers.3.mlp.experts"]}
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (root / "config.json").write_text(json.dumps({"quantization_config": qc}))
    (root / "hf_quant_config.json").write_text(json.dumps(qc))
    return name


def test_bf16_layer_overlay_is_create_only(tmp_path):
    carrier = tmp_path / "carrier"
    name = _carrier(carrier)
    chunk = tmp_path / "dense.safetensors"
    save_file(
        {name: torch.ones(2, 4, dtype=torch.bfloat16)},
        str(chunk),
        metadata={"layer": "3"},
    )
    output = tmp_path / "candidate"
    build(carrier, output, [chunk], {3})
    with pytest.raises(FileExistsError):
        build(carrier, output, [chunk], {3})
