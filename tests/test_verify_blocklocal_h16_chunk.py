from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from glm53_nvfp4.block_rotation import apply_weight_rotation
from glm53_nvfp4.modelopt import PackedNVFP4, dequantize, quantize
from glm53_nvfp4.quantize_blocklocal_h16_layer import descriptor_bank
from glm53_nvfp4.shard_index import sha256_file
from glm53_nvfp4.verify_blocklocal_h16_chunk import verify


def test_saved_chunk_physical_to_bf16_closure(tmp_path):
    base = "model.language_model.layers.3.mlp.experts.0"
    dense_tensors = {}
    physical_tensors = {}
    bank = torch.stack(descriptor_bank(256, torch.device("cpu")))
    logical = 0
    descriptor_bytes = 0
    for projection, shape in (
        ("gate_proj", (8, 32)),
        ("up_proj", (8, 32)),
        ("down_proj", (32, 16)),
    ):
        torch.manual_seed(len(projection))
        weight = torch.randn(*shape)
        indexes = torch.arange(shape[1] // 16, dtype=torch.uint8) + 1
        rotation = bank[indexes.long()]
        rotated = apply_weight_rotation(weight, rotation)
        packed = quantize(rotated)
        stem = f"{base}.{projection}"
        physical_tensors[f"{stem}.weight"] = packed.weight
        physical_tensors[f"{stem}.weight_scale"] = packed.weight_scale
        physical_tensors[f"{stem}.weight_scale_2"] = packed.weight_scale_2
        physical_tensors[f"{stem}.rotation_index"] = indexes
        dense_tensors[f"{stem}.weight"] = apply_weight_rotation(
            dequantize(packed), rotation.transpose(-1, -2)
        ).to(torch.bfloat16)
        logical += weight.numel()
        descriptor_bytes += indexes.numel()
    dense = tmp_path / "dense.safetensors"
    physical = tmp_path / "physical.safetensors"
    save_file(dense_tensors, str(dense), metadata={"layer": "3"})
    save_file(physical_tensors, str(physical), metadata={"layer": "3"})
    physical_bytes = sum(x.numel() * x.element_size() for x in physical_tensors.values())
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-chunk-receipt.v1",
        "expert_range": [0, 1],
        "logical_elements": logical,
        "physical_bytes": physical_bytes,
        "descriptor_bytes": descriptor_bytes,
        "outputs": {
            "dense": {"bytes": dense.stat().st_size, "sha256": sha256_file(dense)},
            "physical": {"bytes": physical.stat().st_size, "sha256": sha256_file(physical)},
        },
    }))
    result = verify(dense, physical, receipt)
    assert result["status"] == "pass"
    assert result["projection_pairs"] == 3
    assert result["maximum_abs_bf16_error"] == 0.0
