from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.repair_safetensors_pt_metadata import repair


def test_repair_changes_only_header_and_adds_pt_tag(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    tensor = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    save_file(
        {"weight": tensor}, str(source),
        metadata={
            "format": "ModelOpt NVFP4 E2M1/E4M3-per-16/FP32-global",
            "descriptor_bank": "procedural structured_hadamard16 bank-v1, 256 entries",
        },
    )
    result = repair(source, output)
    assert result["status"] == "pass"
    assert result["payload_bytes"] == tensor.numel() * tensor.element_size()
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        assert handle.metadata()["format"] == "pt"
        assert torch.equal(handle.get_tensor("weight"), tensor)


def test_repair_is_create_only(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_file(
        {"weight": torch.zeros(8)}, str(source),
        metadata={"format": "ModelOpt NVFP4 E2M1/E4M3-per-16/FP32-global"},
    )
    repair(source, output)
    with pytest.raises(FileExistsError):
        repair(source, output)
