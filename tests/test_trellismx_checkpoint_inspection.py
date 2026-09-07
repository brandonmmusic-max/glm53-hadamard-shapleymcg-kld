import json
from pathlib import Path

import pytest

from trellismx.adapters import GLM53FlashAdapter


def checkpoint(tmp_path: Path, *, drop_name: bool = False) -> Path:
    root = tmp_path / "checkpoint"
    root.mkdir()
    adapter = GLM53FlashAdapter()
    names = [
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in adapter.layers
        for expert in range(adapter.experts)
        for projection in adapter.projections
    ]
    if drop_name:
        names = names[:-1]
    shard = root / "shard.safetensors"
    shard.write_bytes(b"")
    config = {
        "architectures": ["Glm5NextForCausalLM"],
        "text_config": {
            "hidden_size": 4096,
            "n_routed_experts": 288,
            "moe_intermediate_size": 2048,
        },
    }
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {name: shard.name for name in names},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return root


def test_checkpoint_index_inspection_reads_no_tensor_payloads(tmp_path) -> None:
    report = GLM53FlashAdapter().inspect_checkpoint(checkpoint(tmp_path))
    assert report["status"] == "passed"
    assert report["expected_routed_weight_tensors"] == 42 * 288 * 3
    assert report["checkpoint_tensor_reads"] == 0
    assert report["safetensors_payload_bytes_read"] == 0


def test_missing_expert_name_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing 1 routed expert weights"):
        GLM53FlashAdapter().inspect_checkpoint(checkpoint(tmp_path, drop_name=True))
