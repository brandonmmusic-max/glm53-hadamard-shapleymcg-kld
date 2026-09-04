from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.export_native_endpoint import export_native_endpoint


def test_export_strips_codec_state_and_preserves_endpoint(tmp_path: Path) -> None:
    stem = "model.layers.3.mlp.experts.0.gate_proj"
    source = tmp_path / "research.safetensors"
    output = tmp_path / "native.safetensors"
    tensors = {
        f"{stem}.weight": torch.arange(8, dtype=torch.uint8).reshape(2, 4),
        f"{stem}.weight_scale": torch.arange(4, dtype=torch.uint8).reshape(2, 2),
        f"{stem}.weight_scale_2": torch.tensor(1.0),
        f"{stem}.input_scale": torch.tensor(1.0),
        f"{stem}.trellis": torch.arange(8, dtype=torch.int16),
        f"{stem}.selectors_2bit": torch.tensor([0], dtype=torch.uint8),
        "codec.codebooks_e4m3": torch.zeros((3, 4), dtype=torch.uint8),
    }
    save_file(tensors, str(source), metadata={"schema": "research"})

    receipt = export_native_endpoint(source, output)

    with safe_open(str(output), framework="pt", device="cpu") as handle:
        assert sorted(handle.keys()) == sorted(
            [
                f"{stem}.weight",
                f"{stem}.weight_scale",
                f"{stem}.weight_scale_2",
                f"{stem}.input_scale",
            ]
        )
        assert handle.metadata()["runtime_trellis_decode"] == "false"
    assert receipt["logical_weights"] == 16
    assert sorted(receipt["excluded_runtime_state"]) == sorted(
        [f"{stem}.trellis", f"{stem}.selectors_2bit", "codec.codebooks_e4m3"]
    )
