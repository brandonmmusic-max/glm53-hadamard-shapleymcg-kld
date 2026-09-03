import torch

from glm53_nvfp4.calibrate_input_scale import (
    NVFP4_GLOBAL_DENOMINATOR,
    scale_tensors,
)


def test_scale_tensors_emit_matched_gate_up_scalars():
    maxima = torch.arange(1, 289, dtype=torch.float32)
    tensors = scale_tensors(7, maxima)
    assert len(tensors) == 576
    gate = tensors["model.language_model.layers.7.mlp.experts.0.gate_proj.input_scale"]
    up = tensors["model.language_model.layers.7.mlp.experts.0.up_proj.input_scale"]
    assert gate.shape == up.shape == torch.Size([])
    torch.testing.assert_close(gate, torch.tensor(288.0 / NVFP4_GLOBAL_DENOMINATOR))
    torch.testing.assert_close(gate, up)
    torch.testing.assert_close(
        gate,
        tensors["model.language_model.layers.7.mlp.experts.287.gate_proj.input_scale"],
    )


def test_scale_tensors_emit_independent_down_scalars():
    maxima = torch.arange(1, 289, dtype=torch.float32)
    down_maxima = maxima * 3
    tensors = scale_tensors(7, maxima, down_maxima)
    assert len(tensors) == 864
    down = tensors["model.language_model.layers.7.mlp.experts.0.down_proj.input_scale"]
    torch.testing.assert_close(
        down, torch.tensor(864.0 / NVFP4_GLOBAL_DENOMINATOR)
    )
    torch.testing.assert_close(
        down,
        tensors["model.language_model.layers.7.mlp.experts.287.down_proj.input_scale"],
    )
