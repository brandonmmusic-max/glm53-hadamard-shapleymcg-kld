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
    torch.testing.assert_close(gate, torch.tensor(1.0 / NVFP4_GLOBAL_DENOMINATOR))
    torch.testing.assert_close(gate, up)
