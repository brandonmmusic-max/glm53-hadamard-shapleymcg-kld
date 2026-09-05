import math

import pytest
import torch
from safetensors.torch import save_file

from glm53_nvfp4.block_rotation import butterfly16
from glm53_nvfp4.quantize_rotated_hessian_trellis_layer import (
    _linear_float,
    _load_rotation,
    _prequant_middle,
    rotated_runtime_middle,
)
from glm53_nvfp4.shard_index import sha256_file
from glm53_nvfp4.canary_mxfp6_reap import _qdq_e4m3_k32


def test_zero_angle_runtime_middle_matches_unrotated_reference():
    generator = torch.Generator().manual_seed(5307)
    hidden = torch.randn(5, 32, generator=generator).to(torch.bfloat16)
    gate = torch.randn(32, 32, generator=generator)
    up = torch.randn(32, 32, generator=generator)
    expected = _qdq_e4m3_k32(
        _prequant_middle(hidden, gate, up).float(), 1.0, "amax"
    )
    actual = rotated_runtime_middle(hidden, gate, up, 0.0)
    assert torch.equal(actual, expected)


def test_nonzero_runtime_middle_preserves_shape_and_native_carrier():
    generator = torch.Generator().manual_seed(5308)
    hidden = torch.randn(3, 32, generator=generator).to(torch.bfloat16)
    gate = torch.randn(64, 32, generator=generator)
    up = torch.randn(64, 32, generator=generator)
    actual = rotated_runtime_middle(hidden, gate, up, math.pi / 16)
    assert actual.shape == (3, 64)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()


def test_rotation_loader_requires_canonical_selected_payload(tmp_path):
    path = tmp_path / "rotation.safetensors"
    matrix = butterfly16(math.pi / 16)
    save_file(
        {"layer_003_mid": matrix},
        path,
        metadata={
            "schema": "glm53-rotation-v8.shared-butterfly16.v1",
            "layer": "3",
            "scope": "mid-only",
            "angle_pi": "0.0625",
        },
    )
    angle, loaded = _load_rotation(path, 3, sha256_file(path))
    assert angle == 0.0625
    assert torch.equal(loaded, matrix)
    with pytest.raises(RuntimeError, match="differs"):
        _load_rotation(path, 3, "0" * 64)


def test_encoder_metric_linear_accepts_bf16_activation_and_float_weight():
    activation = torch.arange(32, dtype=torch.bfloat16).reshape(1, 32)
    weight = torch.eye(32, dtype=torch.float32)
    actual = _linear_float(activation, weight)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, activation.float())
