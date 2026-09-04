from __future__ import annotations

import torch

from glm53_nvfp4.p8_h128 import diagonal_h128_reconstruct, effective_uncoupled_weights, hadamard128_last, signed_h128_reconstruct, transform_uncoupled_weights, uncoupled_boundary


def test_h128_is_involution() -> None:
    torch.manual_seed(7)
    values = torch.randn(3, 256)
    restored = hadamard128_last(hadamard128_last(values))
    torch.testing.assert_close(restored, values, rtol=2e-6, atol=2e-6)


def test_uncoupled_transformed_expert_closes() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(5, 256)
    gate = torch.randn(256, 256) * 0.03
    up = torch.randn(256, 256) * 0.03
    down = torch.randn(256, 256) * 0.03
    gate_scale = torch.rand(256) + 0.5
    up_scale = torch.rand(256) + 0.5
    down_scale = torch.rand(256) + 0.5
    reference = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate).clamp(max=10.0))
        * torch.nn.functional.linear(hidden, up).clamp(-10.0, 10.0),
        down,
    )
    gate_t, up_t, down_t = transform_uncoupled_weights(gate, up, down, gate_scale, up_scale, down_scale)
    middle = uncoupled_boundary(
        torch.nn.functional.linear(hidden, gate_t),
        torch.nn.functional.linear(hidden, up_t),
        gate_scale,
        up_scale,
        down_scale,
    )
    actual = torch.nn.functional.linear(middle, down_t)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)


def test_signed_h128_identity_quantizer_closes() -> None:
    torch.manual_seed(19)
    middle = torch.randn(4, 256)
    signs = torch.randint(0, 2, (256,)).mul(2).sub(1).float()
    reconstructed = signed_h128_reconstruct(middle, signs, lambda value: value)
    torch.testing.assert_close(reconstructed, middle, rtol=2e-6, atol=2e-6)


def test_diagonal_h128_identity_quantizer_closes() -> None:
    torch.manual_seed(23)
    middle = torch.randn(4, 256)
    diagonal = torch.rand(256) * 3.0 + 0.25
    reconstructed = diagonal_h128_reconstruct(middle, diagonal, lambda value: value)
    torch.testing.assert_close(reconstructed, middle, rtol=3e-6, atol=3e-6)


def test_effective_weights_invert_transform() -> None:
    torch.manual_seed(29)
    gate = torch.randn(256, 256)
    up = torch.randn(256, 256)
    down = torch.randn(256, 256)
    diagonal = torch.rand(256) * 2.0 + 0.25
    transformed = transform_uncoupled_weights(gate, up, down, down_scale=diagonal)
    restored = effective_uncoupled_weights(*transformed, down_scale=diagonal)
    for actual, expected in zip(restored, (gate, up, down)):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
