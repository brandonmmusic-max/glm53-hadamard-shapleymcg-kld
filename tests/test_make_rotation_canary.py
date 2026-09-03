import torch

from glm53_nvfp4.block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
)
from glm53_nvfp4.make_rotation_canary import negative_identity16, signed_permutation16


def test_signed_permutation_is_exact_invariant():
    rotation = signed_permutation16()
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(16), atol=0, rtol=0)
    assert torch.linalg.det(rotation) == 1
    generator = torch.Generator().manual_seed(65)
    x = torch.randn(5, 32, generator=generator)
    weight = torch.randn(11, 32, generator=generator)
    expected = torch.nn.functional.linear(x, weight)
    actual = torch.nn.functional.linear(
        apply_activation_rotation(x, rotation),
        apply_weight_rotation(weight, rotation),
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_negative_identity_is_exact_so16():
    rotation = negative_identity16()
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(16), atol=0, rtol=0)
    assert torch.linalg.det(rotation) == 1
