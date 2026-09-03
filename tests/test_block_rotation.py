import torch

from glm53_nvfp4.block_gptq import block_hessian
from glm53_nvfp4.block_rotation import (
    apply_activation_rotation,
    apply_weight_rotation,
    cayley_rotation,
    hadamard16,
    orthogonality_error,
    rotate_block_hessian,
)


def test_hadamard16_is_orthogonal_and_preserves_linear_map():
    generator = torch.Generator().manual_seed(53)
    x = torch.randn(11, 64, generator=generator)
    w = torch.randn(37, 64, generator=generator)
    r = hadamard16()
    xr = apply_activation_rotation(x, r)
    wr = apply_weight_rotation(w, r)
    assert orthogonality_error(r) < 1e-6
    torch.testing.assert_close(torch.nn.functional.linear(xr, wr), torch.nn.functional.linear(x, w), atol=2e-5, rtol=2e-5)


def test_hessian_rotation_matches_rotated_samples():
    generator = torch.Generator().manual_seed(54)
    x = torch.randn(91, 64, generator=generator)
    route = torch.rand(91, generator=generator) + 0.1
    r = hadamard16()
    expected = block_hessian(apply_activation_rotation(x, r), route)
    actual = rotate_block_hessian(block_hessian(x, route), r)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_per_block_rotation_preserves_linear_map():
    generator = torch.Generator().manual_seed(55)
    x = torch.randn(7, 64, generator=generator)
    w = torch.randn(13, 64, generator=generator)
    blocks = []
    for _ in range(4):
        q, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
        blocks.append(q)
    r = torch.stack(blocks)
    xr = apply_activation_rotation(x, r)
    wr = apply_weight_rotation(w, r)
    torch.testing.assert_close(torch.nn.functional.linear(xr, wr), torch.nn.functional.linear(x, w), atol=2e-5, rtol=2e-5)


def test_cayley_is_orthogonal_and_respects_base():
    generator = torch.Generator().manual_seed(56)
    parameter = torch.randn(16, 16, generator=generator) * 0.1
    rotation = cayley_rotation(parameter)
    based = cayley_rotation(torch.zeros_like(parameter), hadamard16())
    assert orthogonality_error(rotation) < 2e-6
    assert torch.equal(based, hadamard16())


def test_batched_cayley_is_orthogonal():
    generator = torch.Generator().manual_seed(57)
    parameter = torch.randn(4, 16, 16, generator=generator) * 0.1
    rotation = cayley_rotation(parameter, hadamard16())
    assert rotation.shape == (4, 16, 16)
    assert orthogonality_error(rotation) < 2e-6
