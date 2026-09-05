import pytest
import torch

from glm53_nvfp4.block_gptq import block_hessian
from glm53_nvfp4.block_rotation import (
    apply_activation_rotation,
    apply_shared_butterfly16_staged,
    apply_output_rotation,
    apply_weight_rotation,
    butterfly16,
    cayley_rotation,
    hadamard,
    hadamard16,
    orthogonality_error,
    rotate_block_hessian,
    signed_hadamard16,
    structured_hadamard16,
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


def test_butterfly16_identity_quarter_turn_and_linear_closure():
    identity = butterfly16(0.0)
    torch.testing.assert_close(identity, torch.eye(16), atol=0, rtol=0)
    quarter = butterfly16(torch.pi / 4)
    assert orthogonality_error(quarter) < 2e-6
    torch.testing.assert_close(
        quarter.abs(), torch.full((16, 16), 0.25), atol=2e-7, rtol=0
    )
    generator = torch.Generator().manual_seed(5304)
    x = torch.randn(7, 64, generator=generator)
    w = torch.randn(31, 64, generator=generator)
    rotation = butterfly16(torch.tensor([0.07, -0.11, 0.19, -0.23]))
    torch.testing.assert_close(
        torch.nn.functional.linear(
            apply_activation_rotation(x, rotation),
            apply_weight_rotation(w, rotation),
        ),
        torch.nn.functional.linear(x, w),
        atol=3e-5,
        rtol=3e-5,
    )


def test_butterfly16_rejects_invalid_angle_count():
    with pytest.raises(ValueError, match="1, 4, or 32"):
        butterfly16(torch.zeros(3))


def test_shared_butterfly_staged_matches_dense_rotation_in_float32():
    generator = torch.Generator().manual_seed(5305)
    x = torch.randn(7, 64, generator=generator)
    angle = torch.pi / 16
    staged = apply_shared_butterfly16_staged(x, angle)
    dense = apply_activation_rotation(x, butterfly16(angle))
    torch.testing.assert_close(staged, dense, atol=5e-7, rtol=5e-7)


def test_shared_butterfly_staged_rounds_only_final_result():
    generator = torch.Generator().manual_seed(5306)
    x = torch.randn(3, 32, generator=generator).to(torch.bfloat16)
    actual = apply_shared_butterfly16_staged(
        x, torch.pi / 16, output_dtype=torch.bfloat16
    )
    assert actual.dtype == torch.bfloat16
    assert actual.shape == x.shape
    with pytest.raises(ValueError, match="divisible by 16"):
        apply_shared_butterfly16_staged(torch.zeros(2, 17), 0.1)


def test_hadamard32_and_hadamard64_preserve_linear_map():
    generator = torch.Generator().manual_seed(5301)
    x = torch.randn(7, 128, generator=generator)
    w = torch.randn(19, 128, generator=generator)
    for size in (32, 64):
        r = hadamard(size)
        assert orthogonality_error(r) < 1e-6
        torch.testing.assert_close(
            torch.nn.functional.linear(
                apply_activation_rotation(x, r), apply_weight_rotation(w, r)
            ),
            torch.nn.functional.linear(x, w),
            atol=3e-5,
            rtol=3e-5,
        )


def test_signed_hadamard16_is_stable_orthogonal_and_preserves_linear_map():
    generator = torch.Generator().manual_seed(5302)
    x = torch.randn(9, 64, generator=generator)
    w = torch.randn(23, 64, generator=generator)
    first = signed_hadamard16("layer3-expert5-gate-variant2")
    second = signed_hadamard16("layer3-expert5-gate-variant2")
    other = signed_hadamard16("layer3-expert5-gate-variant3")
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert orthogonality_error(first) < 1e-6
    torch.testing.assert_close(
        torch.nn.functional.linear(
            apply_activation_rotation(x, first),
            apply_weight_rotation(w, first),
        ),
        torch.nn.functional.linear(x, w),
        atol=2e-5,
        rtol=2e-5,
    )


def test_structured_hadamard16_is_stable_orthogonal_and_preserves_linear_map():
    generator = torch.Generator().manual_seed(5303)
    x = torch.randn(9, 64, generator=generator)
    w = torch.randn(23, 64, generator=generator)
    first = structured_hadamard16("bank-v1:17")
    second = structured_hadamard16("bank-v1:17")
    other = structured_hadamard16("bank-v1:18")
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert orthogonality_error(first) < 1e-6
    torch.testing.assert_close(
        torch.nn.functional.linear(
            apply_activation_rotation(x, first), apply_weight_rotation(w, first)
        ),
        torch.nn.functional.linear(x, w),
        atol=2e-5,
        rtol=2e-5,
    )


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


def test_two_sided_block_rotation_preserves_linear_map():
    generator = torch.Generator().manual_seed(5501)
    x = torch.randn(7, 64, generator=generator)
    w = torch.randn(32, 64, generator=generator)
    input_rotation = signed_hadamard16("input")
    output_rotation = signed_hadamard16("output")
    transformed = apply_output_rotation(
        apply_weight_rotation(w, input_rotation), output_rotation
    )
    transformed_output = torch.nn.functional.linear(
        apply_activation_rotation(x, input_rotation), transformed
    )
    recovered_output = apply_activation_rotation(
        transformed_output, output_rotation.transpose(-1, -2)
    )
    torch.testing.assert_close(
        recovered_output,
        torch.nn.functional.linear(x, w),
        atol=3e-5,
        rtol=3e-5,
    )


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
