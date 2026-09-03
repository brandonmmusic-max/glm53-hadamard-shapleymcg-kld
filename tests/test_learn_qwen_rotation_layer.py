import torch

from glm53_nvfp4.block_rotation import cayley_rotation
from glm53_nvfp4.learn_qwen_rotation_layer import (
    _expert_output,
    objective,
    projection_objective,
)


def test_qwen_full_expert_objective_fits_independent_rotations():
    generator = torch.Generator().manual_seed(61)
    hidden = torch.randn(2, 5, 16, generator=generator) * 0.1
    route = torch.rand(2, 5, generator=generator) + 0.1
    gate = torch.randn(2, 16, 16, generator=generator) * 0.03
    up = torch.randn(2, 16, 16, generator=generator) * 0.03
    down = torch.randn(2, 16, 16, generator=generator) * 0.03
    reference = _expert_output(hidden, gate, up, down)
    parameter_in = torch.zeros(16, 16, requires_grad=True)
    parameter_mid = torch.zeros(16, 16, requires_grad=True)
    loss = objective(
        hidden,
        route,
        reference,
        gate,
        up,
        down,
        cayley_rotation(parameter_in),
        cayley_rotation(parameter_mid),
        4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_in.grad is not None
    assert parameter_mid.grad is not None


def test_qwen_projection_objective_fits_both_tensor_families():
    generator = torch.Generator().manual_seed(62)
    hidden = torch.randn(2, 5, 16, generator=generator) * 0.1
    middle = torch.randn(2, 5, 16, generator=generator) * 0.1
    route = torch.rand(2, 5, generator=generator) + 0.1
    gate = torch.randn(2, 16, 16, generator=generator) * 0.03
    up = torch.randn(2, 16, 16, generator=generator) * 0.03
    down = torch.randn(2, 16, 16, generator=generator) * 0.03
    parameter_in = torch.zeros(16, 16, requires_grad=True)
    parameter_mid = torch.zeros(16, 16, requires_grad=True)
    loss = projection_objective(
        hidden,
        middle,
        route,
        gate,
        up,
        down,
        cayley_rotation(parameter_in),
        cayley_rotation(parameter_mid),
        4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_in.grad is not None
    assert parameter_mid.grad is not None
    assert parameter_in.grad.norm() > 1e-10
    assert parameter_mid.grad.norm() > 1e-10


def test_qwen_projection_objective_can_fit_families_independently():
    generator = torch.Generator().manual_seed(63)
    hidden = torch.randn(2, 5, 16, generator=generator) * 0.1
    middle = torch.randn(2, 5, 16, generator=generator) * 0.1
    route = torch.rand(2, 5, generator=generator) + 0.1
    weights = [torch.randn(2, 16, 16, generator=generator) * 0.03 for _ in range(3)]
    for family, active in (("in", "in"), ("mid", "mid")):
        parameter_in = torch.zeros(16, 16, requires_grad=True)
        parameter_mid = torch.zeros(16, 16, requires_grad=True)
        loss = projection_objective(
            hidden,
            middle,
            route,
            *weights,
            cayley_rotation(parameter_in),
            cayley_rotation(parameter_mid),
            4,
            family,
        )
        loss.backward()
        expected = parameter_in if active == "in" else parameter_mid
        inactive = parameter_mid if active == "in" else parameter_in
        assert expected.grad is not None and expected.grad.norm() > 1e-10
        assert inactive.grad is None


def test_qwen_projection_objective_supports_independent_block_rotations():
    generator = torch.Generator().manual_seed(64)
    hidden = torch.randn(2, 5, 32, generator=generator) * 0.1
    middle = torch.randn(2, 5, 32, generator=generator) * 0.1
    route = torch.rand(2, 5, generator=generator) + 0.1
    gate = torch.randn(2, 32, 32, generator=generator) * 0.03
    up = torch.randn(2, 32, 32, generator=generator) * 0.03
    down = torch.randn(2, 32, 32, generator=generator) * 0.03
    parameter_in = torch.zeros(2, 16, 16, requires_grad=True)
    rotation_in = cayley_rotation(parameter_in)
    loss = projection_objective(
        hidden,
        middle,
        route,
        gate,
        up,
        down,
        rotation_in,
        torch.eye(16),
        4,
        "in",
    )
    loss.backward()
    assert parameter_in.grad is not None
    assert torch.all(parameter_in.grad.flatten(1).norm(dim=1) > 1e-10)

    # The same independently parameterized basis must cancel exactly before
    # quantization: linear(x @ R_b, W @ R_b) == linear(x, W).
    from glm53_nvfp4.block_rotation import (
        apply_activation_rotation,
        apply_weight_rotation,
    )

    random_parameter = torch.randn(2, 16, 16, generator=generator) * 0.1
    rotation = cayley_rotation(random_parameter)
    x = torch.randn(7, 32, generator=generator)
    weight = torch.randn(11, 32, generator=generator)
    expected = torch.nn.functional.linear(x, weight)
    actual = torch.nn.functional.linear(
        apply_activation_rotation(x, rotation),
        apply_weight_rotation(weight, rotation),
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
