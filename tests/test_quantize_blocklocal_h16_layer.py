from __future__ import annotations

import torch
import torch.nn.functional as F

from glm53_nvfp4.block_rotation import apply_activation_rotation, apply_weight_rotation
from glm53_nvfp4.modelopt import dequantize
from glm53_nvfp4.quantize_blocklocal_h16_layer import (
    block_diagonal_hessian,
    descriptor_bank,
    quantize_projection,
)


def test_descriptor_bank_is_deterministic_and_orthogonal():
    first = descriptor_bank(8, torch.device("cpu"))
    second = descriptor_bank(8, torch.device("cpu"))
    eye = torch.eye(16)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(torch.allclose(item.T @ item, eye, atol=1e-6) for item in first)


def test_block_diagonal_hessian_matches_full_diagonal_blocks():
    torch.manual_seed(1)
    samples = torch.randn(19, 32)
    route = torch.rand(19)
    blocks = block_diagonal_hessian(samples, route)
    weights = route.square() / route.square().sum()
    full = (samples * weights.sqrt()[:, None]).T @ (samples * weights.sqrt()[:, None])
    assert torch.allclose(blocks[0], full[:16, :16])
    assert torch.allclose(blocks[1], full[16:, 16:])


def test_packed_rotated_and_dense_effective_paths_close():
    torch.manual_seed(2)
    weight = torch.randn(8, 32)
    carrier = torch.randn(40, 32)
    route = torch.rand(40)
    bank = descriptor_bank(4, torch.device("cpu"))
    rotation = torch.stack((bank[1], bank[3]))
    packed, effective = quantize_projection(
        weight, carrier, route, rotation, torch.tensor(0.001), 4
    )
    physical_weight = dequantize(packed)
    physical = F.linear(apply_activation_rotation(carrier, rotation), physical_weight)
    dense = F.linear(carrier, effective)
    assert torch.equal(
        effective,
        apply_weight_rotation(physical_weight, rotation.transpose(-1, -2)),
    )
    assert torch.allclose(physical, dense, atol=2e-5, rtol=2e-5)
