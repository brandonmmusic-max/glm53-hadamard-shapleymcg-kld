"""Orthogonal block-16 transforms for rotated-basis NVFP4 routed experts.

Weights use the PyTorch ``[out, in]`` convention.  For an orthogonal block
rotation R, the stored routed gate/up weight is ``W @ R`` and the routed input
is ``x @ R``.  Therefore ``linear(x @ R, W @ R) == linear(x, W)`` before
quantization.  The router and shared expert must continue to see the original
input; the runtime's ``routed_input_transform`` contract provides that split.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


GROUP_SIZE = 16


def hadamard16(*, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the normalized symmetric Sylvester H16 matrix."""
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < GROUP_SIZE:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / 4.0


def cayley_rotation(parameter: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
    """Map an unconstrained 16x16 parameter to an orthogonal rotation."""
    if parameter.ndim not in (2, 3) or parameter.shape[-2:] != (GROUP_SIZE, GROUP_SIZE):
        raise ValueError(f"expected [...,16,16] parameter, got {tuple(parameter.shape)}")
    skew = 0.5 * (parameter - parameter.transpose(-1, -2))
    eye = torch.eye(GROUP_SIZE, dtype=parameter.dtype, device=parameter.device)
    rotation = torch.linalg.solve(
        (eye + skew).transpose(-1, -2),
        (eye - skew).transpose(-1, -2),
    ).transpose(-1, -2)
    if base is not None:
        _validate_rotation(base, parameter.shape[0] if parameter.ndim == 3 else None)
        rotation = base.to(device=rotation.device, dtype=rotation.dtype) @ rotation
    return rotation


def _validate_rotation(rotation: torch.Tensor, blocks: int | None = None) -> None:
    expected = {(GROUP_SIZE, GROUP_SIZE)}
    if blocks is not None:
        expected.add((blocks, GROUP_SIZE, GROUP_SIZE))
    if tuple(rotation.shape) not in expected:
        raise ValueError(f"rotation shape {tuple(rotation.shape)} not in {sorted(expected)}")


def orthogonality_error(rotation: torch.Tensor) -> float:
    """Maximum absolute error of R^T R from identity."""
    r = rotation.float()
    eye = torch.eye(GROUP_SIZE, dtype=r.dtype, device=r.device)
    gram = r.transpose(-1, -2) @ r
    return float((gram - eye).abs().max())


def apply_weight_rotation(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply ``W @ blockdiag(R)`` to a two-dimensional ``[out, in]`` weight."""
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError(f"expected [out,in] with in divisible by 16, got {tuple(weight.shape)}")
    blocks = weight.shape[1] // GROUP_SIZE
    _validate_rotation(rotation, blocks)
    view = weight.float().reshape(weight.shape[0], blocks, GROUP_SIZE)
    r = rotation.to(device=view.device, dtype=view.dtype)
    if r.ndim == 2:
        result = view @ r
    else:
        result = torch.einsum("obg,bgh->obh", view, r)
    return result.reshape_as(weight)


def apply_activation_rotation(hidden: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply the matching block transform to the last activation dimension."""
    if hidden.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension must be divisible by 16, got {tuple(hidden.shape)}")
    blocks = hidden.shape[-1] // GROUP_SIZE
    _validate_rotation(rotation, blocks)
    view = hidden.reshape(*hidden.shape[:-1], blocks, GROUP_SIZE)
    r = rotation.to(device=view.device, dtype=view.dtype)
    if r.ndim == 2:
        result = view @ r
    else:
        result = torch.einsum("...bg,bgh->...bh", view, r)
    return result.reshape_as(hidden)


def rotate_block_hessian(hessian: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Transform block Hessians as ``R^T H R``."""
    if hessian.ndim != 3 or hessian.shape[-2:] != (GROUP_SIZE, GROUP_SIZE):
        raise ValueError(f"expected [blocks,16,16], got {tuple(hessian.shape)}")
    _validate_rotation(rotation, hessian.shape[0])
    r = rotation.to(device=hessian.device, dtype=hessian.dtype)
    if r.ndim == 2:
        return torch.einsum("ij,bjk,kl->bil", r.T, hessian, r)
    return torch.einsum("bij,bjk,bkl->bil", r.transpose(-1, -2), hessian, r)


def load_layer_rotation(path: Path, layer: int, *, device=None) -> torch.Tensor:
    """Load ``layer_NNN`` from a safetensors transform bundle."""
    tensors = load_file(str(path), device="cpu")
    key = f"layer_{layer:03d}"
    if key not in tensors:
        raise KeyError(f"{key} absent from {path}")
    rotation = tensors[key].float()
    _validate_rotation(rotation, 4096 // GROUP_SIZE)
    if orthogonality_error(rotation) > 2e-4:
        raise ValueError(f"{key} is not orthogonal")
    return rotation.to(device=device) if device is not None else rotation
