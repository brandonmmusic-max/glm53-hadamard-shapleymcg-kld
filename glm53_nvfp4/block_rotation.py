"""Orthogonal block transforms for rotated-basis NVFP4 routed experts.

Weights use the PyTorch ``[out, in]`` convention.  For an orthogonal block
rotation R, the stored routed gate/up weight is ``W @ R`` and the routed input
is ``x @ R``.  Therefore ``linear(x @ R, W @ R) == linear(x, W)`` before
quantization.  The router and shared expert must continue to see the original
input; the runtime's ``routed_input_transform`` contract provides that split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file


GROUP_SIZE = 16


def hadamard(size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a normalized symmetric Sylvester Hadamard matrix.

    The logical transform width is independent of NVFP4's physical group size:
    H32/H64 still produce ordinary E2M1 values with one E4M3 scale per 16
    consecutive lanes.
    """
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / (size**0.5)


def hadamard16(*, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the normalized symmetric Sylvester H16 matrix."""
    return hadamard(GROUP_SIZE, device=device, dtype=dtype)


def signed_hadamard16(
    seed: int | str,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return a deterministic randomized in-block ``D @ H16`` transform.

    ``D`` changes the combinations formed before quantization.  Signs placed
    after H16 would only flip already-formed columns and are redundant for the
    sign-symmetric E2M1 alphabet.  The first sign is fixed positive so a matrix
    and its global negation are not both searched.  SHA-256 keeps this mapping
    stable across Python processes and machines.
    """
    bits = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:2], "little")
    signs = torch.ones(GROUP_SIZE, dtype=dtype, device=device)
    for index in range(1, GROUP_SIZE):
        if bits & (1 << index):
            signs[index] = -1
    return signs[:, None] * hadamard16(device=device, dtype=dtype)


def structured_hadamard16(
    seed: int | str,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return deterministic ``D @ P @ H16`` from a compact procedural law.

    The signed permutation is generated with SHA-256 and Fisher-Yates rather
    than a framework RNG, keeping the transform bit-stable across machines.
    A 256-member bank needs only 16 four-bit permutation entries and 16 sign
    bits per member (2.5 KiB total), while each physical weight block stores
    one eight-bit bank index.
    """
    identity = str(seed).encode()
    stream = bytearray()
    counter = 0
    while len(stream) < 64:
        stream.extend(hashlib.sha256(identity + counter.to_bytes(4, "little")).digest())
        counter += 1
    permutation = list(range(GROUP_SIZE))
    cursor = 0
    for index in range(GROUP_SIZE - 1, 0, -1):
        swap = int.from_bytes(stream[cursor : cursor + 2], "little") % (index + 1)
        cursor += 2
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    signs = torch.ones(GROUP_SIZE, dtype=dtype, device=device)
    for index in range(1, GROUP_SIZE):
        if stream[cursor + index] & 1:
            signs[index] = -1
    monomial = torch.zeros((GROUP_SIZE, GROUP_SIZE), dtype=dtype, device=device)
    monomial[torch.arange(GROUP_SIZE, device=device), torch.tensor(permutation, device=device)] = signs
    return monomial @ hadamard16(device=device, dtype=dtype)


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


def _validate_rotation(
    rotation: torch.Tensor,
    blocks: int | None = None,
    *,
    group_size: int | None = None,
) -> int:
    if rotation.ndim not in (2, 3) or rotation.shape[-2] != rotation.shape[-1]:
        raise ValueError(f"rotation must contain square matrices, got {tuple(rotation.shape)}")
    size = int(rotation.shape[-1])
    if group_size is not None and size != group_size:
        raise ValueError(f"rotation width {size} does not match required width {group_size}")
    expected = {(size, size)}
    if blocks is not None:
        expected.add((blocks, size, size))
    elif rotation.ndim == 3:
        expected.add(tuple(rotation.shape))
    if tuple(rotation.shape) not in expected:
        raise ValueError(f"rotation shape {tuple(rotation.shape)} not in {sorted(expected)}")
    return size


def orthogonality_error(rotation: torch.Tensor) -> float:
    """Maximum absolute error of R^T R from identity."""
    r = rotation.float()
    size = _validate_rotation(r)
    eye = torch.eye(size, dtype=r.dtype, device=r.device)
    gram = r.transpose(-1, -2) @ r
    return float((gram - eye).abs().max())


def apply_weight_rotation(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply ``W @ blockdiag(R)`` to a two-dimensional ``[out, in]`` weight."""
    size = _validate_rotation(rotation)
    if weight.ndim != 2 or weight.shape[1] % size:
        raise ValueError(
            f"expected [out,in] with in divisible by {size}, got {tuple(weight.shape)}"
        )
    blocks = weight.shape[1] // size
    _validate_rotation(rotation, blocks)
    view = weight.float().reshape(weight.shape[0], blocks, size)
    r = rotation.to(device=view.device, dtype=view.dtype)
    if r.ndim == 2:
        result = view @ r
    else:
        result = torch.einsum("obg,bgh->obh", view, r)
    return result.reshape_as(weight)


def apply_output_rotation(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply ``blockdiag(S)^T @ W`` on a two-dimensional output dimension.

    A kernel computes the corresponding transformed output and applies ``S^T``
    before the following nonlinearity or routed reduction.  Unlike the existing
    input transform, this is independently selectable for gate, up, and down.
    """
    size = _validate_rotation(rotation)
    if weight.ndim != 2 or weight.shape[0] % size:
        raise ValueError(
            f"expected [out,in] with out divisible by {size}, got {tuple(weight.shape)}"
        )
    blocks = weight.shape[0] // size
    _validate_rotation(rotation, blocks)
    view = weight.float().reshape(blocks, size, weight.shape[1])
    r = rotation.to(device=view.device, dtype=view.dtype)
    if r.ndim == 2:
        result = torch.einsum("ij,bjk->bik", r.transpose(-1, -2), view)
    else:
        result = torch.einsum("bij,bjk->bik", r.transpose(-1, -2), view)
    return result.reshape_as(weight)


def apply_activation_rotation(hidden: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply the matching block transform to the last activation dimension."""
    size = _validate_rotation(rotation)
    if hidden.shape[-1] % size:
        raise ValueError(f"last dimension must be divisible by {size}, got {tuple(hidden.shape)}")
    blocks = hidden.shape[-1] // size
    _validate_rotation(rotation, blocks)
    view = hidden.reshape(*hidden.shape[:-1], blocks, size)
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
    _validate_rotation(rotation, hessian.shape[0], group_size=GROUP_SIZE)
    r = rotation.to(device=hessian.device, dtype=hessian.dtype)
    if r.ndim == 2:
        return torch.einsum("ij,bjk,kl->bil", r.T, hessian, r)
    return torch.einsum("bij,bjk,bkl->bil", r.transpose(-1, -2), hessian, r)


def load_layer_rotation(
    path: Path,
    layer: int,
    *,
    kind: str = "in",
    width: int = 4096,
    device=None,
) -> torch.Tensor:
    """Load one layer/projection-family transform from a safetensors bundle.

    Legacy bundles name the routed gate/up transform ``layer_NNN``.  Exact
    Qwen-style bundles may instead use ``layer_NNN_in`` and must use
    ``layer_NNN_mid`` for the independently transformed down-projection input.
    """
    if kind not in {"in", "mid"}:
        raise ValueError(f"invalid rotation kind {kind!r}")
    if width <= 0 or width % GROUP_SIZE:
        raise ValueError(f"invalid transformed width {width}")
    tensors = load_file(str(path), device="cpu")
    preferred = f"layer_{layer:03d}_{kind}"
    legacy = f"layer_{layer:03d}"
    key = preferred if preferred in tensors else legacy
    if key not in tensors:
        raise KeyError(f"{preferred} absent from {path}")
    rotation = tensors[key].float()
    _validate_rotation(rotation, width // GROUP_SIZE, group_size=GROUP_SIZE)
    if orthogonality_error(rotation) > 2e-4:
        raise ValueError(f"{key} is not orthogonal")
    return rotation.to(device=device) if device is not None else rotation
