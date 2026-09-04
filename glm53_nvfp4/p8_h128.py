"""Exact H128 activation-boundary basis changes for the P8 MoE path.

The ordinary (uncoupled) B12X trellis kernel computes, per 128-wide block,

    g = Dg H (Wg' x), u = Du H (Wu' x)
    h' = H Dd situ(g, u), y = Wd' h'

For positive diagonal vectors ``Dg``, ``Du`` and ``Dd`` the transformed
weights below make that algebraically identical to the original BF16 expert.
The functions are deliberately independent of any quantizer: closure is a
precondition that must be proved before attributing a gain to P8 quantization.
"""
from __future__ import annotations

import torch


HADAMARD_ORDER = 128


def hadamard128_last(values: torch.Tensor) -> torch.Tensor:
    """Apply normalized Sylvester H128 to every last-axis block."""
    if values.shape[-1] % HADAMARD_ORDER:
        raise ValueError("last dimension must be divisible by 128")
    original_shape = values.shape
    work = values.float().reshape(-1, HADAMARD_ORDER).clone()
    width = 1
    while width < HADAMARD_ORDER:
        view = work.reshape(-1, HADAMARD_ORDER // (2 * width), 2, width)
        left = view[:, :, 0, :].clone()
        right = view[:, :, 1, :].clone()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        width *= 2
    work.mul_(HADAMARD_ORDER ** -0.5)
    return work.reshape(original_shape)


def signed_h128_reconstruct(
    middle: torch.Tensor, signs: torch.Tensor, qdq
) -> torch.Tensor:
    """Quantize in a signed-H128 basis and reconstruct in the source basis."""
    if middle.shape[-1] != signs.numel() or signs.shape != (middle.shape[-1],):
        raise ValueError("sign vector must match the last dimension")
    signed = middle.float() * signs.float()
    rotated = hadamard128_last(signed)
    reconstructed = hadamard128_last(qdq(rotated).float())
    return reconstructed * signs.float()


def diagonal_h128_reconstruct(
    middle: torch.Tensor, diagonal: torch.Tensor, qdq
) -> torch.Tensor:
    """Quantize in ``H128 @ diag(d)`` and invert that basis exactly."""
    if middle.shape[-1] != diagonal.numel() or diagonal.shape != (middle.shape[-1],):
        raise ValueError("diagonal must match the last dimension")
    d = diagonal.to(device=middle.device, dtype=torch.float32)
    if not torch.isfinite(d).all() or not (d > 0).all():
        raise ValueError("diagonal must be finite and positive")
    rotated = hadamard128_last(middle.float() * d)
    return hadamard128_last(qdq(rotated).float()) / d


def hadamard128_rows(weight: torch.Tensor) -> torch.Tensor:
    """Left-multiply 128-row blocks of a 2-D weight matrix by H128."""
    if weight.ndim != 2 or weight.shape[0] % HADAMARD_ORDER:
        raise ValueError("weight rows must be divisible by 128")
    return hadamard128_last(weight.float().transpose(0, 1)).transpose(0, 1).contiguous()


def transform_uncoupled_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gate_scale: torch.Tensor | None = None,
    up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform each projection for the kernel's uncoupled H128 boundary.

    Scales are the kernel-side diagonals. They are positive and have one value
    per intermediate coordinate. Unit scales recover the fixed-H128 arm.
    """
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("expected gate/up [I,H] and down [H,I]")
    intermediate = gate.shape[0]
    device = gate.device
    dtype = torch.float32
    one = torch.ones(intermediate, device=device, dtype=dtype)
    rg = one if gate_scale is None else gate_scale.to(device=device, dtype=dtype)
    ru = one if up_scale is None else up_scale.to(device=device, dtype=dtype)
    sd = one if down_scale is None else down_scale.to(device=device, dtype=dtype)
    for name, scale in (("gate", rg), ("up", ru), ("down", sd)):
        if scale.shape != (intermediate,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{name} scale must be finite, positive, and shape [I]")
    gate_t = hadamard128_rows(gate.float() / rg[:, None])
    up_t = hadamard128_rows(up.float() / ru[:, None])
    down_t = hadamard128_last(down.float() / sd[None, :])
    return gate_t, up_t, down_t


def effective_uncoupled_weights(
    gate_transformed: torch.Tensor,
    up_transformed: torch.Tensor,
    down_transformed: torch.Tensor,
    gate_scale: torch.Tensor | None = None,
    up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map decoded kernel-basis weights back to the original expert bases."""
    intermediate = gate_transformed.shape[0]
    one = torch.ones(intermediate, device=gate_transformed.device, dtype=torch.float32)
    rg = one if gate_scale is None else gate_scale.float()
    ru = one if up_scale is None else up_scale.float()
    sd = one if down_scale is None else down_scale.float()
    gate = hadamard128_rows(gate_transformed) * rg[:, None]
    up = hadamard128_rows(up_transformed) * ru[:, None]
    down = hadamard128_last(down_transformed) * sd[None, :]
    return gate, up, down


def uncoupled_boundary(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    gate_scale: torch.Tensor | None = None,
    up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch mirror of the uncoupled H128 kernel activation boundary."""
    intermediate = gate_output.shape[-1]
    one = torch.ones(intermediate, device=gate_output.device, dtype=torch.float32)
    rg = one if gate_scale is None else gate_scale.float()
    ru = one if up_scale is None else up_scale.float()
    sd = one if down_scale is None else down_scale.float()
    gate = hadamard128_last(gate_output.float()) * rg
    up = hadamard128_last(up_output.float()) * ru
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    middle = torch.nn.functional.silu(gate) * up
    return hadamard128_last(middle * sd)
