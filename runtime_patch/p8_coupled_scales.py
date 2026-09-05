"""Validated P8 scale-sandwich metadata and exact CPU reference.

This module deliberately has no CUTLASS/CUDA imports.  It validates the five
FP16 scale roles before the native runtime moves them to a device and records
the storage casts used by Luke's coupled GLM reference.  The scale component
is not the H512/H128/sign transform itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch


SCHEMA = "glm53-p8-coupled-scale-component-tp4-rank.v1"
COMPONENT = "p8-suh-svh-scale-sandwich-v1"
COMPOSITION_TARGET = "coupled-h512-h128-suh-svh-v1"
CAST_ORDER = "mul-f32-cvt-rn-f16-h128"

SCALE_NAMES = (
    "gate_up_suh_fp16",
    "intermediate_scales_fp16",
    "down_svh_fp16",
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU-contiguous tensor's physical bytes."""

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hadamard_128(*, device: torch.device | str = "cpu") -> torch.Tensor:
    cols = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(col)).bit_count() & 1 for col in cols],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(device=device)


def had128_luke(
    value: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    """Luke ordering: optional suh multiply -> FP16 -> H128 -> optional svh."""

    if value.ndim != 2 or value.shape[1] % 128:
        raise ValueError("H128 input must be rank 2 with width divisible by 128")
    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    had = _hadamard_128(device=work.device)
    work = (work.view(rows, width // 128, 128) @ had).reshape(rows, width)
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


@dataclass(frozen=True)
class P8ScaleSandwich:
    """TP-local five-role scale component.

    ``intermediate_scales`` stores ``gate_svh | up_svh | down_suh`` along its
    last dimension.  The shared input/output vectors are replicated per rank.
    """

    gate_up_suh: torch.Tensor
    intermediate_scales: torch.Tensor
    down_svh: torch.Tensor

    @property
    def packed(self) -> torch.Tensor:
        return torch.cat(
            (
                self.gate_up_suh.reshape(-1),
                self.intermediate_scales.reshape(-1),
                self.down_svh.reshape(-1),
            )
        ).contiguous()

    def split_intermediate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intermediate = self.intermediate_scales.shape[1] // 3
        return self.intermediate_scales.split(intermediate, dim=1)


def validate_scale_component(
    metadata: Mapping[str, str],
    tensors: Mapping[str, torch.Tensor],
    *,
    layer: int,
    rank: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> P8ScaleSandwich:
    """Validate scale metadata and byte hashes, rejecting partial coupling."""

    required = {
        "schema": SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": "4",
        "component": COMPONENT,
        "composition_target": COMPOSITION_TARGET,
        "cast_order": CAST_ORDER,
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        # This sidecar implements only the scales.  A caller must separately
        # own the H512/H128/sign stages before it can advertise full coupling.
        "full_coupled": "false",
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"invalid P8 scale component metadata: {mismatches}")

    expected_shapes = {
        "gate_up_suh_fp16": (hidden,),
        "intermediate_scales_fp16": (experts, 3 * intermediate),
        "down_svh_fp16": (hidden,),
    }
    checked: dict[str, torch.Tensor] = {}
    for name, shape in expected_shapes.items():
        tensor = tensors.get(name)
        if tensor is None:
            raise RuntimeError(f"missing P8 scale tensor {name}")
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"invalid {name}: expected FP16 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite P8 scale tensor {name}")
        if torch.count_nonzero(tensor) != tensor.numel():
            raise RuntimeError(f"zero-valued P8 scale tensor {name}")
        expected_hash = metadata.get(f"sha256_{name}")
        actual_hash = tensor_sha256(tensor)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"P8 scale hash mismatch for {name}: "
                f"expected={expected_hash!r} actual={actual_hash}"
            )
        checked[name] = tensor.contiguous()

    # Signed scales are intentional format evidence.  Do not require that
    # every checkpoint contains a negative value, but reject metadata which
    # claims a positive-only ABI.
    if metadata.get("signed_scales") != "true":
        raise RuntimeError("P8 scale component must declare signed_scales=true")

    return P8ScaleSandwich(
        gate_up_suh=checked["gate_up_suh_fp16"],
        intermediate_scales=checked["intermediate_scales_fp16"],
        down_svh=checked["down_svh_fp16"],
    )


def scale_sandwich_reference(
    x: torch.Tensor,
    gate_physical: torch.Tensor,
    up_physical: torch.Tensor,
    down_physical: torch.Tensor,
    scales: P8ScaleSandwich,
) -> torch.Tensor:
    """Exact ordinary H128 scale sequence, without H512 or coupled signs.

    This reference is a closure tool for this component only.  Physical
    weights must already be encoded in the matching transformed bases.
    """

    gate_svh, up_svh, down_suh = scales.split_intermediate()
    source = had128_luke(x, suh=scales.gate_up_suh, store_fp16=True)
    gate = (source.float() @ gate_physical.float().T).to(torch.float16)
    up = (source.float() @ up_physical.float().T).to(torch.float16)
    gate = had128_luke(gate, svh=gate_svh[0], store_fp16=False)
    up = had128_luke(up, svh=up_svh[0], store_fp16=False)
    activated = (gate * torch.sigmoid(gate) * up)
    down_input = had128_luke(activated, suh=down_suh[0], store_fp16=True)
    down = (down_input.float() @ down_physical.float().T).to(torch.float16)
    return had128_luke(down, svh=scales.down_svh, store_fp16=False)


__all__ = [
    "CAST_ORDER",
    "COMPONENT",
    "COMPOSITION_TARGET",
    "P8ScaleSandwich",
    "SCALE_NAMES",
    "SCHEMA",
    "had128_luke",
    "scale_sandwich_reference",
    "tensor_sha256",
    "validate_scale_component",
]
