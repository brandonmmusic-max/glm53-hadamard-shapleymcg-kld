"""Typed contracts for the TrellisMX P8 tensor protocol.

These types expose the model-independent 2-D tensor encoder while preserving
its real constraints.  They do not turn the encoder into an arbitrary-model
converter: an architecture adapter must map named weights and calibration
captures into these requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from glm53_nvfp4.trellis_mxf import (
    TrellisMXF,
    decode_trellis_mxf,
    pack_ue8m0,
    state_lut,
    unpack_trellis_edges,
    unpack_ue8m0,
)
from glm53_nvfp4.trellis_nvfp4 import (
    pack_trellis_edges,
    reconstruct_trellis_states,
)


P8_RATES = (3, 4, 5)
P8_ALPHABET = "e4m3"
P8_LAW = "mcg"
P8_MMA = "mxf8f6f4"
P8_BLOCK_SIZE = 32
P8_COMPANDER_SCALE = 2.0


@dataclass(frozen=True)
class P8EncoderConfig:
    """Frozen production tensor-codec settings used by the GLM campaign."""

    bits: int = 4
    alphabet: str = P8_ALPHABET
    law: str = P8_LAW
    compander_scale: float = P8_COMPANDER_SCALE
    block_size: int = P8_BLOCK_SIZE
    scale_refinement_iterations: int = 2
    percdamp: float = 0.01
    column_block: int = 128
    ldlq: bool = False

    def __post_init__(self) -> None:
        validate_p8_encoder_config(self)

    @property
    def stored_payload_bpw(self) -> float:
        return self.bits + 8.0 / self.block_size

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "TrellisMX-P8",
            "bits": self.bits,
            "alphabet": self.alphabet,
            "law": self.law,
            "mma": P8_MMA,
            "block_scale": f"ue8m0-k{self.block_size}",
            "compander_scale": self.compander_scale,
            "scale_refinement_iterations": self.scale_refinement_iterations,
            "percdamp": self.percdamp,
            "column_block": self.column_block,
            "ldlq": self.ldlq,
            "stored_payload_bpw": self.stored_payload_bpw,
        }


def validate_p8_encoder_config(config: P8EncoderConfig) -> None:
    if config.bits not in P8_RATES:
        raise ValueError(f"P8 stored rate must be one of K{', K'.join(map(str, P8_RATES))}")
    if config.alphabet != P8_ALPHABET:
        raise ValueError("TrellisMX P8 compute operands are E4M3")
    if config.law != P8_LAW:
        raise ValueError("the qualified TrellisMX P8 tensor path uses the procedural MCG law")
    if config.block_size != P8_BLOCK_SIZE:
        raise ValueError("TrellisMX P8 block scales are UE8M0/K32")
    if config.compander_scale <= 0 or not torch.tensor(config.compander_scale).isfinite():
        raise ValueError("compander scale must be finite and positive")
    if config.scale_refinement_iterations < 0:
        raise ValueError("scale refinement iterations cannot be negative")
    if config.percdamp < 0:
        raise ValueError("Hessian damping cannot be negative")
    if config.column_block <= 0:
        raise ValueError("column block must be positive")
    if config.ldlq:
        raise ValueError("the current TrellisMX P8 protocol does not use LDLQ")


def validate_p8_tensor_request(
    weight: torch.Tensor, hessian: torch.Tensor, *, bits: int, block_size: int = 32
) -> None:
    """Validate before touching an encoder extension or device."""

    if bits not in P8_RATES:
        raise ValueError(f"unsupported P8 rate K{bits}")
    if weight.ndim != 2:
        raise ValueError("P8 tensor encoder requires a 2-D [output, input] weight")
    if weight.shape[0] % 16 or weight.shape[1] % math.lcm(16, block_size):
        raise ValueError(
            f"P8 rows must be divisible by 16 and width by lcm(16,{block_size})"
        )
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Hessian shape must match the weight input dimension")


def encode_p8_tensor(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    config: P8EncoderConfig | None = None,
    device_execution_authorized: bool = False,
) -> TrellisMXF:
    """Encode one model-independent 2-D tensor with the existing P8 codec.

    This is deliberately not called by the CLI.  Current TrellisMX P8 encoding
    uses the CUDA trellis encoder; callers must explicitly acknowledge that a
    device run is authorized.
    """

    selected = config or P8EncoderConfig()
    validate_p8_tensor_request(
        weight,
        hessian,
        bits=selected.bits,
        block_size=selected.block_size,
    )
    if not device_execution_authorized:
        raise PermissionError("P8 tensor encoding requires explicit device authorization")
    if weight.device.type != "cuda":
        raise ValueError("the current TrellisMX P8 trellis encoder requires CUDA")
    return quantize_trellis_mxf_gptq(
        weight,
        hessian,
        bits=selected.bits,
        alphabet=selected.alphabet,
        law=selected.law,
        compander_scale=selected.compander_scale,
        block_size=selected.block_size,
        scale_refinement_iterations=selected.scale_refinement_iterations,
        percdamp=selected.percdamp,
        column_block=selected.column_block,
    )


def p8_state_table(bits: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the frozen procedural-MCG state-to-E4M3 table for K3/K4/K5."""

    if bits not in P8_RATES:
        raise ValueError(f"unsupported P8 rate K{bits}")
    return state_lut(
        bits,
        alphabet=P8_ALPHABET,
        law=P8_LAW,
        compander_scale=P8_COMPANDER_SCALE,
        device=device,
    )


def decode_p8_payload(
    trellis: torch.Tensor,
    codebook_e4m3: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    *,
    bits: int,
    rows: int,
    width: int,
    device: str | torch.device = "cpu",
    block_size: int = P8_BLOCK_SIZE,
) -> torch.Tensor:
    """Reference-decode a frozen P8 trellis/UE8M0 payload without CUDA."""

    if bits not in P8_RATES:
        raise ValueError(f"unsupported P8 rate K{bits}")
    if trellis.shape != (width // 16, rows // 16, 16 * bits):
        raise ValueError("P8 trellis shape disagrees with matrix geometry")
    if scale_ue8m0.shape != (rows, width // block_size):
        raise ValueError("P8 UE8M0 scale shape disagrees with matrix geometry")
    return decode_trellis_mxf(
        trellis,
        codebook_e4m3,
        scale_ue8m0,
        bits=bits,
        block_size=block_size,
        rows=rows,
        width=width,
        device=device,
    )


__all__ = [
    "P8EncoderConfig",
    "P8_RATES",
    "decode_p8_payload",
    "encode_p8_tensor",
    "pack_trellis_edges",
    "pack_ue8m0",
    "p8_state_table",
    "reconstruct_trellis_states",
    "unpack_trellis_edges",
    "unpack_ue8m0",
    "validate_p8_encoder_config",
    "validate_p8_tensor_request",
]
