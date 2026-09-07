"""TrellisMX aliases for the versioned CPU P4 matrix codec."""

from __future__ import annotations

from glm53_nvfp4.p4_codec import (
    CONTRACT as P4_CONTRACT,
    P4Operands,
    P4Payload,
    decode_p4,
    deserialize_p4,
    read_p4,
    serialize_p4,
    storage_accounting,
    verify_operands,
)


P8_CONTRACT = {
    "family": "TrellisMX-P8",
    "compute_operand": "e4m3",
    "mma": "mxf8f6f4",
    "block_scale": "ue8m0-k32",
    "stored_rates": (3, 4, 5),
    "production_checkpoint_schema": "glm53-p8-full-coupled-all42-tp4-checkpoint.v1",
    "coupled_rank_schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
    "boundary": "coupled-h512-h128-suh-svh-v1",
    "distribution": "research-source-and-sealed-artifacts",
    "general_converter": False,
}

__all__ = [
    "P4Operands",
    "P4Payload",
    "P4_CONTRACT",
    "P8_CONTRACT",
    "decode_p4",
    "deserialize_p4",
    "read_p4",
    "serialize_p4",
    "storage_accounting",
    "verify_operands",
]
