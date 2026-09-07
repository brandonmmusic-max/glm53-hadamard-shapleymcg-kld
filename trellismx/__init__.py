"""Public entry points for the TrellisMX research toolkit.

The import name is ``trellismx``.  The historical ``glm53_nvfp4`` modules
remain unchanged so sealed artifacts and tests continue to use their original
call paths.
"""

from .capabilities import CAPABILITIES, capabilities
from .adapters import (
    ArchitectureAdapter,
    ArchitecturePlan,
    GLM53FlashAdapter,
    UnsupportedArchitectureError,
    adapter_for,
)
from .codec import (
    P4Operands,
    P4Payload,
    decode_p4,
    deserialize_p4,
    read_p4,
    serialize_p4,
    storage_accounting,
    verify_operands,
)
from .protocol import (
    P8EncoderConfig,
    decode_p8_payload,
    encode_p8_tensor,
    pack_trellis_edges,
    pack_ue8m0,
    p8_state_table,
    reconstruct_trellis_states,
    unpack_trellis_edges,
    unpack_ue8m0,
)
from .workflow import WorkflowConfig, workflow_report

__version__ = "0.1.0"

__all__ = [
    "CAPABILITIES",
    "ArchitectureAdapter",
    "ArchitecturePlan",
    "GLM53FlashAdapter",
    "P4Operands",
    "P4Payload",
    "capabilities",
    "adapter_for",
    "decode_p4",
    "deserialize_p4",
    "read_p4",
    "serialize_p4",
    "storage_accounting",
    "verify_operands",
    "P8EncoderConfig",
    "decode_p8_payload",
    "encode_p8_tensor",
    "pack_trellis_edges",
    "pack_ue8m0",
    "p8_state_table",
    "reconstruct_trellis_states",
    "unpack_trellis_edges",
    "unpack_ue8m0",
    "WorkflowConfig",
    "workflow_report",
]
