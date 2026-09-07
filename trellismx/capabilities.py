"""Machine-readable, evidence-bounded capability reporting."""

from __future__ import annotations

from typing import Any


CAPABILITIES: dict[str, Any] = {
    "name": "TrellisMX",
    "package_version": "0.1.0",
    "cpu": {
        "p4_matrix_codec": {
            "status": "supported",
            "schema": "glm53-p4-mcg-matrix.v1",
            "scope": "portable K4 matrix interchange and deterministic synthetic fixture",
        },
        "p8_reference_codec": {
            "status": "supported",
            "rates": [3, 4, 5],
            "scope": "stream packing/unpacking, state reconstruction, E4M3 table, UE8M0, dense reference decode, and K3/K4/K5 synthetic fixture",
        },
        "p8_tp4_sidecar_validation": {
            "status": "supported",
            "scope": "full-coupled sidecar metadata, tensor inventory, geometry, dtype, tensor hashes, and coupled-sign identity",
            "runtime_loader_closure": "not tested",
        },
        "portable_p8_tensor_container": {
            "status": "supported",
            "schema": "trellismx.p8-tensors.v1",
        },
        "p8_checkpoint_conversion": {
            "status": "not-exposed-by-this-cli",
            "reason": "the sealed GLM-5.3-Flash P8 encoder is research-campaign code with pinned private artifacts; it is not a general converter",
        },
        "architecture_adapters": {
            "status": "glm53-flash-research-only",
            "supported": ["GLM-5.3-Flash routed MoE"],
            "unsupported_policy": "reject before reading or encoding model weights",
        },
        "checkpoint_index_inspection": {
            "status": "glm53-flash-supported",
            "safetensors_payload_bytes_read": 0,
        },
        "p8_viterbi_encoder_backend": {
            "status": "external-plugin-required",
            "default_enabled": False,
            "reason": "the real CUDA Viterbi/Hessian encoder depends on a separately pinned KQuant/QSRT snapshot with unverified redistribution terms",
        },
        "provenance_overlays": {
            "status": "supported",
            "credentials_stored": False,
            "artifact_bytes_read": 0,
        },
        "sm120_runtime_overlay": {
            "status": "separate-source-only-package",
            "package": "trellismx-runtime-sm120",
            "executable": False,
            "requires_separate_pinned_image": True,
        },
    },
    "device": {
        "status": "not-run-by-this-package",
        "note": "CUDA, serving, benchmark, and full-model quality commands are deliberately not launched here",
    },
}


def capabilities() -> dict[str, Any]:
    """Return a shallow copy safe enough for CLI/reporting use."""
    return {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in CAPABILITIES.items()
    }
