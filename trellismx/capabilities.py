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
            "status": "generic-contract-plus-one-bundled-qualified-adapter",
            "contract_schema": "trellismx.architecture-adapter.v1",
            "bundled_qualified": ["GLM-5.3-Flash routed MoE"],
            "additional_architecture_ownership": "downstream implementer",
            "unsupported_policy": "reject before reading or encoding model weights",
        },
        "checkpoint_index_inspection": {
            "status": "glm53-flash-supported",
            "safetensors_payload_bytes_read": 0,
        },
        "p8_encoder_backend": {
            "status": "trellismx-native-coupled-research-source",
            "primary_encoder": "TrellisMX coupled P8 encoder",
            "device": "cuda",
            "default_enabled": False,
            "ordinary_cli_execution": False,
            "optional_search_implementation": "operator-pinned plugin; not a product requirement",
        },
        "calibration_and_evaluation_roles": {
            "status": "supported",
            "existing_trellismx_kld_provenance": "CF32 conditional-fit forced-decode campaign",
            "future_qad_use": "reference-only; not claimed as prior measurement provenance",
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
