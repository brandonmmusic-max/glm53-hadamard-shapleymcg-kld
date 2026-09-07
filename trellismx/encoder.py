"""Primary TrellisMX coupled encoder contract.

This module names the encoder implementation used by the sealed GLM campaign.
It does not import CUDA, read a checkpoint, or execute an encoder.
"""

from __future__ import annotations

from typing import Any


NATIVE_ENCODER_SCHEMA = "trellismx.native-coupled-encoder.v1"


NATIVE_COUPLED_ENCODER: dict[str, Any] = {
    "schema": NATIVE_ENCODER_SCHEMA,
    "name": "TrellisMX coupled P8 encoder",
    "status": "research-source-available",
    "used_for_sealed_glm_campaign": True,
    "device": "cuda",
    "ordinary_cli_execution": False,
    "source": {
        "layer_encoder": "glm53_nvfp4/quantize_p8_coupled_rate_layer.py",
        "coupled_transform": "glm53_nvfp4/p8_coupled_scale.py",
        "tensor_codec": "glm53_nvfp4/trellis_mxf.py",
        "captures": "glm53_nvfp4/capture.py",
        "route_weighted_hessians": "glm53_nvfp4/output_aware.py",
    },
    "algorithm": {
        "rates": [3, 4, 5],
        "operand": "e4m3",
        "scale": "ue8m0-k32",
        "trellis_law": "procedural-mcg-alpha2",
        "hessian_feedback": "gptq-style-full-hessian-feedback",
        "ldlq": False,
        "block_ldlq": False,
        "boundary": "coupled-h512-h128-suh-svh-v1",
    },
    "inputs": {
        "source_checkpoint": "indexed BF16 GLM-5.3-Flash checkpoint",
        "coupled_scale_source": "pinned exact EXL3 suh/svh source",
        "calibration_capture": "sealed fit-role routed layer captures",
        "role_manifest": "role-separated calibration manifest",
    },
    "outputs": {
        "chunk_payload": "per-expert-range TrellisMX P8 codec tensors",
        "receipt": "encoder receipt with hashes and transform identity",
    },
    "qualification_boundary": {
        "cpu_encoder_execution": False,
        "cuda_encoder_execution": "separate explicit authorization required",
        "model_quality_claim": "only from separately authorized full-model evaluation",
    },
}


def native_coupled_encoder() -> dict[str, Any]:
    """Return the primary encoder contract without importing its CUDA path."""

    return dict(NATIVE_COUPLED_ENCODER)
