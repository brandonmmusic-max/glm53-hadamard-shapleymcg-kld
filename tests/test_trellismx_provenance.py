import json
from pathlib import Path

import pytest

from trellismx.provenance import (
    EncoderBackendConfig,
    ProvenanceOverlay,
)
from trellismx.workflow import WorkflowConfig, workflow_report_with_provenance


def workflow_config() -> WorkflowConfig:
    return WorkflowConfig.from_file(
        Path(__file__).resolve().parents[1]
        / "configs/glm53-flash-coupled-p8-uniform-k4-v1.json"
    )


def overlay_value(workflow: str = "glm53-flash-coupled-p8-uniform-k4-v1") -> dict:
    digest = "a" * 64
    return {
        "schema": "trellismx.provenance-overlay.v1",
        "workflow": workflow,
        "artifacts": {
            "source_checkpoint": {
                "kind": "indexed-bf16-safetensors",
                "uri": "uuid:source-checkpoint",
                "sha256": digest,
                "bytes": 123,
            },
            "coupled_scale_source": {
                "kind": "pinned-exl3-suh-svh-checkpoint",
                "uri": "uuid:scale-source",
                "sha256": "b" * 64,
            },
            "calibration_capture": {
                "kind": "sealed-layer-capture",
                "uri": "uuid:capture",
                "sha256": "c" * 64,
            },
            "role_manifest": {
                "kind": "trellismx-role-manifest",
                "uri": "uuid:roles",
                "sha256": "d" * 64,
            },
        },
    }


def test_overlay_matches_workflow_without_reading_artifacts() -> None:
    config = workflow_config()
    overlay = ProvenanceOverlay.from_mapping(overlay_value())
    overlay.validate_for_workflow(config)
    report = workflow_report_with_provenance(config, overlay)
    assert report["provenance"]["credentials_stored"] is False
    assert report["provenance"]["artifact_bytes_read"] == 0
    assert set(report["provenance"]["artifacts"]) == {
        name
        for name, declared in config.inputs.items()
        if declared["required"]
    }


def test_overlay_rejects_wrong_workflow_or_input() -> None:
    config = workflow_config()
    wrong = ProvenanceOverlay.from_mapping(overlay_value("other-workflow"))
    with pytest.raises(ValueError, match="different workflow"):
        wrong.validate_for_workflow(config)


def test_disabled_encoder_backend_is_not_executable() -> None:
    backend = EncoderBackendConfig.from_file(
        Path(__file__).resolve().parents[1]
        / "configs/trellismx-coupled-encoder.cuda-disabled.json"
    )
    assert backend.enabled is False
    assert backend.executable is False
    assert backend.backend == "trellismx-native-coupled"
    assert backend.summary()["license_status"] == "trellismx-source-available"


def test_optional_future_evaluation_reference_is_not_required() -> None:
    config = workflow_config()
    overlay = ProvenanceOverlay.from_mapping(overlay_value())
    overlay.validate_for_workflow(config)
    assert "evaluation_reference" in config.inputs
    assert config.inputs["evaluation_reference"]["required"] is False
