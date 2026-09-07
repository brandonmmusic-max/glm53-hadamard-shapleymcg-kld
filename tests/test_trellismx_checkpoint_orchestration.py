import hashlib
import json
from pathlib import Path

import pytest

from trellismx.checkpoint import CheckpointPlan, sidecar_name
from trellismx.orchestration import EncodingPlan, EncodingStateFile
from trellismx.workflow import WorkflowConfig


def config() -> WorkflowConfig:
    return WorkflowConfig.from_file(
        Path(__file__).resolve().parents[1]
        / "configs/glm53-flash-coupled-p8-uniform-k4-v1.json"
    )


def test_checkpoint_plan_covers_every_layer_and_rank() -> None:
    value = CheckpointPlan(config()).manifest()
    assert value["schema"] == "trellismx.checkpoint-manifest.v1"
    assert len(value["layers"]) == 42
    assert sum(len(layer["ranks"]) for layer in value["layers"].values()) == 168
    assert value["layers"]["3"]["ranks"]["0"]["file"] == sidecar_name(3, 0)
    assert value["status"] == "planned"
    assert value["runtime"]["loader_closure"] == "not tested by manifest construction"
    assert len(value["payload_sha256"]) == 64


def test_checkpoint_plan_rejects_partial_concrete_manifest() -> None:
    plan = CheckpointPlan(config())
    partial = {
        (3, 0): {
            "layer": 3,
            "rank": 0,
            "path": "sidecar.safetensors",
            "sha256": "a" * 64,
            "bits": 4,
        }
    }
    with pytest.raises(ValueError, match="every expected layer/rank"):
        plan.manifest(partial)


def test_encoding_plan_has_resumable_layer_dag() -> None:
    plan = EncodingPlan(config())
    summary = plan.summary()
    assert summary["workflow"] == "glm53-flash-coupled-p8-uniform-k4-v1"
    assert summary["task_count"] == 2 + 42 * 3 + 2
    assert summary["layer_task_count"] == 42 * 3
    assert summary["uses_device"] is False
    assert summary["tasks"][2]["dependencies"] == ["validate-provenance"]


def test_encoding_state_is_atomic_and_dependency_checked(tmp_path) -> None:
    plan = EncodingPlan(config())
    state_path = tmp_path / "state" / "encoding.json"
    state_file = EncodingStateFile(state_path, plan)
    state = state_file.initialize()
    assert state["status"] == "pending"
    assert state["tasks"]["inspect-checkpoint"]["status"] == "pending"

    with pytest.raises(ValueError, match="dependencies are incomplete"):
        state_file.update("validate-provenance", "running")

    state_file.update("inspect-checkpoint", "running")
    state_file.update(
        "inspect-checkpoint",
        "completed",
        receipt_sha256=hashlib.sha256(b"inspect").hexdigest(),
    )
    state_file.update("validate-provenance", "running")
    with pytest.raises(ValueError, match="require a 64-digit receipt hash"):
        state_file.update("validate-provenance", "completed")
    reloaded = state_file.load()
    assert reloaded["tasks"]["inspect-checkpoint"]["status"] == "completed"
    assert reloaded["tasks"]["validate-provenance"]["status"] == "running"
