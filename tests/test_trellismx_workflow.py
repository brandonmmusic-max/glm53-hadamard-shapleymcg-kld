import json
from pathlib import Path

from trellismx.cli import main
from trellismx.protocol import P8EncoderConfig
from trellismx.workflow import WorkflowConfig, workflow_report


def test_upgrades_workflow_matches_measured_rate_assignment() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/glm53-flash-coupled-p8-upgrades-k4k5-v1.json"
    config = WorkflowConfig.from_file(path)
    report = workflow_report(config)
    assert report["plan"]["rate_counts"] == {"4": 25, "5": 17}
    assert config.encoder == P8EncoderConfig(bits=4)
    assert report["plan"]["stored_rates"] == [4, 5]
    assert report["portable_config_contains_artifact_paths"] is False
    assert report["stages"][-1]["status"] == "not-run-by-this-package"


def test_workflow_cli_does_not_execute_research_stages(capsys) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/glm53-flash-coupled-p8-uniform-k4-v1.json"
    assert main(["workflow", "--config", str(path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["plan"]["stored_rates"] == [4]
    assert all(stage["status"] != "executed" for stage in value["stages"])


def test_workflow_cli_can_declare_future_qad_evaluation(capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(
        [
            "workflow",
            "--config",
            str(root / "configs/glm53-flash-coupled-p8-uniform-k4-v1.json"),
            "--evaluation-reference",
            str(root / "configs/glm53-flash-qad-future-evaluation-v1.json"),
        ]
    ) == 0
    value = json.loads(capsys.readouterr().out)
    future = value["future_evaluation_reference"]
    assert future["used_for_existing_trellismx_measurements"] is False
