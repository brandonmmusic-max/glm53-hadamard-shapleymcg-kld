import json
from pathlib import Path

from trellismx.cli import main


def test_capabilities_command(capsys) -> None:
    assert main(["capabilities"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["name"] == "TrellisMX"
    assert value["device"]["status"] == "not-run-by-this-package"


def test_fixture_write_and_validate_roundtrip(tmp_path, capsys) -> None:
    directory = tmp_path / "fixture"
    assert main(["fixture", str(directory), "--write"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "passed"

    path = directory / "fixture.p4.safetensors"
    assert main(["validate", str(path), "--expected-shape", "32x128"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["shape"] == [32, 128]


def test_fixture_containers_validate_through_cli(tmp_path, capsys) -> None:
    from trellismx.p8_format import read_p8_tensor_file

    container = tmp_path / "container"
    assert main(["p8-tensor-fixture", str(container), "--write"]) == 0
    capsys.readouterr()
    path = container / "p8-tensors.safetensors"
    expected = read_p8_tensor_file(path).file_sha256
    assert main(["validate-p8-tensors", str(path), "--expected-sha256", expected]) == 0
    value = json.loads(capsys.readouterr().out)
    assert [item["bits"] for item in value["tensors"]] == [3, 4, 5]


def test_checkpoint_and_encoding_plan_cli(tmp_path, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = root / "configs/glm53-flash-coupled-p8-uniform-k4-v1.json"
    manifest = tmp_path / "checkpoint-manifest.json"
    state = tmp_path / "encoding-state.json"
    assert main(["checkpoint-plan", "--config", str(workflow), "--output", str(manifest)]) == 0
    checkpoint = json.loads(capsys.readouterr().out)
    assert checkpoint["status"] == "planned"
    assert manifest.is_file()

    assert main(["encoding-plan", "--config", str(workflow), "--state", str(state)]) == 0
    encoding = json.loads(capsys.readouterr().out)
    assert encoding["schema"] == "trellismx.encoding-orchestration.v1"
    assert state.is_file()
