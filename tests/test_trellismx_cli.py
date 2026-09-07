import json

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
