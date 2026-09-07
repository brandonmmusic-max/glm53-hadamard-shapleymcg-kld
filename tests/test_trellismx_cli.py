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
