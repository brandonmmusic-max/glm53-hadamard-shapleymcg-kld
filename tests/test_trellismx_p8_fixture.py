import json

from trellismx.cli import main


def test_p8_fixture_cli_writes_and_replays_all_rates(tmp_path, capsys) -> None:
    directory = tmp_path / "fixture"
    assert main(["p8-fixture", str(directory), "--write"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "passed"
    assert first["rates"] == [3, 4, 5]
    assert first["gpu_executed"] is False
    assert not any(first["exact_tensor_mismatches"].values())

    assert main(["p8-fixture", str(directory)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
