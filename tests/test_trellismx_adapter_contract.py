import json
from pathlib import Path

import pytest

from trellismx.adapter_contract import ArchitectureAdapterContract
from trellismx.cli import main


def template() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/architecture-adapter-contract.template.json"
    )
    return json.loads(path.read_text())


def test_generic_adapter_template_is_valid_and_cpu_only() -> None:
    contract = ArchitectureAdapterContract.from_mapping(template())
    assert contract.architecture == "example-architecture"
    assert contract.summary()["qualification_owner"] == "downstream-adapter-implementer"
    assert contract.summary()["uses_device"] is False


def test_adapter_contract_must_preserve_p8_runtime() -> None:
    value = template()
    value["runtime"]["mma"] = "nvfp4"
    with pytest.raises(ValueError, match="runtime field 'mma'"):
        ArchitectureAdapterContract.from_mapping(value)


def test_adapter_contract_cli(tmp_path, capsys) -> None:
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(template()))
    assert main(["validate-adapter-contract", str(path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["contract_valid"] is True
