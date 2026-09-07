import json

import pytest
from safetensors.torch import safe_open

from trellismx.cli import main
from trellismx.sidecar import validate_sidecar, write_synthetic_sidecar


def test_synthetic_k5_sidecar_roundtrip(tmp_path) -> None:
    path = tmp_path / "sidecar" / "p8.safetensors"
    report = write_synthetic_sidecar(path, layer=20, rank=3, bits=5)
    assert report.layer == 20
    assert report.rank == 3
    assert report.bits == 5
    assert report.runtime_loader_closure == "not tested"
    replay = validate_sidecar(
        path,
        expected_layer=20,
        expected_rank=3,
        expected_bits=5,
        expected_design_sha256=report.source_design_sha256,
        expected_scale_source_sha256=report.exl3_scale_source_sha256,
        production_geometry=False,
        experts=1,
        hidden=32,
        local_intermediate=32,
    )
    assert replay.file_sha256 == report.file_sha256


def test_sidecar_corrupt_tensor_hash_fails(tmp_path) -> None:
    path = tmp_path / "sidecar.safetensors"
    write_synthetic_sidecar(path, bits=3)
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        tensors = {
            name: source.get_tensor(name).clone().contiguous()
            for name in source.keys()
        }
    tensors["w2_scale_ue8m0"][0, 0, 0] ^= 1
    from safetensors.torch import save_file

    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        validate_sidecar(
            path,
            production_geometry=False,
            experts=1,
            hidden=32,
            local_intermediate=32,
        )


def test_sidecar_fixture_cli_writes_and_verifies(tmp_path, capsys) -> None:
    directory = tmp_path / "fixture"
    assert main(["sidecar-fixture", str(directory), "--write", "--bits", "3"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "passed"
    assert value["bits"] == 3
    assert value["runtime_loader_closure"] == "not tested"
