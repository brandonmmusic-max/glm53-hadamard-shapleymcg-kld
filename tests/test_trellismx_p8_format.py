import json

import pytest
import torch

from trellismx.cli import main
from trellismx.p8_format import (
    P8TensorPayload,
    read_p8_tensor_file,
    synthetic_p8_payload,
    write_p8_tensor_file,
)


def test_portable_multi_rate_container_roundtrip(tmp_path) -> None:
    path = tmp_path / "payload.safetensors"
    payloads = {
        f"k{bits}": synthetic_p8_payload(f"k{bits}", bits=bits)
        for bits in (3, 4, 5)
    }
    write_p8_tensor_file(path, payloads, source_metadata={"role": "synthetic-test-only"})
    expected_hash = read_p8_tensor_file(path).file_sha256
    loaded = read_p8_tensor_file(path, expected_sha256=expected_hash)
    assert loaded.names() == ("k3", "k4", "k5")
    assert loaded.source_metadata == {"role": "synthetic-test-only"}
    assert loaded.storage_report()["codebook_bytes"] == 0
    for name, original in payloads.items():
        assert torch.equal(loaded.records[name].trellis, original.trellis)
        assert torch.equal(loaded.records[name].scale_ue8m0, original.scale_ue8m0)
        assert loaded.records[name].decode().shape == (16, 32)


def test_invalid_scale_shape_fails() -> None:
    payload = synthetic_p8_payload("bad", bits=4)
    with pytest.raises(ValueError, match="UE8M0 scale"):
        P8TensorPayload(
            name=payload.name,
            rows=payload.rows,
            width=payload.width,
            bits=payload.bits,
            trellis=payload.trellis,
            scale_ue8m0=payload.scale_ue8m0[:, :0],
        )


def test_p8_tensor_fixture_cli(tmp_path, capsys) -> None:
    directory = tmp_path / "fixture"
    assert main(["p8-tensor-fixture", str(directory), "--write"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert [item["bits"] for item in value["tensors"]] == [3, 4, 5]
    assert value["safe_loading"] == "safetensors-no-pickle"
