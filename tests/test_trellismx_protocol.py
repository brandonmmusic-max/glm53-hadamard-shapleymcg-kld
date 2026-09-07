import json

import pytest
import torch

from trellismx.adapters import UnsupportedArchitectureError, adapter_for
from trellismx.cli import main
from trellismx.protocol import (
    P8EncoderConfig,
    decode_p8_payload,
    encode_p8_tensor,
    pack_trellis_edges,
    pack_ue8m0,
    p8_state_table,
    reconstruct_trellis_states,
    unpack_trellis_edges,
    unpack_ue8m0,
    validate_p8_tensor_request,
)


def test_encoder_config_exposes_exact_protocol_and_rejects_ldlq() -> None:
    config = P8EncoderConfig(bits=5)
    assert config.stored_payload_bpw == 5.25
    assert config.metadata()["mma"] == "mxf8f6f4"
    with pytest.raises(ValueError, match="does not use LDLQ"):
        P8EncoderConfig(ldlq=True)


def test_tensor_request_validates_before_device_execution() -> None:
    weight = torch.zeros(16, 32)
    hessian = torch.zeros(32, 32)
    validate_p8_tensor_request(weight, hessian, bits=3)
    with pytest.raises(ValueError, match="width"):
        validate_p8_tensor_request(torch.zeros(16, 31), hessian, bits=3)
    with pytest.raises(PermissionError, match="explicit device authorization"):
        encode_p8_tensor(weight, hessian, config=P8EncoderConfig(bits=3))


def test_unsupported_architecture_fails_before_checkpoint_io() -> None:
    with pytest.raises(UnsupportedArchitectureError, match="qualified adapters"):
        adapter_for("llama-like-model")


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_cpu_p8_stream_roundtrip_and_reference_decode(bits: int) -> None:
    generator = torch.Generator().manual_seed(20260907)
    edges = torch.randint(
        0, 1 << bits, (2, 1, 256), dtype=torch.int64, generator=generator
    )
    packed = pack_trellis_edges(edges, bits)
    assert packed.shape == (2, 1, 16 * bits)
    restored = unpack_trellis_edges(packed, bits)
    assert torch.equal(restored.to(torch.int64), edges)

    states = reconstruct_trellis_states(restored, bits)
    assert states.shape == (2, 1, 256)
    table = p8_state_table(bits)
    assert table.dtype == torch.uint8 and table.numel() == 65_536
    scales = torch.full((16, 1), 127, dtype=torch.uint8)
    decoded = decode_p8_payload(
        packed,
        table,
        scales,
        bits=bits,
        rows=16,
        width=32,
    )
    assert decoded.shape == (16, 32)
    assert bool(torch.isfinite(decoded).all())
    assert torch.equal(unpack_ue8m0(pack_ue8m0(torch.ones(1))), torch.ones(1))


def test_cli_glm_plan_is_cpu_only(capsys) -> None:
    assert main(["plan", "--architecture", "glm53-flash", "--uniform-rate", "5"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema"] == "trellismx.glm53-flash.plan.v1"
    assert value["layers"] == 42
    assert value["stored_rates"] == [5]
    assert value["execution"] == {
        "reads_checkpoint": False,
        "encodes_tensors": False,
        "uses_device": False,
    }
