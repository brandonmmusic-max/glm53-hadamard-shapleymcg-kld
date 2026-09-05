import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.p8_coupled_scale import (
    CoupledScaleSet,
    block_hadamard,
    coupled_expert_reference,
    coupled_input_carrier,
    encode_coupled_scale_weights,
    load_exact_exl3_scales,
    source_expert_reference,
)
from glm53_nvfp4.prepare_p8_coupled_scale_v1 import _parse_layer_paths
from glm53_nvfp4.shard_index import sha256_file


def _signed_scale(length: int, offset: int = 0) -> torch.Tensor:
    values = torch.linspace(0.625, 1.375, length, dtype=torch.float32)
    signs = torch.where(
        (torch.arange(length) + offset) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    return (values * signs).to(torch.float16).contiguous()


def _scales(tmp_path: Path, *, experts: int = 2, hidden: int = 512, intermediate: int = 128):
    gate_up = _signed_scale(hidden)
    down = _signed_scale(hidden, 1)
    private = torch.stack([_signed_scale(intermediate, expert) for expert in range(experts)])
    return CoupledScaleSet(
        gate_up_suh=gate_up,
        gate_svh=private,
        up_svh=torch.stack(
            [_signed_scale(intermediate, expert + 1) for expert in range(experts)]
        ),
        down_suh=torch.stack(
            [_signed_scale(intermediate, expert + 2) for expert in range(experts)]
        ),
        down_svh=down,
        source_path=tmp_path / "synthetic.safetensors",
        source_sha256="a" * 64,
        source_metadata={},
        tensor_hashes={},
    )


def test_coupled_scale_encoder_closes_against_source_expert(tmp_path: Path):
    torch.manual_seed(20260905)
    scales = _scales(tmp_path)
    gate = torch.randn(128, 512) / 32
    up = torch.randn(128, 512) / 32
    down = torch.randn(512, 128) / 16
    hidden = torch.randn(5, 512)

    encoded = encode_coupled_scale_weights(
        gate, up, down, scales, expert=1, intermediate_draw=3
    )
    observed = coupled_expert_reference(
        hidden,
        encoded,
        scales,
        expert=1,
        intermediate_draw=3,
        quantize_activations=False,
    )
    expected = source_expert_reference(hidden, gate, up, down)
    delta = observed.double() - expected.double()
    nmse = delta.square().sum() / expected.double().square().sum()

    # The archived runtime stores three intermediate boundaries in FP16, so
    # closure is numerical rather than bitwise against an all-FP32 source.
    assert float(nmse) < 2e-6


def test_input_carrier_preserves_the_archived_cast_order(tmp_path: Path):
    torch.manual_seed(7)
    hidden = torch.randn(4, 512)
    scale = _signed_scale(512)
    manual = hidden.to(torch.bfloat16).to(torch.float16).float()
    manual = block_hadamard(manual, block_size=512)
    manual = (manual * scale.float()).to(torch.float16).float()
    manual = block_hadamard(manual, block_size=128)

    observed = coupled_input_carrier(hidden, scale, quantize=False)
    assert torch.equal(observed, manual)


def _write_exl3_scales(
    path: Path,
    *,
    layer: int = 3,
    experts: int = 2,
    hidden: int = 512,
    intermediate: int = 128,
    corrupt_shared: bool = False,
) -> None:
    shared_gate = _signed_scale(hidden)
    shared_down = _signed_scale(hidden, 1)
    tensors = {}
    for expert in range(experts):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        gate_shared = shared_gate.clone()
        if corrupt_shared and expert == experts - 1:
            gate_shared[0] = -gate_shared[0]
        tensors[f"{base}.gate_proj.suh"] = gate_shared
        tensors[f"{base}.up_proj.suh"] = gate_shared.clone()
        tensors[f"{base}.gate_proj.svh"] = _signed_scale(intermediate, expert)
        tensors[f"{base}.up_proj.svh"] = _signed_scale(intermediate, expert + 1)
        tensors[f"{base}.down_proj.suh"] = _signed_scale(intermediate, expert + 2)
        tensors[f"{base}.down_proj.svh"] = shared_down.clone()
    save_file(tensors, path, metadata={"codec": "exl3-mcg", "layer": str(layer)})


def _seal(value: dict, field: str) -> dict:
    result = dict(value)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    result[field] = hashlib.sha256(payload).hexdigest()
    return result


def _write_indexed_exl3_checkpoint(
    root: Path,
    *,
    layer: int = 3,
    experts: int = 2,
    hidden: int = 512,
    intermediate: int = 128,
    corrupt_payload_receipt: bool = False,
) -> None:
    root.mkdir()
    loose = root / "loose.safetensors"
    _write_exl3_scales(
        loose,
        layer=layer,
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    with safe_open(loose, framework="pt", device="cpu") as source:
        tensors = {name: source.get_tensor(name) for name in source.keys()}
    loose.unlink()
    shard = root / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    index = {
        "metadata": {"total_size": sum(x.numel() * x.element_size() for x in tensors.values())},
        "weight_map": {name: shard.name for name in tensors},
    }
    config = {
        "text_config": {
            "hidden_size": hidden,
            "n_routed_experts": experts,
            "moe_intermediate_size": intermediate,
        }
    }
    quant = {"quant_method": "exl3", "codebook": "mcg", "bits": 4}
    for name, value in (
        ("model.safetensors.index.json", index),
        ("config.json", config),
        ("quantization_config.json", quant),
    ):
        (root / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    plan_sha = "1" * 64
    rows = [
        {
            "name": name,
            "origin": "sealed_exl3_mcg_packed_choice",
            "payload_sha256": hashlib.sha256(
                tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        for name, tensor in tensors.items()
    ]
    if corrupt_payload_receipt:
        rows[0]["payload_sha256"] = "f" * 64
    shard_receipt = _seal(
        {
            "schema": "quant-pipeline.glm53-k4-materialized-shard-receipt.v1",
            "plan_sha256": plan_sha,
            "shard": shard.name,
            "shard_sha256": sha256_file(shard),
            "shard_bytes": shard.stat().st_size,
            "complete": True,
            "tensors": rows,
        },
        "receipt_sha256",
    )
    receipt_dir = root / ".materialization" / "shards"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{shard.name}.json").write_text(
        json.dumps(shard_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    receipt = _seal(
        {
            "schema": "quant-pipeline.glm53-k4-materialization-receipt.v1",
            "plan_sha256": plan_sha,
            "complete": True,
            "codec_family": "exl3-mcg",
            "mcg_multiplier_hex": "0xCBAC1FED",
            "bits": 4,
            "main_and_mtp_complete": True,
            "nonrouted_native_exact": True,
            "source_model_revision": "2" * 40,
            "index_sha256": sha256_file(root / "model.safetensors.index.json"),
            "config_sha256": sha256_file(root / "config.json"),
            "quantization_config_sha256": sha256_file(root / "quantization_config.json"),
            "shard_sha256": {shard.name: sha256_file(shard)},
            "shard_receipt_sha256": [shard_receipt["receipt_sha256"]],
        },
        "receipt_sha256",
    )
    (root / "materialization-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_exact_exl3_loader_preserves_signed_shared_and_private_scales(tmp_path: Path):
    source = tmp_path / "scales.safetensors"
    _write_exl3_scales(source)

    loaded = load_exact_exl3_scales(
        source,
        layer=3,
        expected_experts=2,
        expected_hidden=512,
        expected_intermediate=128,
    )

    assert loaded.gate_up_suh.shape == (512,)
    assert loaded.gate_svh.shape == (2, 128)
    assert loaded.tensors()["intermediate_scales_fp16"].shape == (2, 384)
    assert bool((loaded.gate_up_suh < 0).any())
    assert len(loaded.source_sha256) == 64
    assert len(loaded.tensor_hashes) == 12


def test_exact_exl3_loader_accepts_sealed_indexed_checkpoint(tmp_path: Path):
    source = tmp_path / "checkpoint"
    _write_indexed_exl3_checkpoint(source)

    loaded = load_exact_exl3_scales(
        source,
        layer=3,
        expected_experts=2,
        expected_hidden=512,
        expected_intermediate=128,
    )

    assert loaded.source_path == source.resolve()
    assert loaded.source_metadata["format"] == "indexed-materialized-exl3-k4"
    assert loaded.source_metadata["receipt_sha256"] == loaded.source_sha256
    assert loaded.gate_svh.shape == (2, 128)
    assert len(loaded.tensor_hashes) == 12


def test_indexed_exl3_loader_fails_closed_on_receipt_or_payload_drift(tmp_path: Path):
    source = tmp_path / "checkpoint"
    _write_indexed_exl3_checkpoint(source)
    receipt = json.loads((source / "materialization-receipt.json").read_text())
    receipt["bits"] = 3
    (source / "materialization-receipt.json").write_text(json.dumps(receipt) + "\n")
    with pytest.raises(RuntimeError, match="seal differs"):
        load_exact_exl3_scales(source, layer=3, expected_experts=2)

    payload_source = tmp_path / "payload-checkpoint"
    _write_indexed_exl3_checkpoint(payload_source, corrupt_payload_receipt=True)
    with pytest.raises(RuntimeError, match="scale payload differs"):
        load_exact_exl3_scales(payload_source, layer=3, expected_experts=2)


def test_exact_exl3_loader_fails_closed_on_nonshared_or_wrong_geometry(tmp_path: Path):
    source = tmp_path / "scales.safetensors"
    _write_exl3_scales(source, corrupt_shared=True)
    with pytest.raises(RuntimeError, match="not shared"):
        load_exact_exl3_scales(source, layer=3, expected_experts=2)

    compatible = tmp_path / "compatible.safetensors"
    _write_exl3_scales(compatible)
    with pytest.raises(RuntimeError, match="hidden size 512, expected 4096"):
        load_exact_exl3_scales(
            compatible,
            layer=3,
            expected_experts=2,
            expected_hidden=4096,
            expected_intermediate=128,
        )


def test_exact_exl3_loader_rejects_layers_outside_frozen_pilot(tmp_path: Path):
    with pytest.raises(ValueError, match="limited to layers"):
        load_exact_exl3_scales(tmp_path / "unused.safetensors", layer=4)


def test_preparation_inputs_are_exactly_three_layers():
    parsed = _parse_layer_paths(["3=/a", "20=/b", "22=/c"])
    assert parsed == {3: Path("/a"), 20: Path("/b"), 22: Path("/c")}
    with pytest.raises(ValueError, match="exactly layers"):
        _parse_layer_paths(["3=/a", "20=/b"])
