from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p8_coupled_real_sidecar_loader_closure.py"


def _module():
    spec = importlib.util.spec_from_file_location("p8_real_loader_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_is_cpu_only_and_bounded_to_loader_abi() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    record = json.loads(completed.stdout)
    protocol = record["protocol"]
    assert record["protocol_sha256"] == _module().PROTOCOL_SHA256
    assert record["protocol_sha256"] == (
        "bd9dbf3273445152ad93e743c2d7ca229c458ed5254045b35d269e2ff32a2b5f"
    )
    assert protocol["immutable_image"].startswith("sha256:")
    assert protocol["geometry"]["ranks"] == [0, 1, 2, 3]
    assert protocol["moe_mma_executed"] is False
    assert protocol["kld_tested"] is False
    assert "no MoE/MMA call" in protocol["claim_boundary"]
    assert "constructor scale repacking may use device kernels" in protocol["claim_boundary"]


@pytest.mark.parametrize("layer", [3, 20, 22])
def test_protocol_is_derived_from_explicit_supported_layer(layer: int) -> None:
    module = _module()
    protocol = module.protocol_for_layer(layer)
    assert protocol["geometry"]["layer"] == layer
    assert f"real layer-{layer} " in protocol["claim_boundary"]
    assert module.protocol_sha256_for_layer(layer) == module.canonical_sha256(protocol)
    with pytest.raises(ValueError, match="one of"):
        module.protocol_for_layer(21)


def _receipt(module, tmp_path: Path):
    sidecars = []
    ranks = []
    hashes = {name: "a" * 64 for name in module.TENSOR_NAMES}
    for rank in range(4):
        path = tmp_path / f"rank-{rank}.safetensors"
        path.write_bytes(f"rank-{rank}".encode())
        sidecars.append(path)
        ranks.append({
            "rank": rank,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": module.M1.sha256_file(path),
            "tensor_count": 8,
            "tensor_sha256": hashes,
            "shapes": module.EXPECTED_SHAPES,
            "dtypes": module.EXPECTED_DTYPES,
            "source_exact": True,
        })
    receipt = tmp_path / "postwrite.json"
    receipt.write_text(json.dumps({
        "schema": "glm53-p8-coupled-tp4-postwrite-closure.v1",
        "status": "pass",
        "evidence_level": "postwrite-source-exact-structural-closure",
        "layer": 3,
        "world_size": 4,
        "source_design_sha256": module.V3_DESIGN_SHA256,
        "runtime_loader_closure": "not tested",
        "retirement_authorized": False,
        "ranks": ranks,
    }))
    return sidecars, receipt


def test_postwrite_binding_requires_four_ordered_real_file_hashes(tmp_path: Path) -> None:
    module = _module()
    sidecars, receipt = _receipt(module, tmp_path)
    loaded = module._load_postwrite(receipt, sidecars, layer=3)
    assert [row["rank"] for row in loaded["ranks"]] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="postwrite receipt contract mismatch"):
        module._load_postwrite(receipt, sidecars, layer=20)
    sidecars[2].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="rank 2 sidecar/postwrite identity"):
        module._load_postwrite(receipt, sidecars, layer=3)


def _small_tensors():
    return {
        "w13_trellis": torch.arange(8, dtype=torch.int16).reshape(2, 1, 1, 1, 4),
        "w2_trellis": torch.arange(4, dtype=torch.int16).reshape(1, 1, 1, 4),
        "w13_scale_ue8m0": torch.arange(4, dtype=torch.uint8).reshape(1, 2, 2),
        "w2_scale_ue8m0": torch.arange(2, dtype=torch.uint8).reshape(1, 2, 1),
        "gate_up_suh_fp16": torch.tensor([1.0, -1.0], dtype=torch.float16),
        "intermediate_scales_fp16": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16),
        "down_svh_fp16": torch.tensor([-2.0, 2.0], dtype=torch.float16),
        "coupled_sign_draw_u8": torch.zeros(1, dtype=torch.uint8),
    }


def _patch_small_geometry(module, monkeypatch, tensors):
    monkeypatch.setattr(module, "EXPECTED_SHAPES", {
        name: list(tensor.shape) for name, tensor in tensors.items()
    })
    monkeypatch.setattr(module, "EXPECTED_DTYPES", {
        name: str(tensor.dtype).removeprefix("torch.") for name, tensor in tensors.items()
    })


def test_stored_sidecar_audits_every_tensor_and_external_pins(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    tensors = _small_tensors()
    _patch_small_geometry(module, monkeypatch, tensors)
    tensor_hashes = {
        name: module._tensor_sha256_chunked(tensor) for name, tensor in tensors.items()
    }
    metadata = {
        "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
        "layer": "3", "rank": "0", "world_size": "4", "bits": "4",
        "alphabet": "e4m3", "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": "coupled-h512-h128-suh-svh-v1", "full_coupled": "true",
        "activation": "silu-cap10", "ldlq": "false",
        "source_design_sha256": module.V3_DESIGN_SHA256,
        "encoder_transform_sha256": module.TRANSFORM_SHA256,
        "sign_draw": "0", "sha256_coupled_signs_fp16": "b" * 64,
        **{f"sha256_{name}": digest for name, digest in tensor_hashes.items()},
    }
    path = tmp_path / "rank-0.safetensors"
    save_file(tensors, path, metadata=metadata)
    row = {"tensor_sha256": tensor_hashes}
    observed = module._inspect_sidecar(path, row, layer=3, rank=0)
    assert observed["tensor_sha256"] == tensor_hashes
    with pytest.raises(RuntimeError, match="sidecar metadata mismatch"):
        module._inspect_sidecar(path, row, layer=20, rank=0)
    with pytest.raises(RuntimeError, match="sidecar metadata mismatch"):
        module._inspect_sidecar(path, row, layer=3, rank=1)
    metadata["source_design_sha256"] = "c" * 64
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(RuntimeError, match="sidecar metadata mismatch"):
        module._inspect_sidecar(path, row, layer=3, rank=0)


def test_actual_wrapper_materializations_match_postwrite_hashes(monkeypatch) -> None:
    module = _module()
    tensors = _small_tensors()
    _patch_small_geometry(module, monkeypatch, tensors)
    row = {
        "tensor_sha256": {
            name: module._tensor_sha256_chunked(tensor)
            for name, tensor in tensors.items()
        }
    }

    class P8NativeTPMoE:
        alternate_descriptor = False

        def __init__(self, _sidecar, **kwargs):
            self.tp_rank = kwargs["tp_rank"]
            self.layer = kwargs["layer"]
            self.experts = 288
            self.hidden = 4096
            self.intermediate = 512
            self.topk = 8
            self.trellis_bits = 4
            self.source_design_sha256 = module.V3_DESIGN_SHA256
            self.full_coupled = True
            self.scale_component = SimpleNamespace(
                full_coupled=True, transform_sha256=module.TRANSFORM_SHA256
            )
            self.small_m_scheduler = True
            self.fc1_tile_n = 128
            self.deterministic_output = True
            self._compiled = {}
            self.w13_stream = tensors["w13_trellis"].view(torch.int32).reshape(-1)
            self.w2_stream = tensors["w2_trellis"].view(torch.int32).reshape(-1)
            self.w13_scale_mx = tensors["w13_scale_ue8m0"].reshape(-1)
            self.w2_scale_mx = tensors["w2_scale_ue8m0"].reshape(-1)
            self.w13_dummy = (
                self.w13_stream.view(torch.uint8).clone()
                if self.alternate_descriptor
                else self.w13_stream.view(torch.uint8)
            )
            self.w2_dummy = self.w2_stream.view(torch.uint8)
            self.scale_component_packed = torch.cat((
                tensors["gate_up_suh_fp16"].reshape(-1),
                tensors["intermediate_scales_fp16"].reshape(-1),
                tensors["down_svh_fp16"].reshape(-1),
                torch.ones(1536, dtype=torch.float16),
            ))

    runtime, result = module._inspect_runtime_rank(
        P8NativeTPMoE,
        Path("/not-opened"),
        row,
        layer=20,
        rank=2,
        device=torch.device("cpu"),
    )
    assert runtime.tp_rank == 2 and runtime.layer == 20
    assert result["layer"] == 20
    assert result["identity_fallback"] is False
    assert result["moe_mma_executed"] is False
    assert result["runtime_loaded_tensor_sha256"] == {
        name: row["tensor_sha256"][name]
        for name in module.TENSOR_NAMES
        if name != "coupled_sign_draw_u8"
    }
    P8NativeTPMoE.alternate_descriptor = True
    with pytest.raises(RuntimeError, match="alternate/fallback descriptor"):
        module._inspect_runtime_rank(
            P8NativeTPMoE,
            Path("/not-opened"),
            row,
            layer=20,
            rank=2,
            device=torch.device("cpu"),
        )


def test_probe_command_is_v9_networkless_and_mounts_all_four_ranks(tmp_path: Path) -> None:
    module = _module()
    sidecars, receipt = _receipt(module, tmp_path)
    args = argparse.Namespace(
        gpu_device="0",
        design=Path("/inputs/design.json"),
        transform=Path("/inputs/transform.json"),
        postwrite=receipt,
        output=tmp_path / "output",
        sidecar=sidecars,
        layer=20,
    )
    command = module.build_probe_command(args, ROOT)
    joined = " ".join(str(item) for item in command)
    assert "--network=none" in command
    assert module.V9_IMAGE in command
    assert "docker build" not in joined and "systemctl" not in joined
    assert sum(f"/inputs/rank-{rank}.safetensors:ro" in joined for rank in range(4)) == 4
    assert "--postwrite-sha256" in command and "--harness-sha256" in command
    layer_at = command.index("--layer")
    assert command[layer_at + 1] == "20"


def test_execute_and_probe_require_explicit_layer() -> None:
    module = _module()
    args = module.parse_args(["--execute"])
    with pytest.raises(ValueError, match="layer"):
        module._require(args, ("layer",))
    source = SCRIPT.read_text()
    assert "global LAYER" not in source
    assert source.count('"layer": args.layer') >= 3
