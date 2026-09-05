import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars import (
    _load_rank_coupled,
    _payload_rates,
    _rank_metadata,
    expected_metadata_bpw,
)
from glm53_nvfp4.shard_index import sha256_file
from glm53_nvfp4.verify_p8_coupled_scale_tp4_sidecars import verify_postwrite
from glm53_nvfp4.p8_coupled_scale import (
    COUPLED_ACTIVATION,
    COUPLED_BOUNDARY,
    COUPLED_CAST_ORDER,
    COUPLED_CHUNK_SCHEMA,
    COUPLED_FC1_INTERLEAVE,
    COUPLED_QUANTIZED_DOWN_ORDER,
    COUPLED_QUANTIZED_INPUT_ORDER,
    COUPLED_SIGN_DRAW,
    COUPLED_SIGN_GENERATOR,
    COUPLED_TP_SLICE,
    COUPLED_TRANSFORM_ID,
    COUPLED_TRANSFORM_SHA256,
    _tensor_sha256,
    rank_local_coupled_signs,
)


def _write_chunk(path: Path, start: int, stop: int, *, shared_marker: float = 1.0):
    hidden, intermediate = 512, 128
    tensors = {
        "coupled.gate_up_suh_fp16": torch.full(
            (hidden,), shared_marker, dtype=torch.float16
        ),
        "coupled.down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
        "coupled.gate_svh_fp16": torch.ones(stop - start, intermediate, dtype=torch.float16),
        "coupled.up_svh_fp16": torch.full(
            (stop - start, intermediate), 2.0, dtype=torch.float16
        ),
        "coupled.down_suh_fp16": torch.full(
            (stop - start, intermediate), -0.5, dtype=torch.float16
        ),
        "coupled.intermediate_draw_u8": torch.tensor(
            [COUPLED_SIGN_DRAW for _ in range(start, stop)], dtype=torch.uint8
        ),
    }
    for expert in range(start, stop):
        base = f"model.language_model.layers.3.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.trellis"] = torch.zeros(2, 8, 64, dtype=torch.int16)
        tensors[f"{base}.up_proj.trellis"] = torch.ones(2, 8, 64, dtype=torch.int16)
        tensors[f"{base}.down_proj.trellis"] = torch.full(
            (8, 2, 64), 2, dtype=torch.int16
        )
        tensors[f"{base}.gate_proj.scale_ue8m0"] = torch.full(
            (128, 1), 127, dtype=torch.uint8
        )
        tensors[f"{base}.up_proj.scale_ue8m0"] = torch.full(
            (128, 1), 128, dtype=torch.uint8
        )
        tensors[f"{base}.down_proj.scale_ue8m0"] = torch.full(
            (32, 4), 129, dtype=torch.uint8
        )
    save_file(
        tensors,
        path,
        metadata={
            "schema": COUPLED_CHUNK_SCHEMA,
            "role": "physical-codec",
            "layer": "3",
            "expert_range": f"{start}:{stop}",
            "bits": "4",
            "alphabet": "e4m3",
            "block_size": "32",
            "scale": "ue8m0-k32",
            "law": "procedural-mcg-alpha2",
            "boundary": COUPLED_BOUNDARY,
            "activation": COUPLED_ACTIVATION,
            "cast_order": COUPLED_CAST_ORDER,
            "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER,
            "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
            "transform_id": COUPLED_TRANSFORM_ID,
            "encoder_transform_sha256": COUPLED_TRANSFORM_SHA256,
            "sign_generator": COUPLED_SIGN_GENERATOR,
            "sign_draw": str(COUPLED_SIGN_DRAW),
            "sign_pre_axis": "1",
            "sign_post_axis": "2",
            "fc1_interleave": COUPLED_FC1_INTERLEAVE,
            "tp_slice": COUPLED_TP_SLICE,
            "ldlq": "false",
            "encoder": "gptq-feedback-static-in-group-act-order",
            "fc1_trellis_slot_order": "gate-up",
            "fc1_scale_plane_order": "up-gate",
            "coupled_scale_order": "gate_svh-up_svh-down_suh",
            "design_sha256": "a" * 64,
            "exl3_scale_source_sha256": "b" * 64,
        },
    )


def _chunks(tmp_path: Path):
    paths = []
    for start in range(0, 288, 72):
        path = tmp_path / f"chunk-{start:03d}-{start + 72:03d}.safetensors"
        _write_chunk(path, start, start + 72)
        paths.append(path)
    return paths


def _write_packed_fixture(tmp_path: Path, chunks: list[Path]):
    sidecars = []
    receipt = {
        "schema": "glm53-p8-mcg-coupled-scale-tp4-sidecars.v2",
        "layer": 3,
        "world_size": 4,
        "boundary": COUPLED_BOUNDARY,
        "weight_payload_bpw": 4.25,
        "ldlq": False,
        "ranks": [],
    }
    for rank in range(4):
        tensors, sources, design_hash, scale_hash = _load_rank_coupled(
            chunks, layer=3, rank=rank, world_size=4
        )
        weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(tensors)
        metadata = _rank_metadata(
            tensors,
            layer=3,
            rank=rank,
            design_sha256=design_hash,
            scale_source_sha256=scale_hash,
            metadata_bpw=metadata_bpw,
        )
        sidecar = tmp_path / f"rank-{rank}.safetensors"
        save_file(tensors, sidecar, metadata=metadata)
        tensor_hashes = {name: _tensor_sha256(value) for name, value in tensors.items()}
        receipt["ranks"].append(
            {
                "rank": rank,
                "path": str(sidecar.resolve()),
                "bytes": sidecar.stat().st_size,
                "sha256": sha256_file(sidecar),
                "weight_payload_bytes": weight_bytes,
                "metadata_bytes": metadata_bytes,
                "weight_payload_bpw": weight_bpw,
                "metadata_bpw": metadata_bpw,
                "stored_bpw": weight_bpw + metadata_bpw,
                "shapes": {name: list(value.shape) for name, value in tensors.items()},
                "tensor_sha256": tensor_hashes,
                "source_design_sha256": design_hash,
                "exl3_scale_source_sha256": scale_hash,
                "sources": sources,
            }
        )
        sidecars.append(sidecar)
    receipt_path = tmp_path / "packer-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return sidecars, receipt_path


def test_coupled_chunks_build_one_tp_rank_without_changing_weight_abi(tmp_path: Path):
    tensors, sources, design_hash, scale_hash = _load_rank_coupled(
        _chunks(tmp_path), layer=3, rank=2, world_size=4
    )
    weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(tensors)

    assert len(sources) == 4
    assert design_hash == "a" * 64
    assert scale_hash == "b" * 64
    assert tensors["w13_trellis"].shape == (2, 288, 2, 2, 64)
    assert tensors["w2_trellis"].shape == (288, 2, 2, 64)
    assert tensors["w13_scale_ue8m0"].shape == (288, 64, 1)
    assert tensors["w2_scale_ue8m0"].shape == (288, 32, 1)
    assert tensors["intermediate_scales_fp16"].shape == (288, 96)
    assert tensors["coupled_sign_draw_u8"].shape == (288,)
    assert weight_bytes > metadata_bytes > 0
    assert weight_bpw == 4.25
    assert metadata_bpw > 0


def test_production_metadata_contract_is_about_point_zero_zero_four_bpw():
    rate = expected_metadata_bpw(hidden=4096, intermediate=2048)
    assert rate == pytest.approx(0.003979859528718171, abs=1e-18)
    assert 4.25 + rate == pytest.approx(4.253979859528719, abs=1e-15)


def test_rank_header_matches_runtime_full_coupled_contract_and_hashes(tmp_path: Path):
    tensors, _, design_hash, scale_hash = _load_rank_coupled(
        _chunks(tmp_path), layer=3, rank=2, world_size=4
    )
    _, _, _, metadata_bpw = _payload_rates(tensors)
    metadata = _rank_metadata(
        tensors,
        layer=3,
        rank=2,
        design_sha256=design_hash,
        scale_source_sha256=scale_hash,
        metadata_bpw=metadata_bpw,
    )
    assert metadata["schema"] == "glm53-p8-coupled-h512-h128-tp4-rank.v1"
    assert metadata["activation"] == "silu-cap10"
    assert metadata["sign_draw"] == "0"
    assert metadata["global_intermediate"] == "128"
    assert metadata["local_atom_begin"] == "2"
    assert metadata["encoder_transform_sha256"] == COUPLED_TRANSFORM_SHA256
    signs = rank_local_coupled_signs(intermediate=32, rank=2)
    assert metadata["sha256_coupled_signs_fp16"] == _tensor_sha256(signs)
    for name, tensor in tensors.items():
        assert metadata[f"sha256_{name}"] == _tensor_sha256(tensor)


def test_coupled_builder_rejects_any_nonzero_unregistered_draw(tmp_path: Path):
    chunks = _chunks(tmp_path)
    changed = chunks[0]
    with safe_open(changed, framework="pt", device="cpu") as src:
        tensors = {name: src.get_tensor(name) for name in src.keys()}
        metadata = src.metadata()
    tensors["coupled.intermediate_draw_u8"][0] = 1
    save_file(tensors, changed, metadata=metadata)
    with pytest.raises(RuntimeError, match="geometry/dtype"):
        _load_rank_coupled(chunks, layer=3, rank=0, world_size=4)


def test_coupled_builder_fails_closed_on_shared_scale_change(tmp_path: Path):
    chunks = _chunks(tmp_path)
    changed = chunks[-1]
    with safe_open(changed, framework="pt", device="cpu") as src:
        tensors = {name: src.get_tensor(name) for name in src.keys()}
        metadata = src.metadata()
    tensors["coupled.gate_up_suh_fp16"][0] = 3.0
    save_file(tensors, changed, metadata=metadata)

    with pytest.raises(RuntimeError, match="shared coupled scales changed"):
        _load_rank_coupled(chunks, layer=3, rank=0, world_size=4)


def test_coupled_builder_rejects_nonproduction_geometry_at_cli_gate(tmp_path: Path):
    with pytest.raises(RuntimeError, match="hidden size 512, expected 4096"):
        _load_rank_coupled(
            _chunks(tmp_path),
            layer=3,
            rank=0,
            world_size=4,
            expected_hidden=4096,
            expected_intermediate=2048,
        )


def test_postwrite_verifier_reopens_every_rank_and_source_closes_all_tensors(
    tmp_path: Path,
):
    chunks = _chunks(tmp_path)
    sidecars, receipt = _write_packed_fixture(tmp_path, chunks)
    result = verify_postwrite(
        chunks=chunks,
        sidecars=sidecars,
        packer_receipt=receipt,
        layer=3,
        expected_hidden=None,
        expected_intermediate=None,
    )
    assert result["status"] == "pass"
    assert result["retirement_authorized"] is False
    assert result["runtime_loader_closure"] == "not tested"
    assert [row["tensor_count"] for row in result["ranks"]] == [8, 8, 8, 8]
    assert all(row["source_exact"] is True for row in result["ranks"])
    assert all(len(row["shapes"]) == 8 for row in result["ranks"])
    assert all(
        row["dtypes"]["w13_trellis"] == "int16"
        and row["dtypes"]["gate_up_suh_fp16"] == "float16"
        for row in result["ranks"]
    )


def test_postwrite_verifier_rejects_self_consistent_file_hash_with_bad_tensor(
    tmp_path: Path,
):
    chunks = _chunks(tmp_path)
    sidecars, receipt_path = _write_packed_fixture(tmp_path, chunks)
    changed = sidecars[2]
    with safe_open(changed, framework="pt", device="cpu") as src:
        tensors = {name: src.get_tensor(name) for name in src.keys()}
        metadata = src.metadata()
    tensors["w13_trellis"][0, 0, 0, 0, 0] += 1
    save_file(tensors, changed, metadata=metadata)
    receipt = json.loads(receipt_path.read_text())
    receipt["ranks"][2]["bytes"] = changed.stat().st_size
    receipt["ranks"][2]["sha256"] = sha256_file(changed)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimeError, match="tensor w13_trellis differs from source"):
        verify_postwrite(
            chunks=chunks,
            sidecars=sidecars,
            packer_receipt=receipt_path,
            layer=3,
            expected_hidden=None,
            expected_intermediate=None,
        )


def test_postwrite_verifier_rejects_serialized_metadata_drift(tmp_path: Path):
    chunks = _chunks(tmp_path)
    sidecars, receipt_path = _write_packed_fixture(tmp_path, chunks)
    changed = sidecars[1]
    with safe_open(changed, framework="pt", device="cpu") as src:
        tensors = {name: src.get_tensor(name) for name in src.keys()}
        metadata = src.metadata()
    metadata["boundary"] = "identity"
    save_file(tensors, changed, metadata=metadata)
    receipt = json.loads(receipt_path.read_text())
    receipt["ranks"][1]["bytes"] = changed.stat().st_size
    receipt["ranks"][1]["sha256"] = sha256_file(changed)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimeError, match="metadata differs from source-derived"):
        verify_postwrite(
            chunks=chunks,
            sidecars=sidecars,
            packer_receipt=receipt_path,
            layer=3,
            expected_hidden=None,
            expected_intermediate=None,
        )
