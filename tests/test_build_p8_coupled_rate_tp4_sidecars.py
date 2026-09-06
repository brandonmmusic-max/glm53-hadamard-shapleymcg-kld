import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.build_p8_coupled_rate_tp4_sidecars import (
    _load_rank_coupled,
    _payload_rates,
    _rank_metadata,
)
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
)
from glm53_nvfp4.shard_index import sha256_file
from glm53_nvfp4.verify_p8_coupled_rate_tp4_sidecars import verify_postwrite


def _write_chunk(path: Path, start: int, stop: int, *, bits: int):
    hidden, intermediate = 512, 128
    tensors = {
        "coupled.gate_up_suh_fp16": torch.ones(hidden, dtype=torch.float16),
        "coupled.down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
        "coupled.gate_svh_fp16": torch.ones(stop - start, intermediate, dtype=torch.float16),
        "coupled.up_svh_fp16": torch.full((stop - start, intermediate), 2.0, dtype=torch.float16),
        "coupled.down_suh_fp16": torch.full((stop - start, intermediate), -0.5, dtype=torch.float16),
        "coupled.intermediate_draw_u8": torch.tensor([COUPLED_SIGN_DRAW for _ in range(start, stop)], dtype=torch.uint8),
    }
    words = 16 * bits
    for expert in range(start, stop):
        base = f"model.language_model.layers.3.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.trellis"] = torch.zeros(2, 8, words, dtype=torch.int16)
        tensors[f"{base}.up_proj.trellis"] = torch.ones(2, 8, words, dtype=torch.int16)
        tensors[f"{base}.down_proj.trellis"] = torch.full((8, 2, words), 2, dtype=torch.int16)
        tensors[f"{base}.gate_proj.scale_ue8m0"] = torch.full((128, 1), 127, dtype=torch.uint8)
        tensors[f"{base}.up_proj.scale_ue8m0"] = torch.full((128, 1), 128, dtype=torch.uint8)
        tensors[f"{base}.down_proj.scale_ue8m0"] = torch.full((32, 4), 129, dtype=torch.uint8)
    save_file(tensors, path, metadata={
        "schema": COUPLED_CHUNK_SCHEMA, "role": "physical-codec", "layer": "3",
        "expert_range": f"{start}:{stop}", "bits": str(bits), "alphabet": "e4m3", "block_size": "32",
        "scale": "ue8m0-k32", "law": "procedural-mcg-alpha2", "boundary": COUPLED_BOUNDARY,
        "activation": COUPLED_ACTIVATION, "cast_order": COUPLED_CAST_ORDER,
        "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER, "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
        "transform_id": COUPLED_TRANSFORM_ID, "encoder_transform_sha256": COUPLED_TRANSFORM_SHA256,
        "sign_generator": COUPLED_SIGN_GENERATOR, "sign_draw": str(COUPLED_SIGN_DRAW),
        "sign_pre_axis": "1", "sign_post_axis": "2", "fc1_interleave": COUPLED_FC1_INTERLEAVE,
        "tp_slice": COUPLED_TP_SLICE, "ldlq": "false", "encoder": "gptq-feedback-static-in-group-act-order",
        "fc1_trellis_slot_order": "gate-up", "fc1_scale_plane_order": "up-gate",
        "coupled_scale_order": "gate_svh-up_svh-down_suh", "design_sha256": "a" * 64,
        "exl3_scale_source_sha256": "b" * 64,
    })


def _chunks(tmp_path: Path, bits: int, *, mixed: bool = False):
    paths = []
    for index, start in enumerate(range(0, 288, 72)):
        path = tmp_path / f"chunk-{start:03d}-{start + 72:03d}.safetensors"
        _write_chunk(path, start, start + 72, bits=(3 if mixed and index == 2 else bits))
        paths.append(path)
    return paths


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_rate_packer_builds_ranks_at_exact_bits_plus_quarter(tmp_path: Path, bits: int):
    chunks = _chunks(tmp_path, bits)
    tensors, sources, design, scale_source, loaded_bits = _load_rank_coupled(chunks, layer=3, rank=1, world_size=4)
    assert loaded_bits == bits and design == "a" * 64 and scale_source == "b" * 64 and len(sources) == 4
    assert tensors["w13_trellis"].shape[-1] == 16 * bits and tensors["w2_trellis"].shape[-1] == 16 * bits
    weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(tensors)
    assert weight_bpw == bits + 0.25
    metadata = _rank_metadata(tensors, layer=3, rank=1, design_sha256=design, scale_source_sha256=scale_source,
                              metadata_bpw=metadata_bpw, bits=bits)
    assert metadata["bits"] == str(bits) and metadata["weight_payload_bpw"] == format(bits + 0.25, ".17g")
    assert float(metadata["full_coupled_accounted_bpw"]) > bits + 0.25


def test_rate_packer_rejects_mixed_rates_and_unknown_rates(tmp_path: Path):
    with pytest.raises(RuntimeError, match="complete immutable layer"):
        _load_rank_coupled(_chunks(tmp_path, 5, mixed=True), layer=3, rank=0, world_size=4)
    bad = tmp_path / "bad"
    bad.mkdir()
    chunks = _chunks(bad, 4)
    # Rewrite one chunk's rate label to an unsupported value.
    with safe_open(chunks[0], framework="pt", device="cpu") as src:
        tensors = {name: src.get_tensor(name) for name in src.keys()}
        metadata = dict(src.metadata())
    metadata["bits"] = "6"
    save_file(tensors, chunks[0], metadata=metadata)
    with pytest.raises(RuntimeError, match="invalid coupled chunk trellis rate"):
        _load_rank_coupled(chunks, layer=3, rank=0, world_size=4)
    with pytest.raises(ValueError, match="K3, K4 or K5"):
        _rank_metadata({}, layer=3, rank=0, design_sha256="a" * 64, scale_source_sha256="b" * 64, metadata_bpw=0.0, bits=6)


def _write_packed(tmp_path: Path, chunks: list[Path], bits: int):
    sidecars = []
    receipt = {"schema": "glm53-p8-mcg-coupled-scale-tp4-sidecars.v2", "layer": 3, "world_size": 4,
               "boundary": COUPLED_BOUNDARY, "bits": bits, "weight_payload_bpw": bits + 0.25, "ldlq": False, "ranks": []}
    for rank in range(4):
        tensors, sources, design, scale_source, loaded_bits = _load_rank_coupled(chunks, layer=3, rank=rank, world_size=4)
        weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(tensors)
        metadata = _rank_metadata(tensors, layer=3, rank=rank, design_sha256=design, scale_source_sha256=scale_source,
                                  metadata_bpw=metadata_bpw, bits=loaded_bits)
        sidecar = tmp_path / f"rank-{rank}.safetensors"
        save_file(tensors, sidecar, metadata=metadata)
        receipt["ranks"].append({
            "rank": rank, "path": str(sidecar.resolve()), "bytes": sidecar.stat().st_size, "sha256": sha256_file(sidecar),
            "weight_payload_bytes": weight_bytes, "metadata_bytes": metadata_bytes, "weight_payload_bpw": weight_bpw,
            "metadata_bpw": metadata_bpw, "stored_bpw": weight_bpw + metadata_bpw,
            "shapes": {name: list(value.shape) for name, value in tensors.items()},
            "tensor_sha256": {name: _tensor_sha256(value) for name, value in tensors.items()},
            "source_design_sha256": design, "exl3_scale_source_sha256": scale_source, "sources": sources,
        })
        sidecars.append(sidecar)
    receipt_path = tmp_path / "packer-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return sidecars, receipt_path


def test_rate_postwrite_verifier_closes_k5_ranks_and_rejects_rate_drift(tmp_path: Path):
    chunks = _chunks(tmp_path, 5)
    sidecars, receipt_path = _write_packed(tmp_path, chunks, 5)
    result = verify_postwrite(chunks=chunks, sidecars=sidecars, packer_receipt=receipt_path, layer=3,
                              expected_hidden=None, expected_intermediate=None)
    assert result["status"] == "pass"
    receipt = json.loads(receipt_path.read_text())
    receipt["bits"] = 4
    receipt["weight_payload_bpw"] = 4.25
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(RuntimeError):
        verify_postwrite(chunks=chunks, sidecars=sidecars, packer_receipt=receipt_path, layer=3,
                         expected_hidden=None, expected_intermediate=None)
