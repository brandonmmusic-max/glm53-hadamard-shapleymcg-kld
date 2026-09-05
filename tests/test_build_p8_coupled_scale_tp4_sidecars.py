from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars import (
    _load_rank_coupled,
    _payload_rates,
    expected_metadata_bpw,
)
from glm53_nvfp4.p8_coupled_scale import COUPLED_BOUNDARY, COUPLED_CHUNK_SCHEMA


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
            [expert % 8 for expert in range(start, stop)], dtype=torch.uint8
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
            "activation": "clipped-silu10",
            "cast_order": "bf16-fp16-h512-fp16-suh-h128-e4m3",
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
