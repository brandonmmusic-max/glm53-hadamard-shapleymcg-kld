from pathlib import Path

import torch
from safetensors.torch import save_file

from glm53_nvfp4.build_p8_tp4_sidecars import _load_rank


def _write_chunk(path: Path, start: int, stop: int, *, bits: int = 4) -> None:
    tensors = {}
    for expert in range(start, stop):
        base = f"model.language_model.layers.4.mlp.experts.{expert}"
        stream_words = 16 * bits
        tensors[f"{base}.gate_proj.trellis"] = torch.zeros(
            1, 4, stream_words, dtype=torch.int16
        )
        tensors[f"{base}.up_proj.trellis"] = torch.ones(
            1, 4, stream_words, dtype=torch.int16
        )
        tensors[f"{base}.down_proj.trellis"] = torch.full(
            (4, 1, stream_words), 2, dtype=torch.int16
        )
        tensors[f"{base}.gate_proj.scale_ue8m0"] = torch.full(
            (4, 1), 127, dtype=torch.uint8
        )
        tensors[f"{base}.up_proj.scale_ue8m0"] = torch.full(
            (4, 1), 128, dtype=torch.uint8
        )
        tensors[f"{base}.down_proj.scale_ue8m0"] = torch.full(
            (1, 4), 129, dtype=torch.uint8
        )
    save_file(
        tensors,
        path,
        metadata={
            "schema": (
                "glm53-hessian-trellis-p8-layer-chunk.v2"
                if bits == 4
                else "glm53-hessian-trellis-p8-k5-layer-chunk.v1"
            ),
            "role": "physical-codec",
            "layer": "4",
            "expert_range": f"{start}:{stop}",
            "bits": str(bits),
            "alphabet": "e4m3",
            "block_size": "32",
            "scale": "ue8m0-k32",
            "law": "procedural-mcg-alpha2",
            "boundary": "identity",
            "ldlq": "false",
            "encoder": "gptq-feedback-static-in-group-act-order",
            "design_sha256": "a" * 64,
        },
    )


def test_v2_codec_only_chunks_build_one_tp_rank(tmp_path: Path):
    chunks = []
    for start in range(0, 288, 72):
        path = tmp_path / f"chunk-{start:03d}-{start + 72:03d}.safetensors"
        _write_chunk(path, start, start + 72)
        chunks.append(path)

    tensors, sources, schema, design_sha256, boundary, angle_pi = _load_rank(
        chunks, layer=4, rank=2, world_size=4
    )

    assert schema == "glm53-hessian-trellis-p8-layer-chunk.v2"
    assert design_sha256 == "a" * 64
    assert boundary == "identity"
    assert angle_pi is None
    assert len(sources) == 4
    assert tensors["w13_trellis"].shape == (2, 288, 1, 1, 64)
    assert tensors["w2_trellis"].shape == (288, 1, 1, 64)
    assert tensors["w13_scale_ue8m0"].shape == (288, 2, 1)
    assert tensors["w2_scale_ue8m0"].shape == (288, 1, 1)
    # Runtime FC1 row order is [up; gate], and rank 2 preserves physical codes.
    assert torch.equal(
        tensors["w13_scale_ue8m0"][0, :, 0], torch.tensor([128, 127], dtype=torch.uint8)
    )
    assert int(tensors["w2_scale_ue8m0"][0, 0, 0]) == 129


def test_rotated_chunks_preserve_frozen_boundary(tmp_path: Path):
    chunks = []
    for start in range(0, 288, 72):
        path = tmp_path / f"rotated-{start:03d}-{start + 72:03d}.safetensors"
        _write_chunk(path, start, start + 72)
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as src:
            tensors = {name: src.get_tensor(name) for name in src.keys()}
        save_file(
            tensors,
            path,
            metadata={
                "schema": "glm53-rotated-hessian-trellis-p8-layer-chunk.v1",
                "role": "physical-codec",
                "layer": "4",
                "expert_range": f"{start}:{start + 72}",
                "bits": "4",
                "alphabet": "e4m3",
                "block_size": "32",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "boundary": "shared-mid-butterfly-p00625",
                "angle_pi": "0.0625",
                "rotation_arithmetic": "bf16-input-fp32-four-stage-final-bf16",
                "ldlq": "false",
                "encoder": "gptq-feedback-static-in-group-act-order",
                "design_sha256": "b" * 64,
            },
        )
        chunks.append(path)

    _, _, schema, design_sha256, boundary, angle_pi = _load_rank(
        chunks, layer=4, rank=0, world_size=4
    )
    assert schema == "glm53-rotated-hessian-trellis-p8-layer-chunk.v1"
    assert design_sha256 == "b" * 64
    assert boundary == "shared-mid-butterfly-p00625"
    assert angle_pi == 0.0625


def test_k5_chunks_preserve_five_bit_stream_geometry(tmp_path: Path):
    chunks = []
    for start in range(0, 288, 72):
        path = tmp_path / f"k5-{start:03d}-{start + 72:03d}.safetensors"
        _write_chunk(path, start, start + 72, bits=5)
        chunks.append(path)

    tensors, _, schema, design_sha256, boundary, angle_pi = _load_rank(
        chunks, layer=4, rank=1, world_size=4, bits=5
    )

    assert schema == "glm53-hessian-trellis-p8-k5-layer-chunk.v1"
    assert design_sha256 == "a" * 64
    assert boundary == "identity"
    assert angle_pi is None
    assert tensors["w13_trellis"].shape == (2, 288, 1, 1, 80)
    assert tensors["w2_trellis"].shape == (288, 1, 1, 80)
    # Each synthetic tile carries exactly 256 five-bit edges in 80 int16
    # words. Production sidecar construction additionally checks the combined
    # stream-plus-UE8M0 payload against 5.25 bpw using real tensor geometry.
    assert tensors["w13_trellis"].shape[-1] * 16 / 256 == 5.0
