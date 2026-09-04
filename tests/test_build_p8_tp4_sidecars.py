from pathlib import Path

import torch
from safetensors.torch import save_file

from glm53_nvfp4.build_p8_tp4_sidecars import _load_rank


def _write_chunk(path: Path, start: int, stop: int) -> None:
    tensors = {}
    for expert in range(start, stop):
        base = f"model.language_model.layers.4.mlp.experts.{expert}"
        tensors[f"{base}.gate_proj.trellis"] = torch.zeros(1, 4, 64, dtype=torch.uint8)
        tensors[f"{base}.up_proj.trellis"] = torch.ones(1, 4, 64, dtype=torch.uint8)
        tensors[f"{base}.down_proj.trellis"] = torch.full(
            (4, 1, 64), 2, dtype=torch.uint8
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
            "schema": "glm53-hessian-trellis-p8-layer-chunk.v2",
            "role": "physical-codec",
            "layer": "4",
            "expert_range": f"{start}:{stop}",
            "bits": "4",
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

    tensors, sources, schema, design_sha256 = _load_rank(
        chunks, layer=4, rank=2, world_size=4
    )

    assert schema == "glm53-hessian-trellis-p8-layer-chunk.v2"
    assert design_sha256 == "a" * 64
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
