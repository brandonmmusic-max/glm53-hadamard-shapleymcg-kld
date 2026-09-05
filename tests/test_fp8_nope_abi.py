from __future__ import annotations

import struct

import pytest
import torch

from glm53_nvfp4.fp8_nope_abi import (
    DATA_BYTES,
    FP8_MAX,
    GLM53_FP8_NOPE_GEOMETRY,
    GROUP_SIZE,
    LATENT_DIM,
    NUM_GROUPS,
    RECORD_BYTES,
    SCALE_OFFSET,
    cache_byte_offset,
    decode_fp8_nope,
    encode_fp8_nope,
    select_fp8_nope_geometry,
)


def test_exact_nope_geometry_has_no_rope_lane() -> None:
    geometry = select_fp8_nope_geometry(
        q_head_dim=512,
        qk_rope_head_dim=0,
        kv_lora_rank=512,
        kv_cache_dtype="fp8_ds_mla",
    )
    assert geometry is GLM53_FP8_NOPE_GEOMETRY
    assert (geometry.data_bytes, geometry.scale_offset) == (512, 512)
    assert (geometry.scale_bytes, geometry.record_bytes) == (16, 528)
    assert geometry.rope_dim == 0


@pytest.mark.parametrize(
    "override",
    [
        {"q_head_dim": 576},
        {"qk_rope_head_dim": 64},
        {"kv_lora_rank": 256},
        {"kv_cache_dtype": "nvfp4_ds_mla"},
    ],
)
def test_geometry_selection_fails_closed(override: dict[str, object]) -> None:
    args: dict[str, object] = {
        "q_head_dim": 512,
        "qk_rope_head_dim": 0,
        "kv_lora_rank": 512,
        "kv_cache_dtype": "fp8_ds_mla",
    }
    args.update(override)
    with pytest.raises(ValueError, match="requires"):
        select_fp8_nope_geometry(**args)  # type: ignore[arg-type]


def test_known_record_bytes_are_bit_exact() -> None:
    latent = torch.zeros((1, LATENT_DIM), dtype=torch.float32)
    # Group 1 has unit scale.  These are fixed E4M3FN encodings, not values
    # derived by this module's decoder.
    values = [-448.0, -2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 448.0]
    latent[0, GROUP_SIZE : GROUP_SIZE + len(values)] = torch.tensor(values)
    # Group 2 has a 0.5 scale; its first value normalizes to E4M3 448 (0x7e).
    latent[0, 2 * GROUP_SIZE] = 224.0
    # Group 3 proves that the inline FP32 scale follows the four value groups.
    latent[0, 3 * GROUP_SIZE] = 1.0

    record = encode_fp8_nope(latent)

    assert tuple(record.shape) == (1, RECORD_BYTES)
    assert record[0, :GROUP_SIZE].tolist() == [0] * GROUP_SIZE
    assert record[0, GROUP_SIZE : GROUP_SIZE + len(values)].tolist() == [
        0xFE,
        0xC0,
        0xB8,
        0xB0,
        0x80,
        0x00,
        0x30,
        0x38,
        0x40,
        0x7E,
    ]
    assert int(record[0, 2 * GROUP_SIZE]) == 0x7E
    assert int(record[0, 3 * GROUP_SIZE]) == 0x7E
    expected_scales = b"".join(
        struct.pack("<f", value) for value in (1.0, 1.0, 0.5, 1.0 / FP8_MAX)
    )
    assert bytes(record[0, SCALE_OFFSET:].tolist()) == expected_scales


def test_encode_matches_independent_group_oracle_byte_for_byte() -> None:
    generator = torch.Generator().manual_seed(20260905)
    latent = torch.randn((5, LATENT_DIM), generator=generator, dtype=torch.float32)
    latent[0, :GROUP_SIZE] = 0

    actual = encode_fp8_nope(latent)
    expected = torch.empty((5, RECORD_BYTES), dtype=torch.uint8)
    for row in range(5):
        for group in range(NUM_GROUPS):
            start = group * GROUP_SIZE
            block = latent[row, start : start + GROUP_SIZE]
            amax = float(block.abs().max())
            scale = torch.tensor(1.0 if amax == 0.0 else amax / 448.0, dtype=torch.float32)
            quant = (block / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            expected[row, start : start + GROUP_SIZE] = quant.view(torch.uint8)
            expected[row, SCALE_OFFSET + 4 * group : SCALE_OFFSET + 4 * group + 4] = (
                scale.reshape(1).view(torch.uint8)
            )
    assert torch.equal(actual, expected)


def test_decode_reconstructs_exact_quantized_values() -> None:
    latent = torch.zeros((1, LATENT_DIM), dtype=torch.bfloat16)
    latent[0, GROUP_SIZE] = 448.0
    latent[0, GROUP_SIZE + 1] = 2.0
    latent[0, 2 * GROUP_SIZE] = 224.0
    latent[0, 3 * GROUP_SIZE] = 1.0

    decoded = decode_fp8_nope(encode_fp8_nope(latent))

    assert decoded.dtype == torch.float32
    assert torch.equal(decoded, latent.float())


def test_record_validation_rejects_wrong_shape_dtype_and_bad_scale() -> None:
    with pytest.raises(ValueError, match="shape"):
        encode_fp8_nope(torch.zeros((1, LATENT_DIM - 1), dtype=torch.float32))
    with pytest.raises(TypeError, match="dtype"):
        encode_fp8_nope(torch.zeros((1, LATENT_DIM), dtype=torch.float16))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.zeros((1, LATENT_DIM), dtype=torch.float32)
        bad[0, 0] = float("nan")
        encode_fp8_nope(bad)
    with pytest.raises(TypeError, match="uint8"):
        decode_fp8_nope(torch.zeros((1, RECORD_BYTES), dtype=torch.int8))
    with pytest.raises(ValueError, match="shape"):
        decode_fp8_nope(torch.zeros((1, RECORD_BYTES + 128), dtype=torch.uint8))
    record = encode_fp8_nope(torch.zeros((1, LATENT_DIM), dtype=torch.float32))
    record[0, SCALE_OFFSET : SCALE_OFFSET + 4] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        decode_fp8_nope(record)


def test_wide_padded_page_address_is_not_narrowed() -> None:
    page_size = 64
    semantic_page_bytes = page_size * RECORD_BYTES
    pooled_tail_bytes = (page_size // 4) * 128 * 2
    page_stride = semantic_page_bytes + pooled_tail_bytes
    high_page = 2**31 // page_stride + 1
    slot = high_page * page_size + 7

    offset = cache_byte_offset(
        slot,
        page_size=page_size,
        page_stride_bytes=page_stride,
    )

    assert offset == high_page * page_stride + 7 * RECORD_BYTES
    assert offset > 2**31
    assert DATA_BYTES == SCALE_OFFSET
