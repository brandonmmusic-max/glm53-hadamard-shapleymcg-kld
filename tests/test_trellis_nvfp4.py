import hashlib

import torch
import pytest

from glm53_nvfp4.modelopt import E2M1_LEVELS
from glm53_nvfp4.trellis_nvfp4 import (
    decode_trellis_endpoint,
    e2m1_rank_lut,
    e2m1_state_lut,
    pack_trellis_edges,
    pack_two_bit_selectors,
    procedural_state_values,
    quantize_trellis_nvfp4_gptq,
    reconstruct_trellis_states,
    sqg_rank_permutation,
    sqg_xor_cheb_t12_e4m3_lut,
    sqg_xor_rank_permutation,
    tensor_core_permutation,
    unpack_trellis_edges,
    unpack_two_bit_selectors,
    validate_e2m1_lut,
)


def test_sqg_rank_is_bijective_and_e2m1_lut_is_exact():
    rank = sqg_rank_permutation(3)
    assert torch.equal(torch.sort(rank).values, torch.arange(1 << 16))
    raw = e2m1_rank_lut(3, compander_scale=1.25)
    decoded = raw.view(torch.float8_e4m3fn).float()
    legal = torch.cat((-E2M1_LEVELS[1:].flip(0), E2M1_LEVELS))
    assert bool((decoded[:, None] == legal[None, :]).any(1).all())
    assert 6.0 in decoded and -6.0 in decoded


def test_trellis_pack_is_exact_three_bpw_and_roundtrips_edges():
    generator = torch.Generator().manual_seed(123)
    states = torch.randint(-(1 << 15), 1 << 15, (7, 256), generator=generator)
    packed = pack_trellis_edges(states, 3)
    assert packed.shape == (7, 48)
    assert packed.numel() * packed.element_size() * 8 == states.numel() * 3
    expected = (states.to(torch.int64) & 7).to(torch.int16)
    assert torch.equal(unpack_trellis_edges(packed, 3), expected)


def test_mcg_and_mul1_projections_are_finite_exact_e2m1():
    for law in ("mcg", "mul1"):
        source = procedural_state_values(law)
        assert source.dtype == torch.float16 and torch.isfinite(source).all()
        raw = e2m1_state_lut(4, law=law, compander_scale=1.0)
        decoded = raw.view(torch.float8_e4m3fn).float()
        legal = torch.cat((-E2M1_LEVELS[1:].flip(0), E2M1_LEVELS))
        assert bool((decoded[:, None] == legal[None, :]).any(1).all())


def test_cyclic_state_reconstruction_and_tensor_core_permutation():
    edges = (torch.arange(256) % 8).to(torch.int16)[None]
    states = reconstruct_trellis_states(edges, 3)
    assert torch.equal((states.to(torch.int64) & 7).to(torch.int16), edges)
    permutation = tensor_core_permutation()
    assert torch.equal(torch.sort(permutation).values, torch.arange(256))


def test_mature_sqg_matches_frozen_kquant_source_bytes():
    expected = {
        3: "afe7b3633e7d243b00b379b18ec4dca573722b3727cafef47fcb6470d7e7e6c9",
        4: "5a9620f0c4d8f0a60d0b6fbea921dcecbd193e6febf31043aaa6403c20389c2f",
    }
    for bits, digest in expected.items():
        rank = sqg_xor_rank_permutation(bits)
        assert torch.equal(torch.sort(rank).values, torch.arange(1 << 16))
        raw = sqg_xor_cheb_t12_e4m3_lut(bits)
        assert hashlib.sha256(raw.numpy().tobytes()).hexdigest() == digest
        projected = e2m1_state_lut(bits, law="sqg-xor-cheb-t12")
        assert validate_e2m1_lut(projected).shape == (1 << 16,)


def test_custom_lut_rejects_non_e2m1_labels():
    raw = e2m1_state_lut(3, law="mcg")
    raw[123] = torch.tensor(0.25, dtype=torch.float8_e4m3fn).view(torch.uint8)
    with pytest.raises(ValueError, match="non-E2M1"):
        validate_e2m1_lut(raw)


def test_two_bit_selector_pack_roundtrip_and_exact_rate():
    selectors = (torch.arange(35).reshape(5, 7) % 3).to(torch.uint8)
    packed = pack_two_bit_selectors(selectors)
    assert packed.numel() == 9
    assert torch.equal(unpack_two_bit_selectors(packed, (5, 7)), selectors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Viterbi encoder required")
def test_gptq_trellis_p4_decodes_bit_exactly_to_nvfp4():
    generator = torch.Generator(device="cuda").manual_seed(20260904)
    weight = torch.randn((16, 32), generator=generator, device="cuda") * 0.05
    samples = torch.randn((64, 32), generator=generator, device="cuda")
    hessian = samples.T @ samples / samples.shape[0]
    payload = quantize_trellis_nvfp4_gptq(
        weight,
        hessian,
        global_scale=weight.abs().max() / (6.0 * 448.0),
        bits=4,
        scale_refinement_iterations=1,
    )
    decoded = decode_trellis_endpoint(payload)
    assert torch.equal(decoded.weight, payload.endpoint.weight)
    assert torch.equal(decoded.weight_scale, payload.endpoint.weight_scale)
    assert payload.trellis_bpw == 4.0
    assert payload.scale_bpw == 0.5
