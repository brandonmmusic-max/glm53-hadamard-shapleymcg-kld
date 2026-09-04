import pytest
import torch

from glm53_nvfp4.trellis_mxf import (
    alphabet_levels,
    nearest_levels,
    pack_ue8m0,
    state_lut,
    expand_xor_t12_lut,
    unpack_ue8m0,
)


def test_native_alphabet_sizes_and_ranges() -> None:
    assert alphabet_levels("e2m1").numel() == 15
    assert alphabet_levels("e2m3").numel() == 63
    assert alphabet_levels("e3m2").numel() == 63
    assert alphabet_levels("e4m3").numel() == 253
    assert float(alphabet_levels("e2m3").abs().max()) == 7.5
    assert float(alphabet_levels("e3m2").abs().max()) == 28.0
    assert float(alphabet_levels("e4m3").abs().max()) == 448.0


def test_nearest_levels_ties_choose_lower() -> None:
    levels = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(
        nearest_levels(torch.tensor([-0.5, 0.5]), levels),
        torch.tensor([-1.0, 0.0]),
    )


def test_fp6_luts_are_exact_e4m3_representable() -> None:
    for alphabet in ("e2m3", "e3m2"):
        lut = state_lut(4, alphabet=alphabet, law="mcg", compander_scale=2.0, device="cpu")
        decoded = lut.view(torch.float8_e4m3fn).float()
        levels = alphabet_levels(alphabet)
        assert bool((decoded[:, None] == levels[None, :]).any(1).all())


def test_ue8m0_scale_roundtrip() -> None:
    scales = torch.pow(2.0, torch.tensor([-20.0, -1.0, 0.0, 7.0]))
    assert torch.equal(unpack_ue8m0(pack_ue8m0(scales)), scales)


def test_xor_t12_table_expands_to_reference_lut() -> None:
    from glm53_nvfp4.trellis_nvfp4 import _sqg_xor_cheb_t12_rank_lut_e4m3

    table = _sqg_xor_cheb_t12_rank_lut_e4m3()
    assert table.numel() == 4096
    assert torch.equal(
        expand_xor_t12_lut(4, table, device="cpu"),
        state_lut(
            4,
            alphabet="e4m3",
            law="sqg-xor-cheb-t12",
            compander_scale=1.0,
            device="cpu",
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA trellis encoder required")
def test_gptq_trellis_payload_closes_against_reference_decoder() -> None:
    from glm53_nvfp4.trellis_mxf import (
        decode_trellis_mxf,
        quantize_trellis_mxf_gptq,
    )

    torch.manual_seed(11)
    weight = torch.randn(16, 32, device="cuda") * 0.1
    samples = torch.randn(64, 32, device="cuda")
    hessian = samples.T @ samples / samples.shape[0]
    payload = quantize_trellis_mxf_gptq(
        weight,
        hessian,
        bits=4,
        alphabet="e4m3",
        law="mcg",
        compander_scale=2.0,
        scale_refinement_iterations=0,
        column_block=32,
    )
    decoded = decode_trellis_mxf(
        payload.trellis,
        payload.codebook_e4m3,
        pack_ue8m0(payload.scales),
        bits=4,
        block_size=32,
        rows=16,
        width=32,
        device="cuda",
    )
    assert torch.equal(decoded, payload.reconstruction)
