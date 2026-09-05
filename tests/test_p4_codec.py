from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.numpy import load

from glm53_nvfp4.p4_codec import (
    CONTRACT, P4Payload, SCALE_LAYOUT, SWIZZLED_SCALE_LAYOUT,
    decode_p4, deserialize_p4, e2m1_codes, e2m1_values, e4m3_scale_values,
    fit_e4m3_scales, mcg_half_values, pack_e2m1, pack_k4_edges, pack_k4_states,
    reconstruct_k4_states, scale_byte_offset, serialize_p4, storage_accounting,
    swizzle_scales, unpack_e2m1, unpack_k4_edges, unswizzle_scales,
    verify_operands,
)
from glm53_nvfp4.p4_fixture import (
    make_fixture, scalar_mcg_code, scalar_operands, scalar_state, scalar_swizzle,
    verify_fixture,
)


def sample(rows=32, width=48):
    rng = np.random.default_rng(20260904)
    edges = rng.integers(0, 16, (width // 16, rows // 16, 256), dtype=np.uint8)
    scales = rng.integers(1, 127, (rows, width // 16), dtype=np.uint8)
    return P4Payload(rows, width, pack_k4_edges(edges), scales, np.array(.125, dtype="<f4"))


def edit_header(raw, update):
    size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8:8 + size])
    update(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:]


def test_k4_golden_word_bytes_and_every_sliding_window():
    edges = (np.arange(256) % 16).astype(np.uint8)
    packed = pack_k4_edges(edges)
    assert packed.shape == (64,) and packed.nbytes * 8 == 4 * 256
    assert packed.tobytes() == bytes.fromhex("67452301efcdab89") * 16
    np.testing.assert_array_equal(unpack_k4_edges(packed), edges)
    states = reconstruct_k4_states(edges)
    assert states[:4].tolist() == [0xDEF0, 0xEF01, 0xF012, 0x0123]
    for position in range(256):
        assert states[position] == scalar_state(packed.tobytes(), position)
    np.testing.assert_array_equal(pack_k4_states(states), packed)
    # All 256 positions slide, not just one state per eight weights or per word.
    assert len(set(states[:16].tolist())) == 16


def test_cyclic_windows_do_not_read_neighbor_tiles():
    edges = np.stack((np.zeros(256, dtype=np.uint8), np.full(256, 15, dtype=np.uint8)))
    edges[0, -3:] = [1, 2, 3]
    states = reconstruct_k4_states(edges)
    assert states[0, 0] == 0x1230 and states[1, 0] == 0xFFFF
    altered = states.copy()
    altered[0, 0] ^= 0x10
    with pytest.raises(ValueError, match="recurrence"):
        pack_k4_states(altered)


@pytest.mark.parametrize("bad", [np.zeros((), dtype=np.uint8), np.zeros((0, 256), dtype=np.uint8),
                                  np.zeros(255, dtype=np.uint8), np.zeros(256, dtype=np.float32),
                                  np.full(256, 16, dtype=np.uint8), np.full(256, -1, dtype=np.int16)])
def test_edge_packer_rejects_truncation_and_bad_geometry(bad):
    with pytest.raises(ValueError):
        pack_k4_edges(bad)


def test_existing_exl3_packer_and_layout_compatibility_without_legacy_luts():
    import torch
    from glm53_nvfp4.trellis_nvfp4 import pack_trellis_edges, tensor_core_permutation
    edges = np.random.default_rng(17).integers(0, 16, (3, 2, 256), dtype=np.uint8)
    reference = pack_trellis_edges(torch.from_numpy(edges), 4).numpy()
    assert pack_k4_edges(edges).tobytes() == reference.tobytes()
    positions = np.arange(256)
    lane, slot = positions // 8, positions % 8
    n = lane // 4 + (slot // 4) * 8
    k = (lane % 4) * 2 + slot % 2 + ((slot // 2) % 2) * 8
    np.testing.assert_array_equal(k * 16 + n, tensor_core_permutation().numpy())


def test_all_65536_mcg_states_against_scalar_ieee_oracle_in_small_batches():
    seen = set()
    for start in range(0, 65536, 256):
        states = np.arange(start, start + 256, dtype="<u2")
        actual = e2m1_codes(mcg_half_values(states))
        expected = np.array([scalar_mcg_code(int(state)) for state in states], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
        seen.update(actual.tolist())
    # Alpha-1 MCG reaches +/-4 but not the +/-6 bins. Those native codes
    # remain legal and are covered independently by the conversion tests.
    assert seen == set(range(16)) - {7, 15}


def test_e2m1_all_nibbles_signed_zero_and_native_midpoint_ties():
    codes = np.arange(16, dtype=np.uint8)
    assert pack_e2m1(codes).tobytes().hex() == "1032547698badcfe"
    values = e2m1_values(codes)
    assert values[0] == values[8] == 0
    assert not np.signbit(values[0]) and np.signbit(values[8])
    np.testing.assert_array_equal(e2m1_codes(values), codes)
    np.testing.assert_array_equal(unpack_e2m1(pack_e2m1(codes)), codes)
    mids = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=np.float64)
    expected = np.array([0, 2, 2, 4, 4, 6, 6], dtype=np.uint8)
    np.testing.assert_array_equal(e2m1_codes(mids), expected)
    np.testing.assert_array_equal(e2m1_codes(-mids), expected | 8)
    np.testing.assert_array_equal(e2m1_codes(np.nextafter(mids, -np.inf)), np.arange(7))
    np.testing.assert_array_equal(e2m1_codes(np.nextafter(mids, np.inf)), np.arange(1, 8))
    assert e2m1_codes(np.array([0., -0., -.01, 1e100, -1e100])).tolist() == [0, 8, 8, 7, 15]
    with pytest.raises(ValueError, match="non-finite"):
        e2m1_codes(np.array([np.nan, np.inf, -np.inf]))
    with pytest.raises(ValueError):
        pack_e2m1(np.array([16, 0], dtype=np.uint8))


@pytest.mark.parametrize("shape", [(16, 16), (32, 48), (144, 80)])
def test_rectangular_multi_tile_operand_bytes_match_scalar_inverse_mapping(shape):
    payload = sample(*shape)
    assert verify_operands(payload, scalar_operands(payload)) == {"weight": 0, "scale_e4m3": 0, "global_scale": 0}


def test_scale_alphabet_all_positive_codes_and_reserved_sign_nan_rejection():
    import torch
    raw = np.arange(1, 127, dtype=np.uint8)
    expected = torch.from_numpy(raw).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_array_equal(e4m3_scale_values(raw), expected)
    assert expected[0] == 2**-9 and expected[-1] == 448
    for byte in [0, 127, 128, 129, 254, 255]:
        with pytest.raises(ValueError, match="positive finite"):
            e4m3_scale_values(np.array([byte], dtype=np.uint8))


def test_fixed_code_scale_fit_matches_exhaustive_sse_and_tie_policy():
    rng = np.random.default_rng(19)
    codes = rng.integers(0, 16, (16, 32), dtype=np.uint8)
    source = rng.normal(0, 3, (16, 32)).astype("<f4")
    codes[0] = 2
    source[0, :16], source[0, 16:] = 1.0625, 1.1875
    codes[1] = 8
    codes[2] = 2
    source[2, :16], source[2, 16:] = -4, 1e6
    gs = np.array(1., dtype="<f4")
    actual = fit_e4m3_scales(source, codes, gs)
    assert actual[0].tolist() == [0x38, 0x3A]
    assert actual[1].tolist() == [0x38, 0x38]
    assert actual[2].tolist() == [1, 126]
    values = e2m1_values(codes).astype(np.float64).reshape(16, 2, 16)
    scale_values = e4m3_scale_values(np.arange(1, 127, dtype=np.uint8))
    for row in range(16):
        for group in range(2):
            basis = values[row, group]
            target = source[row, group*16:group*16+16].astype(np.float64)
            scores = [(float(((target - basis * value) ** 2).sum()), code % 2, code)
                      for code, value in enumerate(scale_values, 1)]
            expected = min(scores)[2] if np.any(basis) else 0x38
            assert actual[row, group] == expected


def test_scale_addressing_swizzle_padding_and_modelopt_cpu_compatibility():
    import torch
    from glm53_nvfp4.modelopt import swizzle_block_scale
    payload = sample(144, 80)
    raw = payload.scale_e4m3
    physical = swizzle_scales(raw)
    np.testing.assert_array_equal(physical, scalar_swizzle(raw))
    np.testing.assert_array_equal(physical, swizzle_block_scale(torch.from_numpy(raw)).numpy())
    np.testing.assert_array_equal(unswizzle_scales(physical, rows=144, width=80), raw)
    seen = set()
    for row in range(144):
        for group in range(5):
            for inside in (0, 15):
                logical = scale_byte_offset(row, group*16+inside, rows=144, width=80, layout=SCALE_LAYOUT)
                offset = scale_byte_offset(row, group*16+inside, rows=144, width=80, layout=SWIZZLED_SCALE_LAYOUT)
                assert raw.flat[logical] == physical.flat[offset]
                seen.add(offset)
    assert len(seen) == raw.size
    for offset in set(range(physical.size)) - seen:
        assert physical.flat[offset] == 0
    corrupted = physical.copy()
    corrupted.flat[min(set(range(physical.size)) - seen)] = 1
    with pytest.raises(ValueError, match="padding"):
        unswizzle_scales(corrupted, rows=144, width=80)
    with pytest.raises(ValueError, match="layout"):
        scale_byte_offset(0, 0, rows=144, width=80, layout="ue8m0-k32")
    with pytest.raises(ValueError, match="outside"):
        scale_byte_offset(144, 0, rows=144, width=80, layout=SCALE_LAYOUT)


def test_scale_fitter_rejects_nonfinite_inputs_and_incompatible_shapes():
    weight = np.ones((16, 32), dtype="<f4")
    codes = np.full((16, 32), 2, dtype=np.uint8)
    gs = np.array(.125, dtype="<f4")
    for source, labels, scale in (
        (np.full_like(weight, np.nan), codes, gs),
        (np.full_like(weight, np.inf), codes, gs),
        (weight, codes, np.array(0., dtype="<f4")),
        (weight, codes, np.array([.125], dtype="<f4")),
        (weight, codes[:8], gs),
        (weight, np.full_like(codes, 16), gs),
        (weight.reshape(-1), codes, gs),
    ):
        with pytest.raises(ValueError):
            fit_e4m3_scales(source, labels, scale)


def test_canonical_safetensors_and_exact_byte_rate():
    payload = make_fixture()
    raw = serialize_p4(payload)
    digest = hashlib.sha256(raw).hexdigest()
    decoded = deserialize_p4(raw, expected_shape=(32, 128), expected_sha256=digest)
    assert serialize_p4(decoded) == raw
    ordinary = load(raw)
    for name in ("trellis", "scale_e4m3", "global_scale"):
        assert ordinary[name].tobytes() == getattr(payload, name).tobytes()
    rate = storage_accounting(payload)
    assert rate["trellis_bytes"] == 2048 and rate["scale_e4m3_bytes"] == 256
    assert rate["global_scale_bytes"] == 4 and rate["payload_bytes"] == 2308
    assert rate["file_bytes"] == len(raw)
    assert rate["container_overhead_bytes"] == 8 + struct.unpack_from("<Q", raw)[0]
    assert rate["payload_bpw"]["decimal"] == 4.5 + 32 / 4096
    assert rate["stream_and_scales_bpw"]["decimal"] == 4.5
    ratio = rate["file_bpw"]
    assert Fraction(ratio["numerator"], ratio["denominator"]) == Fraction(len(raw)*8, 4096)


@pytest.mark.parametrize("key,value", [(key, "incompatible") for key in CONTRACT])
def test_every_contract_field_is_enforced(key, value):
    raw = edit_header(serialize_p4(sample()), lambda h: h["__metadata__"].update({key: value}))
    with pytest.raises(ValueError, match=key):
        deserialize_p4(raw)


def test_shape_endianness_layout_and_scale_validation_before_serialization():
    payload = sample()
    invalid = [replace(payload, rows=0), replace(payload, rows=True), replace(payload, width=31),
               replace(payload, rows=16), replace(payload, trellis=payload.trellis.astype(">i2")),
               replace(payload, trellis=payload.trellis[..., ::-1]),
               replace(payload, scale_e4m3=payload.scale_e4m3.T.copy()),
               replace(payload, scale_e4m3=payload.scale_e4m3.astype(np.float32)),
               replace(payload, scale_e4m3=np.zeros_like(payload.scale_e4m3)),
               replace(payload, global_scale=np.array(1., dtype=">f4")),
               replace(payload, global_scale=np.array([1.], dtype="<f4"))]
    invalid += [replace(payload, global_scale=np.array(v, dtype="<f4")) for v in (0., -0., -1., np.nan, np.inf)]
    for value in invalid:
        with pytest.raises(ValueError):
            serialize_p4(value)
    with pytest.raises(ValueError):
        unpack_k4_edges(np.zeros((), dtype="<i2"))


def test_container_corruption_layout_aliases_and_consumer_pins_fail_closed():
    raw = serialize_p4(sample())
    invalid = [b"", raw[:7], raw[:-1], raw + b"\0", raw[:8][::-1] + raw[8:], raw[:-2] + raw[-2:][::-1]]
    invalid += [edit_header(raw, change) for change in (
        lambda h: h["trellis"].update(dtype="U16"),
        lambda h: h["trellis"].update(shape=[2, 3, 64]),
        lambda h: h["global_scale"].update(shape=[1]),
        lambda h: h["global_scale"].update(data_offsets=[False, 4]),
        lambda h: h["scale_e4m3"].update(data_offsets=[5, 101]),
        lambda h: h.update(extra={}),
        lambda h: h["__metadata__"].update(rows="032"),
        lambda h: h["__metadata__"].update(rows=float("inf")),
        lambda h: h["__metadata__"].update(rows=32),
    )]
    for blob in invalid:
        with pytest.raises(ValueError):
            deserialize_p4(blob)
    with pytest.raises(ValueError, match="consumer"):
        deserialize_p4(raw, expected_shape=(48, 32))
    with pytest.raises(ValueError, match="SHA-256"):
        deserialize_p4(raw, expected_sha256="0" * 64)
    header_size = struct.unpack_from("<Q", raw)[0]
    header = raw[8:8+header_size].rstrip()
    duplicate = header[:-1] + b',"trellis":{}}'
    duplicate += b" " * (-len(duplicate) % 8)
    with pytest.raises(ValueError, match="duplicate"):
        deserialize_p4(struct.pack("<Q", len(duplicate)) + duplicate + raw[8+header_size:])


def test_verifier_catches_signed_zero_scale_and_global_byte_changes():
    payload = make_fixture()
    original = decode_p4(payload)
    bad_weight = original.weight.copy()
    negative_zero = np.flatnonzero((bad_weight & 15) == 8)[0]
    bad_weight.flat[negative_zero] ^= 8  # numerically equal +0, different operand
    bad_scale = original.scale_e4m3.copy()
    bad_scale.flat[0] ^= 1
    for expected in (replace(original, weight=bad_weight), replace(original, scale_e4m3=bad_scale),
                     replace(original, global_scale=np.array(.25, dtype="<f4"))):
        with pytest.raises(ValueError, match="byte mismatches"):
            verify_operands(payload, expected)


def test_committed_fixture_is_reproducible_and_independently_verified():
    directory = Path(__file__).resolve().parents[1] / "evidence/opened/codec-v2/p4-astra"
    result = verify_fixture(directory)
    assert result["evidence_level"] == "structural" and result["status"] == "passed"
