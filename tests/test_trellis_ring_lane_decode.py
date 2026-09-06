"""Pure-Python model of the device lane decode against the CPU trellis reference.

The device kernels decode a lane's eight weights from ring words selected by
``trellis256_lane_geom_bits`` and funnel-shifted by ``s2``; this test reproduces that
arithmetic bit for bit (u32 ring words as loaded from the int16 payload, the funnel
shifts, the 16-bit window extraction order of ``packed_decode_mcg2_to_e4m3x8``) and
checks it against ``reconstruct_trellis_states`` for random payloads.

It documents the K5 defect found by the device closure: a lane's eight windows span
16 + 7*bits bits, which at K5 crosses into a third ring word for half the lanes, so the
two-word merge is wrong there.  The three-word rule used by the mixed-rate MCG decoder
must be exact for every lane at every rate, and must reduce to the two-word arithmetic
wherever the span fits in two words.
"""
import numpy as np
import pytest
import torch

from glm53_nvfp4.trellis_nvfp4 import reconstruct_trellis_states, unpack_trellis_edges


def lane_geom(lane: int, bits: int, count: int = 8, offset: int = 0):
    """Port of trellis256_lane_geom_bits: (ia, ib, s2, span)."""
    ring = 8 * bits
    t = 8 * lane + offset
    b1 = (t + 257) * bits
    b0 = b1 - 16
    b2 = b1 + (count - 1) * bits
    i0 = b0 >> 5
    i2 = (b2 - 1) >> 5
    ia = i0 - ring * (i0 >= ring)
    ib = i2 - ring * (i2 >= ring)
    s2 = (i2 + 1) * 32 - b2
    return ia, ib, s2, i2 - i0


def ring_words(packed_int16: np.ndarray) -> list[int]:
    """u32 ring words exactly as ld_shared_u32 sees the int16 payload in memory."""
    u16 = packed_int16.astype(np.uint16)
    return [int(u16[2 * j]) | (int(u16[2 * j + 1]) << 16) for j in range(len(u16) // 2)]


def windows_to_states(win_a: int, win_b: int, bits: int) -> list[int]:
    """The asm's extraction order, returned as elements t+0 .. t+7."""
    w = [(win_a >> (j * bits)) & 0xFFFF for j in range(4)]   # w7, w6, w5, w4
    v = [(win_b >> (j * bits)) & 0xFFFF for j in range(4)]   # w3, w2, w1, w0
    return [v[3], v[2], v[1], v[0], w[3], w[2], w[1], w[0]]


def decode_two_word(words: list[int], lane: int, bits: int) -> list[int]:
    ia, ib, s2, _ = lane_geom(lane, bits)
    merged = (words[ia] << 32) | words[ib]
    return windows_to_states((merged >> s2) & 0xFFFFFFFF, (merged >> (s2 + 4 * bits)) & 0xFFFFFFFF, bits)


def decode_three_word(words: list[int], lane: int, bits: int) -> list[int]:
    """The mixed-rate decoder's K5 rule, written exactly as the kernel computes it."""
    ia, ib, s2, _ = lane_geom(lane, bits)
    ring = 8 * bits
    im = ia + 1 - ring * (ia + 1 >= ring)
    top, mid = (words[ia], words[im]) if ib != im else (0, words[ia])
    lo64 = ((mid << 32) | words[ib]) & (2 ** 64 - 1)
    hi64 = ((top << 32) | mid) & (2 ** 64 - 1)
    t = s2 + 4 * bits
    win_b = (hi64 >> (t - 32)) & 0xFFFFFFFF if t >= 32 else (lo64 >> t) & 0xFFFFFFFF
    return windows_to_states((lo64 >> s2) & 0xFFFFFFFF, win_b, bits)


def _reference(packed: np.ndarray, bits: int) -> list[int]:
    edges = unpack_trellis_edges(torch.from_numpy(packed.copy()), bits)
    return (reconstruct_trellis_states(edges, bits).to(torch.int64) & 0xFFFF).tolist()


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_three_word_lane_decode_matches_the_reference_on_every_lane(bits: int):
    rng = np.random.default_rng(1000 + bits)
    for _ in range(4):
        packed = rng.integers(-32768, 32767, size=16 * bits, dtype=np.int64).astype(np.int16)
        ref = _reference(packed, bits)
        words = ring_words(packed)
        for lane in range(32):
            assert decode_three_word(words, lane, bits) == ref[8 * lane: 8 * lane + 8], (bits, lane)


@pytest.mark.parametrize("bits", [3, 4])
def test_two_word_merge_is_exact_where_every_lane_fits_in_two_words(bits: int):
    rng = np.random.default_rng(2000 + bits)
    packed = rng.integers(-32768, 32767, size=16 * bits, dtype=np.int64).astype(np.int16)
    ref = _reference(packed, bits)
    words = ring_words(packed)
    assert {lane_geom(l, bits)[3] for l in range(32)} == {1}
    for lane in range(32):
        assert decode_two_word(words, lane, bits) == ref[8 * lane: 8 * lane + 8]
        # And the K5 rule reduces to the same arithmetic when the span is two words.
        assert decode_three_word(words, lane, bits) == decode_two_word(words, lane, bits)


def test_k5_spans_three_words_on_half_the_lanes_and_the_two_word_merge_fails_there():
    bits = 5
    spans = [lane_geom(l, bits)[3] for l in range(32)]
    assert sorted(set(spans)) == [1, 2]
    assert spans.count(2) == 16
    rng = np.random.default_rng(5)
    packed = rng.integers(-32768, 32767, size=16 * bits, dtype=np.int64).astype(np.int16)
    ref = _reference(packed, bits)
    words = ring_words(packed)
    wrong = [l for l in range(32) if decode_two_word(words, l, bits) != ref[8 * l: 8 * l + 8]]
    assert wrong == [l for l in range(32) if spans[l] == 2], "exactly the three-word lanes break"
    assert all(decode_three_word(words, l, bits) == ref[8 * l: 8 * l + 8] for l in range(32))
