// P4 port of ExLlamaV3's cyclic K4 stream and procedural MCG law.
// Copyright turboderp and contributors: see ../../LICENSE.exllamav3.
// The codec/lane conventions were ported through KQuant/QSRT and vendored
// B12X w4a8_trellis; see ../../THIRD_PARTY_NOTICES.md (including license status).
// Integer implementation and P4 projection: ShapleyMCG, Brandon M. Music.
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define P4_HD __host__ __device__ __forceinline__
#else
#define P4_HD inline
#endif

namespace p4 {
constexpr int kLookupBytes = 0;
constexpr int kScaleVector = 16;
constexpr int kMmaK = 64;

// The frozen MCG mask gives half exponents 12..15. Express these values
// exactly in units of 2^-13, then round the sum to half (nearest/even).
// No floating weight tile, half tensor, or 64-KiB state table is needed.
P4_HD int half_units(uint32_t h) {
    int magnitude = int((h & 1023u) | 1024u) << (((h >> 10) & 31u) - 12u);
    return (h & 0x8000u) ? -magnitude : magnitude;
}

P4_HD uint8_t mcg_code(uint32_t state) {
    uint32_t word = (((state & 65535u) * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    int sum = half_units(word & 65535u) + half_units(word >> 16);
    int a = sum < 0 ? -sum : sum;
    int shift = (a >= 2048) + (a >= 4096) + (a >= 8192) + (a >= 16384);
    if (shift) {
        int q = a >> shift;
        int remainder = a & ((1 << shift) - 1);
        int midpoint = 1 << (shift - 1);
        a = (q + (remainder > midpoint || (remainder == midpoint && (q & 1)))) << shift;
    }
    // v2 ABI: native nearest/even E2M1. At ties whose lower code is odd,
    // advance to the even code. Preserve the sign of tiny negative values
    // even when they round to -0. Exact half cancellation is +0 under RNE.
    uint8_t code = (a > 2048) + (a > 6144) + (a > 10240) + (a > 14336)
                 + (a > 20480) + (a > 28672) + (a > 40960);
    code += (a == 6144 || a == 14336 || a == 28672);
    return uint8_t(code | (sum < 0 ? 8 : 0));
}

// Native adjacent int16 halves are swapped by pack_trellis_edges. On the
// supported little-endian host/device, each uint32 contains eight symbols
// from oldest (MS nibble) to newest (LS nibble). Wrap ONLY inside this tile.
P4_HD uint16_t sliding_state(const uint32_t* tile, int element) {
    int word = element >> 3;
    uint64_t window = (uint64_t(tile[(word + 31) & 31]) << 32) | tile[word];
    return uint16_t(window >> (4 * (7 - (element & 7))));
}

// Inverse of the repository's EXL3 tensor_core_permutation, applied to a
// logical weight W[output, input]; the encoder tile is [input16, output16].
P4_HD int stream_element(int output, int input) {
    int row = input & 15, col = output & 15;
    int lane = (col & 7) * 4 + (row & 7) / 2;
    return lane * 8 + (row & 1) + (row / 8) * 2 + (col / 8) * 4;
}

P4_HD uint64_t tile_word_offset(int expert, int output, int input, int n, int k) {
    return ((uint64_t(expert) * (k / 16) + input / 16) * (n / 16) + output / 16) * 32;
}

P4_HD uint64_t scale_byte_offset(int expert, int output, int input, int n, int k) {
    return (uint64_t(expert) * n + output) * (k / 16) + input / 16;
}

P4_HD uint32_t weight_word(const uint32_t* stream, int expert, int output,
                           int input, int n, int k) {
    const uint32_t* tile = stream + tile_word_offset(expert, output, input, n, k);
    uint32_t packed = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int j = 0; j < 8; ++j)
        packed |= uint32_t(mcg_code(sliding_state(tile, stream_element(output, input + j)))) << (4 * j);
    return packed;
}

// PTX ISA 8.8, dense mma.m16n8k64 register mapping (no P8 K32 permutation).
P4_HD int a_row(int lane, int reg) { return lane / 4 + (reg & 1) * 8; }
P4_HD int a_k(int lane, int reg) { return (lane & 3) * 8 + (reg / 2) * 32; }
P4_HD int b_row(int lane) { return lane / 4; }
P4_HD int b_k(int lane, int reg) { return (lane & 3) * 8 + reg * 32; }
// {byte-id, thread-id} = {0, 0}: lanes 4g and 4g+1 supply A rows g/g+8;
// lane 4g supplies all four B scale bytes for output g.
P4_HD int sfa_row(int lane) { return lane / 4 + (lane & 1) * 8; }
P4_HD int sfb_row(int lane) { return lane / 4; }
}  // namespace p4
