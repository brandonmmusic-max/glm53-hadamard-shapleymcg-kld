// CPU execution of the exact header used in p4_moe.cu; no CUDA dependency.
#include "../runtime_patch/p4/p4_decode.cuh"
extern "C" {
void host_codes(uint8_t* out) {
    for (uint32_t s = 0; s < 65536; ++s) out[s] = p4::mcg_code(s);
}
void host_states(const uint32_t* tile, uint16_t* out) {
    for (int i = 0; i < 256; ++i) out[i] = p4::sliding_state(tile, i);
}
void host_decode(const uint32_t* stream, uint32_t* out, int e, int n, int k) {
    for (int expert = 0; expert < e; ++expert)
        for (int row = 0; row < n; ++row)
            for (int col = 0; col < k; col += 8)
                out[(expert * n + row) * (k / 8) + col / 8] =
                    p4::weight_word(stream, expert, row, col, n, k);
}
uint64_t host_scale_offset(int e, int n, int k, int rows, int cols) {
    return p4::scale_byte_offset(e, n, k, rows, cols);
}
// Coordinates actually used by the production register loads.
void host_fragment_coords(int* a, int* b, int* sf) {
    for (int lane = 0; lane < 32; ++lane) {
        for (int reg = 0; reg < 4; ++reg)
            for (int j = 0; j < 8; ++j)
                a[(lane * 4 + reg) * 8 + j] = p4::a_row(lane, reg) * 64 + p4::a_k(lane, reg) + j;
        for (int reg = 0; reg < 2; ++reg)
            for (int j = 0; j < 8; ++j)
                b[(lane * 2 + reg) * 8 + j] = p4::b_row(lane) * 64 + p4::b_k(lane, reg) + j;
        sf[lane * 2] = p4::sfa_row(lane);
        sf[lane * 2 + 1] = p4::sfb_row(lane);
    }
}
}
