// Experimental SM120 P4 MoE projection kernels. See p4_decode.cuh for ports.
// This translation unit has no framework or dense-weight matmul dependency.
#include "p4_decode.cuh"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math.h>

namespace {
struct Stage {
    uint32_t a[16][8];          // packed E2M1, 16 x 64 values
    uint32_t b[8][8];           // packed E2M1, 8 output rows x 64 inputs
    uint8_t sfa[16][4];         // physical E4M3, four scales per K64
    uint8_t sfb[8][4];
};
static_assert(sizeof(Stage) == 864, "packed operands only");

__device__ __forceinline__ void rendezvous() {
    // A named 64-thread barrier deliberately joins two different warp roles.
    // At iteration t+1 the consumer has finished stage t before that stage
    // can be overwritten at t+2. All stores/loads are ordinary shared memory.
    asm volatile("bar.sync 1, 64;" ::: "memory");
}

__device__ __forceinline__ void mma(float (&d)[4], const Stage& s, int lane) {
    uint32_t a0 = s.a[p4::a_row(lane, 0)][p4::a_k(lane, 0) / 8];
    uint32_t a1 = s.a[p4::a_row(lane, 1)][p4::a_k(lane, 1) / 8];
    uint32_t a2 = s.a[p4::a_row(lane, 2)][p4::a_k(lane, 2) / 8];
    uint32_t a3 = s.a[p4::a_row(lane, 3)][p4::a_k(lane, 3) / 8];
    uint32_t b0 = s.b[p4::b_row(lane)][p4::b_k(lane, 0) / 8];
    uint32_t b1 = s.b[p4::b_row(lane)][p4::b_k(lane, 1) / 8];
    uint32_t sa = *reinterpret_cast<const uint32_t*>(s.sfa[p4::sfa_row(lane)]);
    uint32_t sb = *reinterpret_cast<const uint32_t*>(s.sfb[p4::sfb_row(lane)]);
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "%10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

__device__ __forceinline__ void produce(
    Stage& s, const float* x, const uint32_t* stream, const uint8_t* scales,
    const int64_t* order, int64_t first, int64_t stop, int expert,
    int n0, int k0, int n, int k, int row_divisor, int lane) {
    // Each lane owns one full activation group, then the corresponding row
    // in the upper M8. The A boundary is NVFP4 E2M1 + dynamic E4M3/16.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        int row = lane / 4 + half * 8, block = lane & 3;
        int64_t position = first + row;
        int64_t route = position < stop ? order[position] : 0;
        float v[16], maximum = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = position < stop ? x[(route / row_divisor) * k + k0 + block * 16 + j] : 0.0f;
            maximum = fmaxf(maximum, fabsf(v[j]));
        }
        // Keep a nonzero minimum subnormal scale for tiny nonzero groups.
        // Large groups saturate to 448: this activation policy is explicit
        // and reproduced by the CPU probe, not a quality/performance claim.
        __nv_fp8_e4m3 sf(maximum == 0.0f ? 0.0f : fmaxf(maximum / 6.0f, 0x1p-9f));
        s.sfa[row][block] = sf.__x;
        float inv = maximum == 0.0f ? 0.0f : 1.0f / float(sf);
#pragma unroll
        for (int word = 0; word < 2; ++word) {
            uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                packed |= uint32_t(__nv_cvt_float_to_fp4(v[word * 8 + j] * inv, __NV_E2M1, cudaRoundNearest)) << (j * 4);
            s.a[row][block * 2 + word] = packed;
        }
    }
    int row = lane / 4, block = lane & 3;
#pragma unroll
    for (int word = 0; word < 2; ++word)
        s.b[row][block * 2 + word] = p4::weight_word(stream, expert, n0 + row,
                                                   k0 + block * 16 + word * 8, n, k);
    s.sfb[row][block] = scales[p4::scale_byte_offset(expert, n0 + row, k0 + block * 16, n, k)];
}

// Tiles are grouped by expert via stable route sorting and prefix sums.
// One producer warp and one MMA warp, two stages. FC1 emits gate/up
// activations; FC2 consumes SwiGLU activations. No weight expansion in gmem.
__global__ __launch_bounds__(64) void p4_project_kernel(
    const float* x, const uint32_t* stream, const uint8_t* scales,
    const float* global_scale, const int64_t* order, const int64_t* offsets,
    const int64_t* tile_offsets, float* out,
    int experts, int n, int k, int projections, int row_divisor) {
    int64_t task = blockIdx.y;
    if (task >= tile_offsets[experts]) return;  // whole CTA exits uniformly
    int lo = 0, hi = experts;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (tile_offsets[mid] <= task) lo = mid; else hi = mid;
    }
    int expert = lo, projection = blockIdx.x / (n / 8), n0 = (blockIdx.x % (n / 8)) * 8;
    int64_t first = offsets[expert] + (task - tile_offsets[expert]) * 16;
    int64_t stop = offsets[expert + 1];
    stream += uint64_t(projection) * experts * n * k / 8;
    scales += uint64_t(projection) * experts * n * k / 16;
    __shared__ Stage stages[2];
    int lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        for (int tile = 0; tile < k / 64; ++tile) {
            produce(stages[tile & 1], x, stream, scales, order, first, stop,
                    expert, n0, tile * 64, n, k, row_divisor, lane);
            rendezvous();
        }
    } else {
        float d[4] = {0, 0, 0, 0};
        for (int tile = 0; tile < k / 64; ++tile) {
            rendezvous();
            mma(d, stages[tile & 1], lane);
        }
        float gs = global_scale[projection * experts + expert];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = lane / 4 + (i / 2) * 8, col = n0 + (lane & 3) * 2 + (i & 1);
            if (first + row < stop) {
                int64_t route = order[first + row];
                out[(route * projections + projection) * n + col] = d[i] * gs;
            }
        }
    }
}

__global__ void p4_swiglu_kernel(const float* gu, float* middle, int64_t count, int n, float limit) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        int64_t row = i / n, col = i % n;
        float g = fminf(gu[row * 2 * n + col], limit);
        float u = fminf(fmaxf(gu[row * 2 * n + n + col], -limit), limit);
        middle[i] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ void p4_sum_kernel(const float* routed, const float* weights, float* out,
                              int64_t count, int topk, int hidden) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        int64_t token = i / hidden, col = i % hidden;
        float sum = 0;
        for (int slot = 0; slot < topk; ++slot) {
            int64_t route = token * topk + slot;
            sum = __fadd_rn(sum, __fmul_rn(weights[route], routed[route * hidden + col]));
        }
        out[i] = sum;
    }
}

// Diagnostic only: materializes PACKED nibbles, never dense float weights.
__global__ void p4_decode_probe_kernel(const uint32_t* stream, uint32_t* packed,
                                       int experts, int n, int k) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < uint64_t(experts) * n * k / 8) {
        int input = (i % (k / 8)) * 8;
        int output = (i / (k / 8)) % n;
        int expert = i / (uint64_t(n) * k / 8);
        packed[i] = p4::weight_word(stream, expert, output, input, n, k);
    }
}
}  // namespace

// C ABI keeps offline compilation independent of Torch/CuTe and its driver.
extern "C" int p4_project(
    const float* x, const uint32_t* stream, const uint8_t* scales, const float* gs,
    const int64_t* order, const int64_t* offsets, const int64_t* tiles, float* out,
    int experts, int routes, int n, int k, int projections, int row_divisor, cudaStream_t cuda_stream) {
    if (experts <= 0 || routes <= 0 || n <= 0 || n % 16 || k <= 0 || k % 64 ||
        projections < 1 || projections > 2 || row_divisor <= 0) return int(cudaErrorInvalidValue);
    unsigned max_tiles = (routes + 15) / 16 + experts;
    if (max_tiles > 65535) return int(cudaErrorInvalidConfiguration);
    p4_project_kernel<<<dim3(projections * n / 8, max_tiles), 64, 0, cuda_stream>>>(
        x, stream, scales, gs, order, offsets, tiles, out, experts, n, k, projections, row_divisor);
    return int(cudaGetLastError());
}
extern "C" int p4_swiglu(const float* gu, float* mid, int routes, int n, float limit, cudaStream_t stream) {
    int64_t count = int64_t(routes) * n;
    p4_swiglu_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gu, mid, count, n, limit);
    return int(cudaGetLastError());
}
extern "C" int p4_sum(const float* routed, const float* weights, float* out,
                       int tokens, int topk, int hidden, cudaStream_t stream) {
    int64_t count = int64_t(tokens) * hidden;
    p4_sum_kernel<<<(count + 255) / 256, 256, 0, stream>>>(routed, weights, out, count, topk, hidden);
    return int(cudaGetLastError());
}
extern "C" int p4_decode_probe(const uint32_t* stream, uint32_t* packed,
                                int experts, int n, int k, cudaStream_t cuda_stream) {
    uint64_t words = uint64_t(experts) * n * k / 8;
    p4_decode_probe_kernel<<<(words + 255) / 256, 256, 0, cuda_stream>>>(stream, packed, experts, n, k);
    return int(cudaGetLastError());
}
