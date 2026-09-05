// Independent CPU conversion oracle using the installed NVIDIA CUDA headers.
// Build with g++, not nvcc. No CUDA runtime/driver linking or GPU execution.
// EXL3 MCG constants: turboderp and contributors, LICENSE.exllamav3.
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (std::strcmp(argv[1], "mcg") == 0) {
        for (uint32_t state = 0; state < 65536; ++state) {
            uint32_t word = ((state * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
            __half lo(__half_raw{static_cast<unsigned short>(word)});
            __half hi(__half_raw{static_cast<unsigned short>(word >> 16)});
            __half sum = __float2half_rn(__half2float(lo) + __half2float(hi));
            __nv_fp4_e2m1 result(sum);
            if (std::putchar(result.__x) == EOF) return 3;
        }
    } else if (std::strcmp(argv[1], "scales") == 0) {
        for (unsigned code = 1; code <= 126; ++code) {
            __nv_fp8_e4m3 scale;
            scale.__x = static_cast<unsigned char>(code);
            float decoded = static_cast<float>(scale);
            uint32_t bits;
            std::memcpy(&bits, &decoded, sizeof(bits));
            for (unsigned shift = 0; shift < 32; shift += 8)
                if (std::putchar((bits >> shift) & 255) == EOF) return 3;
        }
    } else return 2;
    return std::fflush(stdout) == 0 ? 0 : 3;
}
