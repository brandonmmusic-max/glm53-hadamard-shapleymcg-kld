# P4 CPU operand contract, version 1

`glm53_nvfp4/p4_codec.py` defines `glm53-p4-mcg-matrix.v1`: one physical
K4 stream, positive E4M3 residual scales per 16 weights, and one stored FP32
global multiplier. It decodes to the native NVFP4 **numeric operand bytes**.
It does not implement MMA register loading, activation quantization, a GPU
kernel, or an encoder that chooses good trellis paths for model weights.

The implementation and fixture are structural CPU evidence. GPU arithmetic,
integrated MoE/KLD, determinism on device, and speed require separate closure.
In particular, native operand encodings do not prove that `mxf4nvf4` executed.

## Frozen stream and projection

Both logical dimensions of `W[N,K]` must be positive multiples of 16.
There are `K/16 * N/16` separately wrapped 256-symbol tiles. The stream array
is contiguous little-endian `int16[K/16,N/16,64]`, in the same byte order as
`trellis_nvfp4.pack_trellis_edges(..., 4)` and the P8 trellis sidecars.
The signed int16 dtype is a byte carrier; it is not a signed state label.

For one tile, `e[j]` is a branch nibble in `[0,15]` and

```text
s[j] = e[j] | e[(j-1)%256]<<4 | e[(j-2)%256]<<8 | e[(j-3)%256]<<12
u32[t] = sum(e[8*t+i] << (28-4*i) for i in 0..7)
```

Each `u32` is serialized little-endian. Equivalently, the natural big-bit-order
16-bit words are pair-swapped and then stored as little-endian int16. For
edges `0,1,...,15`, the first eight bytes are `67 45 23 01 ef cd ab 89` and
the first wrapped state is `0xdef0`. No initial-state, tail, or alignment bits
are stored. Cyclic closure is per entire 256-symbol tile, not per 16-symbol
span or eight-weight word. `pack_k4_states` rejects an inconsistent full-state
sequence; `pack_k4_edges` accepts only actual branch nibbles.

For lane `t=j//8` and slot `v=j%8`, the logical coordinates within a tile are

```text
n = t//4 + 8*(v//4)
k = 2*(t%4) + v%2 + 8*((v//2)%2)
```

Tiles are ordered `[k_tile,n_tile]`. This includes the transposition between
the ExLlamaV3 lane permutation and the logical `[N,K]` matrix.

The fixed law is **`procedural-mcg-alpha1-rne-e2m1`**:

```text
p = (s * 0xcbac1fed) mod 2**32
w = (p & 0x8fff8fff) ^ 0x3b603b60
h = round_binary16_RNE(half_from_bits(w & 0xffff) + half_from_bits(w >> 16))
code = round_E2M1_RNE_satfinite(h)       # alpha is exactly 1
```

There is one rounding to binary16 after the addition, followed by E2M1
conversion. There is no intermediate E4M3 rounding, fitted compander, or
state-indexed lookup table. The largest numeric lookup array used by scale
fitting is 126 float64 values (1008 bytes); the state LUT occupies zero bytes.
Exhaustive state tests process 256 states at a time rather than storing an
expanded state-to-label table.

## E2M1 nibble semantics

Codes `0..7` mean `+0, .5, 1, 1.5, 2, 3, 4, 6`; bit 3 is the sign, including
code `8` for negative zero. All 16 codes are valid. The NVIDIA format has no
Inf/NaN encoding; its converting constructors use finite saturation and
nearest rounding. See the [NVIDIA E2M1 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html).
The installed `cuda_fp4.hpp` host conversion independently checks the sign
and ties-to-even rules in this receipt.

Midpoints `.25,.75,1.25,1.75,2.5,3.5,5` map to magnitude codes
`0,2,2,4,4,6,6`. Finite out-of-range values saturate to magnitude 6; negative
underflows preserve negative zero. This reference rejects nonfinite source
values before conversion as an **interchange policy**, rather than relying
on CUDA's nonfinite conversion behavior.

The output is `uint8[N,K/2]` with `W[n,2*i]` in the low nibble and
`W[n,2*i+1]` in the high nibble. The complete ordered code sequence `0..15`
packs to `10 32 54 76 98 ba dc fe`. Every verifier compares bytes, so `-0`
cannot disappear behind numerical equality.

The exhaustive alpha-1 MCG census reaches 14 codes: the `+6` and `-6` bins
are not emitted by this fixed law. They remain valid native nibbles, covered
separately by converter/packer tests. This contract does not assert that an
arbitrary packed NVFP4 matrix can be encoded by the MCG trellis.

## Residual scale bytes and addresses

`scale_e4m3` is a physically stored contiguous `uint8[N,K/16]` plane of raw
positive E4M3FN encodings. Its values are decoded with exponent bias 7,
including subnormals. Codes `0x01..0x7e` span `2**-9..448`. E4M3FN does not
encode infinity and reserves `0x7f` and `0xff` for NaNs, per the
[NVIDIA E4M3 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html).
This contract restricts logical scales to **strictly positive finite** values:
it rejects zero and every sign-bit encoding as well as NaNs. Zero is legal in
the hardware format, but is used only for swizzle padding by this contract.

The reconstructed weight is
`E2M1(code[n,k]) * E4M3(scale[n,k//16]) * global_scale`. The FP32 scalar is
stored even when it equals one. It must be finite and positive, and must be
carried to a future kernel's math/epilogue without pretending the E4M3 scale
already contains it. P4 MMA consumes the per-16 scale plane; an identity
sentinel is not a substitute for these bytes.

`fit_e4m3_scales` minimizes unweighted fixed-code block SSE in float64 by
projecting the least-squares scalar onto the 126 legal positive scale values.
E4M3 midpoint ties choose the even encoding. Negative optima clamp to the
smallest positive scale. An all-zero code block uses `0x38` (1.0), because its
scale cannot change its error. Source weights must be finite FP32. The fitter
is checked against exhaustive discrete SSE; it is neither a trellis encoder
nor an activation/Hessian-weighted optimization.

For a weight at `[n,k]`, set `g=k//16`:

```text
logical scale byte offset = n*(K/16) + g
G = ceil((K/16)/4)*4
128x4 byte offset = ((n//128)*(G//4) + g//4)*512
                 + (n%32)*16 + ((n%128)//32)*4 + g%4
```

`swizzle_scales` produces the separate padded ModelOpt 128x4 layout;
`unswizzle_scales` rejects nonzero padding. The address oracle is checked
against the repository's ModelOpt helper and NVIDIA's
[128x4 scale layout](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/).
Padding is runtime staging storage, not part of this sidecar's stored scale
plane. Any format that persists padding must charge those extra bytes.

## Container and exact stored rate

The container is a canonical, ordinary-reader-compatible safetensors subset:
an unsigned little-endian 64-bit header length, UTF-8 JSON padded with spaces
to a multiple of eight, then three contiguous data regions. JSON keys are
sorted and compact. Tensor regions appear in this order:

| Tensor | Dtype | Shape | Bytes |
|---|---|---|---|
| `global_scale` | F32 | `[]` | 4 |
| `scale_e4m3` | U8 | `[N,K/16]` | `N*K/16` |
| `trellis` | I16 | `[K/16,N/16,64]` | `N*K/2` |

Metadata contains the exact `CONTRACT` fields, decimal dimension strings,
and SHA-256 of every tensor region. There are no learned tables, duplicate
decoded weights, hidden tail states, or per-tile selectors in the payload.
For `W=N*K` weights and measured padded header size `H` bytes:

```text
stream and block scales:       4.5 bpw
payload including FP32 scalar: 4.5 + 32/W bpw
complete sidecar:              4.5 + (32 + 8*(8+H))/W bpw
```

`storage_accounting` reports actual integer bytes and reduced rational bpw
for each total. Exactly 4.5 is only the stream-plus-block-scale rate. K4 P4
does not save payload bytes over an equally scaled packed NVFP4 matrix.
The 32x128 fixture deliberately exposes scalar and container overhead; its
receipt reports its measured totals. Fixture expectations, receipts, and
source code are auxiliary validation artifacts and are not loaded weights.

The loader rejects incorrect/missing/extra metadata, unsupported laws,
P8 scale geometry, endian declarations or dtypes, noncontiguous arrays,
incorrect tensor inventories, shapes, types or offsets, duplicate JSON keys,
noncanonical padding, trailing data, byte-count/hash mismatches, and invalid
scales. Consumers can pin both expected shape and whole-file SHA-256. Hashes
detect accidental corruption and swaps; they do not authenticate a producer
that changes data and recomputes its hashes. Correctly sized arbitrary bytes
cannot reveal a producer's undisclosed transpose or endian mistake without
a trusted external expectation. The fixture verifier also regenerates the
fixed input recipe and scalar expected bytes to catch such substitutions.

## Compatibility and future GPU gate

The old `trellis_nvfp4.e2m1_state_lut` law canonicalizes zero and resolves
midpoints toward smaller magnitudes. Its 65,536-entry encoder table is not
this contract. A stream labeled merely `mcg` is insufficient to identify
either law. Existing encoded P4 artifacts require explicit re-encoding or an
explicitly separately versioned legacy decoder; they must not be relabeled
as this new RNE contract. The legacy APIs are left unchanged.

P8 v2 uses `procedural-mcg-alpha2`, E4M3 output and UE8M0/32 scales. Only its
K4 stream storage geometry is shared. Its sidecar W13 stream is `[gate,up]`,
but its FC1 scale rows are `[up;gate]`; P4 matrix fixtures contain one
projection and make no combined-W13 ordering claim. The existing P8 schema,
launch path, and fused kernel are untouched.

For the intended P4 GPU target, PTX associates `mxf4nvf4` E2M1 operands with
UE4M3 `.scale_vec::4X` scales; see the
[PTX block-scale operand contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
A later `m16n8k64` probe must implement the declared alpha-1 arithmetic,
match every fixture nibble and scale byte, prove its register/lane mapping
and global multiplier, then check actual MMA outputs. GPU kernel timing and
model quality remain separate gates.

Replay from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python3 -m glm53_nvfp4.p4_fixture evidence/opened/codec-v2/p4-astra
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_p4_codec.py tests/test_modelopt.py
# Optional independent NVIDIA conversion oracle: CPU compilation and execution only.
g++ -std=c++17 -O2 -I/usr/local/cuda/include scripts/p4_cuda_host_oracle.cpp -o /tmp/p4-cuda-host-oracle
CUDA_VISIBLE_DEVICES='' python3 -m glm53_nvfp4.p4_fixture evidence/opened/codec-v2/p4-astra --cuda-host-oracle /tmp/p4-cuda-host-oracle
```

`--write` creates the deterministic fixture in a new directory and refuses
overwrites. `fixture.expected.json` contains packed operand and scale hex,
swizzled scale bytes, global-scale bytes, and boundary state probes. A GPU
probe can consume these using only JSON, raw bytes, and safetensors.

## Attribution

The MCG constants, cyclic stream packing, and tile/lane geometry are ported
from ExLlamaV3 (turboderp and contributors, retained `LICENSE.exllamav3`).
The neighboring SQG/encoder work derives from KQuant/QSRT, Luke Alonso and
contributors, audited snapshot `104dd9233f850a3955f4991bea68b07dd34deeb8`.
That snapshot's license remains unverified. The vendored B12X
`w4a8_trellis`/`w4a8_trellis_decode` pipeline is a QSRT port, not original
P4 work. This change adds the independent NumPy/scalar reference, explicit
native rounding contract, serialization, and validation; it does not
relicense those sources. See `THIRD_PARTY_NOTICES.md`.
