# Experimental P4 runtime contract

`runtime_patch/p4_native_kernel.py:P4NativeTPMoE` supplies the CUDA projections
used by GLM's selected routed layers through
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p4_native.py`. The actual
factory/load/forward integration is described in [P4_GLM_SERVING.md](P4_GLM_SERVING.md).
P8 and the CuTe dynamic backend are unchanged. Evidence is host/structural
and offline compilation only; device execution and speed remain untested.

Each routed projection uses a two-warp CTA: a producer decodes cyclic K4
trellis windows to packed E2M1 and gathers physical E4M3/16 scale bytes while
the consumer issues `mxf4nvf4` `m16n8k64`. Two shared-memory stages separate
ownership; a 64-thread named barrier publishes each stage and prevents reuse
before the previous consumer read completes. The next decode can overlap
the preceding MMA. Actual overlap has not been profiled.

Weights never become dense floating tensors. Gate/up, SwiGLU and down
**activations** use FP32 scratch; input/output activations are BF16. FC1 reads
BF16 directly into a native E2M1/E4M3/16 activation prepass. FC2 quantizes the
FP32 SwiGLU output once. All N8 tiles reuse those packed activation operands.
This is a pair of fused trellis projections plus activation/reduction kernels,
not one machine instruction or a single-kernel entire MoE. Activation scratch
and stable-sort outputs are reused per shape and CUDA stream; returned model
outputs have independent storage.

The kernel's integer MCG law reproduces the two IEEE-half components and
nearest/even half addition exactly in units of 2^-13. It then projects to
E2M1 with the matrix codec's **nearest/even saturating** rule and preserves
negative zero. Compander is exactly 1, with zero lookup bytes. The former v1
ties-to-smaller-magnitude/canonical-zero ABI is rejected, not reinterpreted.
The CPU oracle instead uses `struct` IEEE-half arithmetic and explicit symbol
reconstruction; all 65,536 states and all logical projection bytes are checked
against it and `p4_codec.e2m1_codes(mcg_half_values(...))`. The exhaustive
v2 state census has SHA256
`195d9e9aac6dca94828fa8f693e9bcbba6804f8e160566ce02cad7ba742b7ba1`, including
5,432 negative-zero codes. This tests the shared header on
CPU; device byte equivalence remains a separate gate.

The activation policy is RN-even E2M1 with dynamic E4M3 scales for each 16
values, `amax / 6` rounded to E4M3 and clamped to `[2^-9,448]` for nonzero
groups; zero groups have zero scale. It is reproduced by the CPU reference.
It has not been qualified against the stock NVFP4 activation recipe.

## Physical sidecar ABI

All tensors are contiguous and validated on CPU. `E` is the expert count,
`H` the hidden width and `I` the TP-local intermediate width; H and I must be
positive multiples of 64. W13 is **gate,up in both streams and scales**.

| Key | Dtype | Shape |
|---|---|---|
| `w13_trellis` | int16 | `[2,E,H/16,I/16,64]` |
| `w2_trellis` | int16 | `[E,I/16,H/16,64]` |
| `w13_scale_e4m3` | uint8, raw E4M3 | `[2,E,I,H/16]` |
| `w2_scale_e4m3` | uint8, raw E4M3 | `[E,H,I/16]` |
| `w13_global_scale` | float32 | `[2,E]` |
| `w2_global_scale` | float32 | `[E]` |

Within a projection, trellis tile order is `[E,K/16,N/16]`. A tile contains
64 swapped-half int16 words (128 bytes). Logical scale byte offset is
`(expert*N + output)*(K/16) + input/16`. Scales are not repacked or replaced
by identity. Four successive K16 scale bytes feed each K64 MMA with
`scale_vec::4X`, `ue4m3` and selectors `{0,0}` for A and B. Per-projection,
per-expert global scales multiply the FP32 accumulators in the epilogue.
Payload is exactly 4.5 bpw before the three FP32 global scalars per expert
and container/metadata overhead. The synthetic fixture includes those
scalars at 4.50390625 bpw. No cross-tile state padding is stored.

Required metadata is enforced in `validate_payload`: schema
`glm53-p4-mcg-tp-rank.v2`, role `physical-codec`, layer/rank and TP4 identity,
K4, `e2m1`, `e4m3-k16`, `procedural-mcg-alpha1-rne-e2m1`, compander `1`,
rounding `nearest-even-satfinite`, `signed_zero=preserve`, positive physical
scale bytes `1..126`, identity boundary, W13 order `gate,up`, exact dimensions,
lane/state/expert/shard layouts, `ldlq=false`, and a caller-pinned
`source_design_sha256`. P8 sidecars and
rotated/learned-codebook P4 candidates are rejected. This work supplies no
qualified model sidecar. The separately versioned TP4 bridge produces these
six tensors from compatible v1 matrix-codec containers.

## Verification and launch seams

CPU/static validation (does not load the CUDA library or query a device):

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python3 -m pytest -q -s tests/test_p4_native_kernel.py tests/test_p4_glm_serving.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m glm53_nvfp4.probe_p4_native --output /tmp/p4-reference.json
```

The tests compile the exact device source to PTX and cubin, inspect all MMA
sites, and link the full C launch ABI without loading it. The shared library
must use explicit `-gencode=arch=compute_120a,code=sm_120a`; the nvcc 13.2
`-arch=sm_120a` shorthand includes a generic target when linking that cannot
accept this MMA. The initial failed link log is retained in the receipt.

After parent review and separate device authorization, the synthetic probe
can exercise decode-byte equality, FC1/SwiGLU/FC2/output arithmetic, empty
experts, partial M tiles, and repeated hashes:

```sh
python3 -m glm53_nvfp4.probe_p4_native --device-closure --device cuda:0 \
  --tokens 1 3 17 33 --repeats 5 --output /tmp/p4-device.json
```

That command was **not run** in the agent phase. The explicit runtime API
loads only the six physical tensors and accepts BF16 X, FP32 route weights,
and integer expert IDs. The GLM adapter hooks the ordinary post-load and
modular-forward path. It loads the CUDA functions during weight initialization
before graph capture, uses the current CUDA stream, and retains workspace
buffers. CUDA graph execution parity, scheduling overhead, real dimensions, sanitizer checks,
thermals, integrated KLD and matched TP4 speed/determinism remain open.

The optimized source launches at most
`min(routes, floor((routes + 15*experts)/16))` M16 tiles per output tile.
For C1/topk8/E288 this is 8 Y blocks instead of 289; it is an exact count-bound
reduction, not a measured latency gain. The optimized projection compiles to
62 registers, 1,728 bytes of shared memory, and zero spills under the recorded
CUDA 13.2 compiler. Every assembled matrix instruction is native
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.

P4 targets NVFP4's K64 issue class. P8 uses `mxf8f6f4` K32, twice the MMA
issue count for the same K span; it remains the separate quality product.
No prefill/decode speed or hidden-decode-overlap claim follows from assembly.

## Attribution and ISA source

ExLlamaV3's procedural constants, cyclic bitstream and tensor-core tile
permutation are ported through the local KQuant/QSRT codec. The producer and
MMA separation follows the vendored B12X `w4a8_trellis` approach; its P8 K32
lane permutation, T12 table and E4M3 decoder are not reused as P4 operands.
The C++ implementation and C launch ABI are new. Retain `LICENSE.exllamav3`,
`runtime_patch/b12x_h16/LICENSE.b12x`, and `THIRD_PARTY_NOTICES.md`, including
the KQuant/QSRT snapshot's unverified license status. No LDLQ/BlockLDLQ is
implemented or used.

Operand/register and scale-selector mapping follows NVIDIA's
[PTX ISA 8.8: block scaling](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html#warp-level-block-scaling)
and
[dense m16n8k64 fragments](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16864).
