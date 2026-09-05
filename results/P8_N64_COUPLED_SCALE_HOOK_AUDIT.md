# P8 N64 coupled-scale hook audit

Date: 2026-09-05
Scope: read-only design audit; no GPU run, build, or service action was performed.

## Verdict

The proposed scale sandwich is implementable in the fused P8 N64 path without
changing the K4 trellis stream or its physical E4M3/UE8M0-per-32 MMA operand.
The lowest-cost evidence-backed tensor layout is:

| tensor | dtype / shape at TP4 | broadcast | runtime location |
|---|---|---|---|
| `gate_up_suh` | FP16 `[H]` | layer/rank shared across experts and gate/up | before the FC1 H128 and before E4M3 amax/quantization |
| `gate_svh` | FP16 `[E,I]` | expert-private | after the FC1 output H128, before activation |
| `up_svh` | FP16 `[E,I]` | expert-private | after the FC1 output H128, before activation |
| `down_suh` | FP16 `[E,I]` | expert-private | after activation, before down-input H128 and E4M3 amax/quantization |
| `down_svh` | FP16 `[H]` | layer/rank shared across experts | after down-output H128, before routed sum |

For this runtime, `H=4096`, `E=288`, and TP-local `I=512`
(`runtime_patch/p8_native_kernel.py:58-72`). These five arrays total 901,120
bytes per TP rank and layer, about 0.0040 additional bits per routed weight.
They are codec metadata, not replacements for the physical UE8M0 weight scale
planes.

The scale-only sandwich is **not by itself the full Luke/QSRT coupled
topology**. True coupling also requires the residual H512 boundary, the
interleaved FC1 H128/sign sandwich, and the final H512 after the routed sum.
Calling the reduced ordinary-H128 sandwich `trellis_coupled` would conflate two
different transforms.

## Evidence for the tensor layout

The local GLM-5.3 EXL3 K3 checkpoint is direct format evidence, not a proxy
from another architecture:

- file:
  `/media/brandonmusic/klcstore/GLM-5.3-EXL3-TR3-uniform-K3-runtime/exl3-r7-layer-003.safetensors`
- file SHA-256:
  `8206e36ee2bb640b75e4076f8a18672b0028bd5e37db55e2dc0ec680502a84d3`
- expert 0 stores FP16 `gate/up.suh [6144]`, `gate/up.svh [2048]`,
  `down.suh [2048]`, and `down.svh [6144]`.
- `gate.suh == up.suh` bit-for-bit for experts 0, 1, and 255; the shared
  vector hash is
  `6ae7f0619db3f2f678f86d24105fbcf1a89637723d7f72198064ce11151856e2`.
- `down.svh` is likewise shared in those experts; hash
  `8baf4cec14e0409b1fd59e83cf7859408d35ff64fffac5b384e344c3b1eca54b`.
- values are signed, not positive-only: expert-0 `gate.suh` spans
  `[-0.0183868408203125, 0.0193939208984375]`, and `down.svh` spans
  `[-1.345703125, 1.3427734375]`.

The archived GLM loader independently enforces one byte-identical shared
gate/up `suh` and shared down `svh`, then stacks expert-private gate `svh`, up
`svh`, and down `suh`
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/vllm/model_executor/layers/quantization/exl3.py:1555-1616`).
Its prepared ABI requires `[H]`, `[E,I]`, `[E,I]`, `[E,I]`, `[H]`
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/b12x/moe/glm_sqg_w4a8.py:290-321`).

The archived Luke/QSRT closure uses the same FP16 roles. Its reference applies
input `suh`, rounds to FP16, then H128; output `svh` follows H128
(`b12x-coupled-glm52-20260813/tests/moe/test_fused_moe_trellis.py:756-771`).
The K2 closure also asserts that the prepared coupled path aliases gate/up to
one selected `source_suh`
(`b12x-coupled-glm52-20260813/tests/moe/test_fused_moe_trellis.py:1432-1440,1476-1494`).

## Exact ordering

Let vectors be row vectors and let every H128/H512 be normalized and
self-inverse. Ignore only the explicitly identified storage casts below.

### Ordinary EXL3-style scale sandwich

For a projection with transformed physical weight `Wq`:

```text
a  = H128(FP16(x * suh))
z  = MMA(Q_E4M3_UE8M0(a), Wq)
y  = H128(z) * svh
```

The encoder must transform the source weight to the corresponding input and
output bases before trellis quantization. Merely adding runtime scales to the
current identity-coded weights does not preserve the source linear map.

More explicitly, for row vectors define `A = C * D_suh * H_in`, where `C` is
identity for the ordinary path or H512 for the coupled residual input, and
define `B = H_out * D_svh`. Runtime evaluates
`y = x * A * Wq^T * B`. Exact unquantized equivalence to
`y = x * W^T` therefore requires
`Wq = B^(-T) * W * A^(-T)`. Because the normalized Hadamards are symmetric
involutions, this is
`Wq = H_out * D_svh^(-1) * W * C * D_suh^(-1) * H_in`.
The explicit FP16 storage casts make actual closure numerical rather than
symbolically bit-exact. The coupled FC1 pair also interleaves the two physical
outputs around the nonlinearity, so its encoder must follow the archived joint
slot transform rather than apply this single-projection formula independently.

The archived staged GLM path performs the same order: shared input scale then
H128 and MXFP8 quantization
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/b12x/moe/glm_sqg_w4a8.py:500-512`),
H128 then private gate/up scales and activation
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/b12x/moe/_shared/kernels/glm_trellis_transform.py:87-170`),
private down scale then H128
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/b12x/moe/_shared/kernels/glm_trellis_transform.py:174-242`),
and down H128 then shared output scale before top-k sum
(`glm52_sqg_w4a8_runtime_release_20260812/glm52-sqg-w4a8/build-context/overlay/b12x/moe/_shared/kernels/glm_trellis_transform.py:245-313`).

### True Luke/QSRT coupled topology

The outer boundaries add H512 and the inner FC1 path is interleaved:

```text
source      = H512(FP16(x))
fc1_input   = H128(FP16(source * gate_up_suh))
fc1_input_q = Q_E4M3_UE8M0_per32(fc1_input)

physical gate/up MMA
interleaved = H128(physical) -> gate_svh/up_svh -> H128 -> pre-signs
activated   = coupled_gate(up) -> post-signs -> H128
down_input  = H128(FP16(activated * down_suh))
down_input_q= Q_E4M3_UE8M0_per32(down_input)

physical down MMA
route       = H128(FP16(down)) * down_svh
output      = H512(sum_route(route_weight * route))
```

The authoritative reference is
`b12x-coupled-glm52-20260813/tests/moe/test_fused_moe_trellis.py:919-1058`.
The archived device implementation names the input order “shared H512 ...
followed by ordinary H128 inputs” and executes H512, FP16 `suh`, then H128
(`b12x-coupled-glm52-20260813/b12x/moe/_shared/kernels/w4a16/kernel.py:7377-7526`).
Its output first reduces weighted routes, then performs the residual H512
(`b12x-coupled-glm52-20260813/b12x/moe/_shared/kernels/w4a16/kernel.py:8677-8718`).

These operations do not generally commute. In particular:

1. `suh` cannot move across H128 or H512.
2. `svh` cannot move across the output H128 or final H512.
3. An arbitrary per-channel `suh` cannot be absorbed into the one UE8M0 byte
   for a K32 block. It must multiply each channel before block amax and E4M3
   rounding. The resulting UE8M0 byte then scales the already transformed K32
   values.
4. Signed FP16 scales must be accepted; requiring positive values would reject
   the actual GLM-5.3 EXL3 metadata.

## Exact current hooks and missing hooks

### Encoder and sidecar

The production P8 encoder currently quantizes identity-basis gate/up directly,
constructs the down Hessian from an identity carrier, and emits only trellis
plus UE8M0 planes
(`glm53_nvfp4/quantize_hessian_trellis_layer.py:27-44,115-170`). It hard-codes
`boundary=identity` (`:176-190`). A coupled encoder must therefore:

1. create encoder-basis gate/up/down weights using the exact coupled formula;
2. build FC1 and down Hessians from the transformed, E4M3-quantized runtime
   carriers, not `_middle()`'s identity carrier;
3. trellis-quantize those transformed weights with the existing K4 MCG and
   UE8M0/32 functions;
4. save the five FP16 vectors plus immutable sign seed/draw and transform ID.

`glm53_nvfp4/p8_h128.py:71-140` is reusable only for the ordinary uncoupled
H128 sandwich. It is not a full coupled encoder and currently rejects signed
diagonals (`:93-95`).

`glm53_nvfp4/build_p8_tp4_sidecars.py:42-57` accepts only identity chunks;
`:116-126` emits only weight stream/scale tensors; and `:177-190` stamps an
identity sidecar. Add a new schema rather than weakening the v2 identity
checks. Proposed coupled sidecar keys:

```text
gate_up_suh_fp16       [H]
intermediate_scales    [E,3I]  # gate_svh | up_svh | down_suh
down_svh_fp16          [H]
coupled_sign_draw      scalar u8/u32, generated procedurally at runtime
```

At TP4, gate/up output and down-input arrays are sliced on the intermediate
axis; the two `[H]` arrays are replicated because hidden remains full on each
rank. Every array needs a byte hash in sidecar metadata/receipt.

### Fused N64 runtime

`runtime_patch/p8_native_kernel.py:116-166` currently rejects any non-identity
boundary and loads no transform metadata. It passes four all-one scalar arrays
and a one-element zero rotation buffer at `:460-486`.

The M=1 preparation packs one raw activation row exactly once, before it knows
an expert, at
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py:3156-3211`.
That is an exact cheap hook for **one layer-shared gate/up input vector**. It
cannot support per-expert or distinct gate/up `suh` without route-specific
packing or folding those factors into the encoded weights. True H512 also
couples four H128 groups, so the current independent thread-per-K32 preparation
needs a warp/shared-memory H512 stage before the per-channel multiply, H128,
amax, and quantization.

`P8NarrowFC1Kernel` explicitly hard-codes uncoupled identity at
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py:78-90`.
Its E4M3/UE8M0 MMA is at `:534-608`; its current
epilogue directly activates raw accumulators at `:642-674`; and its down-input
K32 amax/quantization is at `:678-720`. Therefore:

- port the Luke/QSRT interleaved coupled epilogue into `:642-674`;
- load `gate_svh`, `up_svh`, and `down_suh` by `[expert, local_col]`;
- apply down `suh` before the down-input H128 and compute `block_max` from the
  final transformed values, not from the pre-scale activation;
- retain `_w4a8_trellis_permute_k32` immediately before
  `quantize_block_fp8_mx`.

The existing `trellis_rotations` argument reaches the N64 phase
(`p8_narrow_fc1.py:309-334,728-770`) but is unused under identity. It can carry
the `[E,3I]` FP16 scale slab. A distinct procedural sign draw is preferable to
doubling it to `[E,6I]`; the latter would violate the intended small-table
contract in spirit even though it is metadata rather than a codebook LUT.

`P8SmallMPhase2Kernel` performs the native down MMA at
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py:215-251`, then
immediately rounds scaled partials to BF16,
weights them, and accumulates routes at `:253-276`. The down-output H128 and
`down_svh` must occur before the routed sum (or, because both are layer-shared,
be moved together into a replacement deterministic top-k reducer before the
final H512). The current wrapper invokes an ordinary top-k reducer at
`runtime_patch/p8_native_kernel.py:488-496`; it has no H128/H512/scale inputs.

The generic dynamic kernel's `trellis_coupled` branch at
`dynamic.py:6131-6141` is only an inner activation-boundary hook. It does not
implement the residual input/output H512, and N64 replaces that generic phase
with specialized `P8NarrowFC1Kernel`/`P8SmallMPhase2Kernel`. Enabling its flag
does not complete the N64 coupled path.

## Predeclared closure gates

No KLD or speed run is admissible until all earlier gates pass.

1. **Schema and shape gate (CPU):** new schema only; exact five FP16 shapes;
   finite and nonzero signed values; gate/up shared identity asserted; hashes
   recorded; wrong/missing transform ID or TP slice fails closed.
2. **Feature-off gate (CPU/device fixture):** identity schema, all-one scales,
   and transforms disabled must leave existing v2 sidecars and frozen N64
   outputs byte-identical. Unsupported N128, monolithic, M2/M3, or prefill
   paths must fail closed rather than silently use identity.
3. **Algebraic encoder gate (CPU):** with trellis and E4M3 quantization
   disabled, compare original BF16 expert output with transformed weights plus
   the exact reference sequence above. Report max absolute, max relative, and
   NMSE using a tolerance predeclared for the explicit FP16/BF16 casts. This is
   numeric closure, not bit identity to the source because the transform adds
   storage casts.
4. **Weight-codec bit gate (CPU):** decoded E4M3 bytes and every UE8M0 weight
   byte must match the encoder reference bit-for-bit. Existing
   `decode_trellis_mxf` equality at
   `quantize_hessian_trellis_layer.py:141-153` remains mandatory.
5. **Activation bit gate (reference first, then device):** for zero, subnormal,
   saturation, alternating-sign, and captured production rows, compare every
   post-H512/FP16-scale/H128 E4M3 payload byte and every UE8M0 K32 byte. The
   reference must use the same cast points and
   `_w4a8_trellis_permute_k32` order. Recomputing amax before `suh` is a fail.
6. **Epilogue bit gate:** capture FC1 pre-transform accumulators, post-coupled
   activation, down-input payload/SF, raw down accumulator, per-route canonical
   output, and final H512 output. Require bit identity to the exact reference
   at every stored FP16/BF16 boundary and stated numeric tolerance only for
   FP32 accumulators.
7. **Pseudoquant closure:** on the same three preregistered layers and windows,
   compare (a) dense decoded coupled carrier, (b) byte-decoded physical codec
   reference, and later (c) native N64 kernel. Separate encoder effect from
   runtime closure; do not use a KLD delta to excuse byte mismatch.
8. **Determinism and serving gate:** five bitwise-identical runs, CUDA-graph
   parity, then matched end-to-end KLD. Only after closure may throughput be
   compared with identity P8, NVFP4, and EXL3.

## Implementation decision to freeze before coding

Choose one named topology:

- `ordinary-h128-suh-svh-v1`: the five-vector EXL3-style sandwich, no H512 or
  coupled signs; or
- `coupled-h512-h128-suh-svh-v1`: the full Luke/QSRT ordering above.

The user's stated goal is the second. The first is a useful, cheaper ablation,
but it is not a substitute and must not inherit the `coupled` label.
