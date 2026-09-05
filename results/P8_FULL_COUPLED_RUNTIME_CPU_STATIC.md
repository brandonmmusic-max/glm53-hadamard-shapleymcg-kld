# P8 full-coupled runtime: CPU/static implementation receipt

Date: 2026-09-05
Branch: `p8-coupled-scale-kernel-v1`
Scope: CPU reference, schema, runtime source and static validation only. No
CUDA build, GPU execution, service action, KLD measurement or speed result was
performed while the sealed CF32 baseline was active.

## Outcome

The isolated runtime now has two deliberately distinct modes:

1. `glm53-p8-coupled-scale-component-tp4-rank.v1` is the previously audited
   scale-only H128 sandwich and declares `full_coupled=false`.
2. `glm53-p8-coupled-h512-h128-tp4-rank.v1` is the complete Luke/QSRT-style
   candidate and declares `full_coupled=true`.

The complete mode preserves the K4 procedural-MCG stream, E4M3 weight
alphabet, physical UE8M0/32 scale planes and the native `mxf8f6f4` MMA. It adds
no GEMM and no 64 KiB codebook. The existing deterministic top-k reducer is
replaced, rather than supplemented, by a reducer that owns the final H512.

This is implementation-ready source, not device closure. It must not be called
bit-exact on device, performant, or qualified until it compiles on SM120 and
passes the predeclared device gates.

## Exact full-coupled order implemented

```text
BF16 input -> FP16 -> H512
           -> signed shared suh -> H128 (FP32)
           -> E4M3/UE8M0/32 activation quantization

physical gate/up MMA -> FP16
  -> interleave gate32,up32,gate32,up32
  -> H128 -> private gate/up svh -> H128 -> fixed pre signs
  -> capped SiLU(gate max 10, up range -10..10)
  -> fixed post signs -> H128 -> private down suh -> H128 (FP32)
  -> E4M3/UE8M0/32 activation quantization

physical down MMA -> FP16 -> H128 -> shared down svh
  -> unweighted FP32 route scratch
  -> router-order weighted FP32 top-k sum -> H512 -> BF16 output
```

The two activation-quantized transform boundaries retain FP32 until E4M3
quantization, matching the archived reference's `quantize_activations=True`
branch. The only input storage cast is BF16 to FP16 before outer H512; physical
MMA outputs are stored to FP16 before their following transforms.

## Ownership and synchronization

- Input: one warp owns all four H128 quarters of one H512 block. The normalized
  H4 coupling happens in registers, then each quarter receives `suh`, H128 and
  K32 quantization.
- FC1: one N128 CTA owns both physical projections and every inner H128 needed
  for its 128 down-input values. Cross-layout values are exchanged with warp
  shuffles; a CTA barrier publishes the final FP32 activation tile.
- FC2: one N128 CTA owns a complete K=512 accumulation and its route-local
  H128/down-svh epilogue.
- Output: one 128-thread CTA owns one H512 output block. Four warps compute its
  four H128 quarters, synchronize through 2 KiB of shared FP32 storage, then
  apply normalized H4 and store BF16.

Unsupported non-M1, non-N128 or non-K4 execution remains fail-closed.

## Sidecar and identity gate

The full schema requires exact metadata for normalized H512/H128,
`slot0-atom32-slot1-atom32-v1`, contiguous atom32 TP slicing, sign draw and
axes, target activation, and
`transform_id=normalized-h512-outer-h128-inner-sign-draw0-v1`.

The runtime regenerates the fixed shared rank-local FP16 sign vector using the
archived QSRT seed formula, then verifies its sidecar hash. At GLM TP4/I=512,
the packed `pre[1024]|post[512]` vectors have frozen SHA-256 values:

| rank | SHA-256 |
|---:|---|
| 0 | `95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0` |
| 1 | `95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0` |
| 2 | `95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0` |
| 3 | `95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0` |

Startup also requires an external transform receipt through
`GLM53_P8_NATIVE_TRANSFORM`; its file SHA-256 must equal the sidecar's
`encoder_transform_sha256`. The existing design receipt remains separately
pinned through `GLM53_P8_NATIVE_DESIGN` and `source_design_sha256`.

The full sign table is shared across experts: 1,536 FP16 values = 3,072 bytes
per rank/layer. Combined with the 901,120-byte scale component, the transform
metadata is 904,192 bytes per rank/layer. It is not a weight codebook.

## Expected cost before device measurement

These are operation/traffic bounds, not timing claims:

- Input outer H512 adds nine normalized butterfly stages over 4,096 values per
  decoded token, in addition to the already-required inner H128.
- The joint FC1 boundary adds a second H128 over 2I, two H128s around the
  I-wide down input, two fixed sign multiplies and capped SiLU compared with
  identity. Draw 0 makes the sign values all +1, but their identity remains
  explicit and hash-gated rather than silently removing the preregistered role.
- The output reducer keeps the same one-reducer-launch structure, but route
  scratch grows from BF16 to FP32: 8 routes x 4,096 x 4 bytes = 128 KiB/token,
  versus 64 KiB/token previously. It adds the final H512 arithmetic.
- Tensor-core issue count and native MMA format are unchanged from P8:
  `mxf8f6f4` remains 2x the MMA issue count of NVFP4. No throughput conclusion
  is justified until the five-cold device campaign.

## Validation performed

```text
python3 -m py_compile [all changed Python runtime sources]
git diff --check
PYTHONPATH=. pytest -q tests/test_p8*.py
```

Result: `693 passed` after adding four frozen-sign cases (the last full rerun
must accompany the commit receipt). CPU tests cover schema rejection,
externally pinned transform identity, exact generated sign bytes, H512
algebra, full reference ordering, mode separation and static owner/reducer
seams.

## Pinned-evidence reconciliation

An independent compatibility audit of encoder commit `60d4905` found the
target producer/config contract is draw 0 with GLM-5.3 Flash capped SiLU.
The earlier runtime draft used draw 6 and SiTU solely because those values
appeared in a synthetic archived Luke/QSRT fixture; there was no target
selected-draw receipt supporting that choice. The final runtime therefore
uses draw 0/capped SiLU while retaining the audited coupled topology and FP32
activation-quantization cast points. The schema and transform hash make any
future activation/draw change a fail-closed format change.

## Changed source identities before commit

| file | SHA-256 |
|---|---|
| `runtime_patch/p8_coupled_scales.py` | `5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2` |
| `runtime_patch/p8_native_kernel.py` | `830ed27797c3d6ec1a227df3e04f64ab20fcd938e5ee90b39062fb58621a0b04` |
| `runtime_patch/sitecustomize.py` | `2f9b1384828df9140289c09b372cd6feafc2189eba2ff628242d62655931c098` |
| `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py` | `5486449d247a6c692fc79c769d4cf51bd3c5a82f02756adf7c60a90e6db1e4fd` |
| `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py` | `78a0a896328d5ba0c53287e6fa99cc974091de391fdb0e50e5670a5a226d5cd6` |
| `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py` | `34fe6bcb6822a516ee67da60d46e4f1314b699877a11774b02bbbb8cbbfcf621` |
| `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py` | `632a3bc4174440c04fb28e6dfcefc8876a16938738811c3487f47e6585c4706d` |
| `tests/test_p8_full_coupled_runtime.py` | `c7b2d1b9eebf4ec77ac4fb1a18dd23fa67ac3e7b1b8921c9e5817ea6cd1ba4f2` |

The commit hash remains the authoritative aggregate source identity.
