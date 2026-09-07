# TrellisMX codec and checkpoint contracts

This document separates the portable CPU codec from the larger sealed P8
research artifacts. It is a release contract, not a promise that every listed
research operation is exposed as a general converter.

## 1. Portable P4 matrix format, version 1

Schema: `glm53-p4-mcg-matrix.v1`

This is the first format loaded by the public `trellismx` CLI. It is a matrix
interchange format, not a complete model checkpoint.

### Semantics

- Rows `N` and columns `K` must be positive multiples of 16.
- A K4 cyclic stream stores four-bit branch symbols for each 16×16 tile.
- Each output position reconstructs a 16-bit state from the last four branch
  symbols inside that tile; windows do not cross tile boundaries.
- The MCG law is `procedural-mcg-alpha1-rne-e2m1`: uint32 wrap/mask/xor/add,
  one correctly rounded binary16 addition, then nearest-even E2M1 projection.
  Signed zero is preserved.
- The output is packed E2M1 nibbles. Even columns use the low nibble.
- Scales are raw positive E4M3FN bytes in logical `[N,K/16]` order.
- A positive finite little-endian FP32 global scale is always present and is
  not folded into block-scale bytes.

The detailed tile/lane and scale-swizzle equations are retained in
[P4_CODEC.md](P4_CODEC.md). That document remains authoritative for byte-level
behavior; this page adds the package-level boundary.

### Safe loading

`read_p4` rejects missing or duplicate fields, unsupported metadata, wrong
dtypes or shapes, noncanonical padding, trailing bytes, invalid tensor offsets,
hash mismatches, non-positive or non-finite scales, and unexpected tensors.
Callers may pin both expected shape and whole-file SHA-256.

A content hash detects corruption and accidental swaps. It does not prove that
an arbitrary producer chose the intended logical transpose or scale convention;
external expectations or the deterministic fixture are still needed.

### Storage accounting

For `W=N*K` logical weights and measured header bytes `H`:

- trellis stream plus E4M3/K16 scales: 4.5 payload bits per weight;
- global scalar adds `32/W` bits per weight;
- the complete container adds `(32 + 8*(8+H))/W` bits per weight.

Do not collapse these into one "4.5 bpw model" claim.

## 2. Sealed P8 research checkpoint contract

The current P8 family is **TrellisMX-P8**:

| Property | Contract |
|---|---|
| Stored branch rates | K3, K4, K5 in the mixed-rate sealed path |
| Decoded compute operand | E4M3 |
| MMA | `mxf8f6f4` |
| Block scales | UE8M0, K32 |
| Law | `procedural-mcg-alpha2` |
| Coupled boundary | `coupled-h512-h128-suh-svh-v1` |
| Full checkpoint schema | `glm53-p8-full-coupled-all42-tp4-checkpoint.v1` |
| Coupled rank schema | `glm53-p8-coupled-h512-h128-tp4-rank.v1` |
| Tensor-parallel ranks | four |
| Routed layers | 3 through 44 |

### CPU reference codec

The public `trellismx.protocol` API exposes the proven CPU-safe P8 primitives:

- K3/K4/K5 edge packing and unpacking;
- cyclic state reconstruction;
- 65,536-state procedural-MCG alpha-2 to E4M3 table;
- UE8M0/K32 scale encoding and decoding;
- dense pseudoquant reference decoding.

For a payload with `T_k = width/16` input tiles and `T_n = rows/16` output
tiles, `trellis` has shape `[T_k,T_n,16*K]`. Random access reconstructs each
state from the cyclic branch symbols in its native 16×16 tile. The K5 device
decoder adds a third ring-word read for half the lanes; any independent device
implementation must preserve that corrected rule and pass bit-exact closure.

These APIs do not execute the Hessian-aware TrellisMX coupled encoder. That
encoder is CUDA-only and needs explicit operator authorization. Alternate
trellis-search implementations are optional plugin details, not a requirement
of the format.

### Portable model-independent P8 tensor container

Schema: `trellismx.p8-tensors.v1`

This safetensors container stores one trellis and one UE8M0 scale tensor for
each named logical matrix:

```text
tensor.<name>.trellis        int16 [width/16, rows/16, 16*K]
tensor.<name>.scale_ue8m0    uint8 [rows, width/32]
```

The deterministic procedural codebook is reconstructed from the frozen
protocol; it is not duplicated in each file. Metadata records the full law,
alphabet, MMA family, scale geometry, compander, per-record dimensions/rate,
and tensor hashes. The loader enforces an exact tensor inventory and never
uses pickle. `file_bpw` includes header overhead and is reported separately
from payload bpw.

### Production full-coupled TP4 sidecar validation

`trellismx.validate_sidecar` validates the current GLM sidecar contract:

- schema `glm53-p8-coupled-h512-h128-tp4-rank.v1`;
- layers 3–44, ranks 0–3, world size four;
- K3/K4/K5;
- E4M3, UE8M0/K32, procedural MCG alpha2;
- complete coupled H512/H128/suh/svh/sign identity;
- exact eight-tensor inventory and shapes;
- dtype and metadata checks;
- per-tensor hashes and runtime-regenerated coupled-sign hash;
- production `K + 0.25` payload-rate accounting.

The validator reports `runtime_loader_closure: not tested`; it does not import
or launch the CUDA runtime.

The full coupled checkpoint stores four rank sidecars per layer. Weight tensors
are `w13_trellis`, `w2_trellis`, `w13_scale_ue8m0`, and `w2_scale_ue8m0`.
Coupled metadata tensors are `gate_up_suh_fp16`, `down_svh_fp16`,
`intermediate_scales_fp16`, and `coupled_sign_draw_u8`. Manifests bind source
design hashes, layer rates, shapes, and transform identity. The runtime must
refuse a mismatched image, rate, or transform.

P8 is not native NVFP4 compute. Its decoded operands are E4M3. Consequently
the `mxf8f6f4` path has twice the MMA issue count of NVFP4 for equivalent K.

### Current support boundary

The P8 implementation is research-campaign source plus sealed artifacts. The
public CLI intentionally does not advertise arbitrary checkpoint conversion
because exact calibration, scale, transform, Hessian, campaign artifact, and
runtime identity requirements are not yet packaged behind a stable general API.
The smallest faithful next release can add a typed P8 sidecar inspector without
weakening that boundary.

## 3. Numerical and execution contracts

CPU bit-exactness means exact packed stream/operand closure. It does not mean
floating-point GEMM accumulation is universally byte-identical on every device.
Device closure, full-model KLD, CUDA graph parity, repeated determinism, and
performance are separate gates and must be executed explicitly.

Mixed-rate serving must key its compiled specialization on the stored rate.
A K4 path must not silently serve K3 or K5 data. Unsupported shapes and
fallbacks must fail closed or be explicitly observable.

In the measured mixed-rate runtime, grouped M64 prefill is K4-only; K5 layers
serve rows larger than one through the M1 path. This is an implementation
boundary, not a claim that K5 has equivalent grouped-prefill throughput.
