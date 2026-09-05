# P8 coupled M1 input-carrier diagnosis

Date: 2026-09-05

Status: CPU/source diagnosis after failed SM120 closure; no GPU, service,
image, runtime source, fixture, protocol, or acceptance-gate change

## Result

The first closure result overstated the carrier failure. Both payload equality
checks compared tensors with different shapes, so `torch.equal` returned false
without proving any byte mismatch:

| Carrier | Device shape | Reference shape |
|---|---:|---:|
| input payload | `[1,4096]` | `[1,128,32]` |
| middle payload | `[8,512]` | `[8,16,32]` |

The corrected diagnostic compares flattened bytes while retaining both shapes.
It finds only **3 of 4,096** input-payload bytes different and **0 of 128**
input-scale bytes different. The exact mismatching packed byte indices are
`800`, `1988`, and `2000`. Each mismatch is between adjacent finite E4M3 codes:

| Packed index | Device byte/value | CPU byte/value |
|---:|---:|---:|
| 800 | `99` / `44.0` | `98` / `40.0` |
| 1988 | `145` / `-0.03515625` | `144` / `-0.03125` |
| 2000 | `95` / `30.0` | `96` / `32.0` |

This pattern, together with exact UE8M0 bytes, is evidence for a numerical
operation-order mismatch at E4M3 rounding boundaries, not an H512/H128 or K32
layout permutation. It is not proof that a proposed source repair will close
all three bytes; the unchanged exact gate must decide that on device.

The middle carrier remains a separate, material failure: 4,057 of 4,096
payload bytes and 59 of 128 UE8M0 bytes differ after flattening. No conclusion
about its cause follows from the input diagnosis.

## Immutable evidence

| Object | SHA-256 / identity |
|---|---|
| Runtime image | `sha256:77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64` |
| Frozen sidecar | `c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433` |
| Frozen seed | `20260905128` |
| Diagnostic JSON | `9cbdb80998a54601e1320f7944243be1e124670f9f3e8ef6df380f5602299d27` |
| Diagnostic NPZ | `6c60560e4ca9eee1c86ab0be8d680d9b86abc91dd77fa8b895d21994db35a25a` |
| Diagnostic result | `3f0a9d2398eb1f0d763f05322247766d56fc04463f68fc4ec636434e3c0d537e` |
| Closure protocol | `9556885c32385574530acf7b07c1d5636b46852ca9714c27883f98f17368efe1` |
| Diagnostic runner at audit | `a956f4a4000a6f6d52af733675b5da18b7b4b98f067ca805143cb7ee14cc393e` |
| CPU transform reference | `5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2` |
| Device dynamic kernel source | `aa73d9cd9a813daf22969a69b77fb1603a4eb105e0abc30422bd075e7dc06266` |

Evidence root:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/m1-device-carrier-diagnostic-v2`.

## Layout audit

The device and CPU implement the same mathematical ordering:

```text
BF16 -> FP16 -> normalized H512 -> signed suh -> normalized H128
     -> per-K32 amax -> UE8M0 -> E4M3 -> K32 MMA permutation
```

The input owner applies H128 independently to four contiguous 128-element
quarters, then couples corresponding elements with H4. This is the same
natural Sylvester H512 ordering produced by the CPU's iterative strides
`1,2,4,...,256`. The following H128 is also natural order.

The runner's `K32_PERM` is exactly the in-image
`_W4A8_TRELLIS_A_K32_PERM`:

```text
0,1,8,9,4,5,12,13,2,3,10,11,6,7,14,15,
20,21,28,29,16,17,24,25,22,23,30,31,18,19,26,27
```

The kernel quantizes the permuted 32-value register tensor and stores its eight
packed `uint32` words as four little-endian `uint64` words. Flattening the CPU
payload after `index_select(K32_PERM)` produces that same byte order. A wrong
permutation would produce a blockwise, repeated mismatch, not three isolated
adjacent-code differences.

## Numerical-order audit

The implementations are algebraically equivalent but not FP32-operation
equivalent:

- CPU `hadamard_blocks(H512)` performs nine butterfly stages, then one division
  by `sqrt(512)`.
- Device H512 performs normalized H128 in each quarter (multiplication by the
  literal `0.088388347648`), a left-associated H4 expression, then
  multiplication by `0.5`.
- CPU `hadamard_blocks(H128)` divides once by `sqrt(128)`.
- Device `_w4a8_had128_quad` multiplies by the FP32 literal
  `0.088388347648`.

A CPU emulation on the frozen input and sidecar reproduced exact UE8M0 bytes
and showed that merely changing these legal associations moves only isolated
E4M3 rounding decisions. At packed index `2000`, the current CPU normalized
value lies just above the midpoint (`31.0000019` in scaled E4M3 units), while
the device-order emulation lies just below it (`30.9999905`), explaining the
observed `32.0` versus `30.0` codes. The other two observed differences are
likewise adjacent-code cases; actual device pre-quant FP32 values were not
captured, so their precise last-bit path is not claimed.

## Proposed source change before implementation

Preserve the byte-exact acceptance gate. Make device arithmetic follow the
already frozen CPU reference rather than accepting a tolerance:

1. Add an unnormalized H128 butterfly helper for the input owner's H512 stage.
2. Form the outer H4 with explicit pairwise temporaries matching CPU strides
   128 and 256:

   ```text
   r0=x0+x1; r1=x0-x1; r2=x2+x3; r3=x2-x3
   y0=r0+r2; y1=r1+r3; y2=r0-r2; y3=r1-r3
   ```

3. Apply one explicit FP32 normalization by `sqrt(512)` after those nine
   butterfly stages.
4. Multiply the signed `suh` in FP32.
5. Apply the inner H128 butterflies and one explicit FP32 normalization by
   `sqrt(128)`.
6. Leave the K32 amax, UE8M0 selection, E4M3 conversion, permutation, and
   packed-store order unchanged.
7. Rerun the same immutable input carrier gate. Any remaining byte difference
   is a failure; no thresholds or fixture rerolls are permitted.

Use explicit pairwise temporaries so compiler reassociation cannot silently
restore the old rounding order. The exact division or multiplication operation
must be chosen to reproduce the Torch reference's terminal division, then
confirmed by device closure. Its latency and register impact are unmeasured;
the change is confined to the once-per-token input prologue and adds no GEMM.

## Required harness correction

Reshape reference payloads to the device carrier shape before exact equality:

```python
expected_input = reference["input_payload"].reshape_as(input_payload)
expected_middle = reference["middle_payload"].reshape_as(middle_payload)
```

This does not weaken the acceptance rule: it compares the same contiguous
bytes under the same logical ordering and still requires zero mismatches.

## Claim boundary

The audit establishes that the input scale and layout close and that three
input payload bytes sit on operation-order-sensitive E4M3 decisions. It does
not qualify the input kernel, prove the proposed arithmetic rewrite, explain
the middle-carrier failure, qualify numerical output, or support KLD or speed
claims.
