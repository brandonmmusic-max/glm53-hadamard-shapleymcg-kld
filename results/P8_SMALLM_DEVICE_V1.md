# P8 M1 route/slice scheduler: first device result

Decision: **advance to captured-graph and four-rank integration.**

This is a single-GPU developmental kernel A/B on identical synthetic inputs,
routes, sidecar bytes and dense reference. It is not serving tokens/s, full-model
KLD, or a new product pass.

## Result

| Metric | Frozen monolithic baseline | P8 M1 route/slice candidate | Effect |
|---|---:|---:|---:|
| CUDA-event median, 20 warmups + 100 repeats | 1.134288 ms | 0.313248 ms | **72.3837% lower; 3.6211x** |
| Five output hashes | one identical hash | same identical hash | bit-exact pass |
| Cosine to dense reference | 0.9997080 | 0.9997080 | identical |
| Relative L2 to dense reference | 0.02418345 | 0.02418345 | identical |

The separately seeded M1 closure also matched bit for bit across five repeats.
M2 and M3 correctly declined the candidate schedule and matched the separately
frozen baseline output hashes exactly across five repeats per shape.

The test used rank 0 / layer 3 with the full E288 sidecar geometry and routed
the synthetic token to eight valid experts. The device was 31 C before the
sequence and 34 C after it. No role data or teacher logits were opened.

## What changed

The baseline deterministic M1 path reserves 289 M16 physical tiles and groups
all four intermediate slices into one useful task per selected expert. The
candidate uses eight physical route tiles, 32 route/K128 FC1 tasks and 128
route/N256 FC2 tasks. It therefore changes routing work decomposition, scratch
capacity, and launch count together. This A/B supports the combined candidate;
it does not isolate which one produced the gain.

Both candidate compute launches decode the K4 procedural-MCG stream in their
kernel prologues, reconstruct fully scaled E4M3 fragments with physical
UE8M0/32 scales, and feed `mxf8f6f4 m16n8k32` directly. Only quantized
intermediate activations and scales cross the FC1/FC2 launch boundary. No
decoded-weight buffer or BF16 weight dequantization exists. The fixed-order
top-k reduction remains separate and deterministic.

P8 remains an E4M3 quality product whose MMA issues at twice the rate of
NVFP4 for equal K. This result does not make P8 native-P4 speed class.

## Preserved bring-up failures

1. The first Dockerfile form named the local base by repository digest;
   BuildKit attempted unauthorized remote resolution before executing a layer.
   The corrected Dockerfile names a local tag only after verifying its image ID
   equals the pinned base SHA-256.
2. The first device compile stopped before launch because Python `assert`
   statements consumed CuTe staged tensor extents. The host wrapper already
   fail-closes the exact M1/E288/H4096/I512 contract, so the duplicate
   non-lowerable assertions were removed. The next build compiled and ran.

These are implementation failures, not codec measurements.

## Remaining gates

- poison inactive rows/scales, mixed invalid routes and repeated experts;
- prove stable addresses and output under captured CUDA-graph replay;
- repeat timing across all four GPUs and attribute the two new compute kernels;
- integrate the opt-in through the GLM serving wrapper and obtain a fresh
  four-rank C1 trace;
- only after closure, freeze a new multi-run product comparison against EXL3.

The original 20.69 versus 91.26 tokens/s product failure and the stopped
Shapley allocation game remain unchanged. The microbenchmark suggests a real
recovery path, but no end-to-end speed claim is made yet.

Candidate image ID:
`sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed`.
Pinned baseline image ID:
`sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8`.
No LDLQ or BlockLDLQ was used.
