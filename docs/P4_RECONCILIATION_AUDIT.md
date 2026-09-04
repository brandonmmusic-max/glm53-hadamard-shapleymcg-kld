# P4 codec/kernel reconciliation audit

Date: 2026-09-04  
Evidence level: independent CPU/static source audit  
Kernel commit audited: `ab5a851559e8f7593af77be01b71605f7ee1ab48`  
Codec commit audited: `d9c791cdcc18e28c49165d678fbc08a54a7d5c0f`  
Critic commit audited: `19c14c3927cfc4ae6b77fa60344275ce03680672`

No GPU, protected role, model download, or service mutation was used by this
audit. The two implementation commits are individually coherent structural
prototypes, but they are not interoperable as delivered.

## Blocking findings

1. The kernel uses smaller-magnitude midpoint ties and canonical positive zero.
   The codec uses native round-to-nearest-even and preserves negative zero. A
   compiled C++ decoder versus codec-oracle census found `5,477 / 65,536`
   different nibbles: `5,432` signed-zero choices and `45` numerical midpoint
   choices. The codec's committed 32-by-128 fixture differs in 339 nibbles and
   319 packed bytes, despite identical reconstructed trellis states.
2. The codec serializes one matrix with three tensors. The runtime requires a
   TP-rank sidecar containing gate, up, and down as six tensors. There is no
   exporter/bridge in the audited commits.
3. No GLM model hook imports the P4 endpoint. Static assembly of the desired
   instruction is not serving integration.
4. Codec scale bytes are restricted to positive finite `1..126`; the kernel
   validator also accepts zero. The canonical interchange policy must agree.
5. Sparse launch sizing uses `ceil(routes/16)+experts`. For `M=1`, top-k 8,
   288 experts, this launches 289 Y tasks for only eight useful routes. A safe
   static upper bound is `min(routes, ceil(routes/16)+experts)`; performance
   still requires measurement.

## Reconciliation contract

The next interoperable ABI is `glm53-p4-mcg-tp-rank.v2` with:

- `law=procedural-mcg-alpha1-rne-e2m1`
- `weight_rounding=nearest-even-satfinite`
- `signed_zero=preserve`
- `scale=e4m3-k16`, with positive finite bytes `1..126`
- K4 cyclic-per-tile state, identity boundary, no LDLQ
- W13 order `[gate, up]`, W2 down, TP4 rank and source-design binding

For an integer RNE decoder, use inclusive thresholds at E2M1 midpoints 0.75,
1.75, and 3.5 (fixed-point units 6144, 14336, and 28672), strict thresholds
at the other midpoints, and retain the sign bit when the magnitude code is
zero.

## Required gates

1. Exhaustive 65,536-state equality between Python, host CUDA-format oracle,
   compiled C++ decoder, and CUDA source law.
2. Exact packed-byte closure on deterministic gate/up/down fixtures, including
   scale-address and corruption controls.
3. Versioned one-matrix-to-TP4 sidecar export and fail-closed reload.
4. Actual B12X GLM factory/dispatch selection for FC1, SwiGLU, FC2, routing,
   and TP4, without dense floating weight materialization.
5. Device operand and arithmetic closure, graph capture/replay, changed-route
   replay, scratch poisoning, and five-run bitwise determinism.
6. Conditional-fit KLD closure, then matched TP4 prefill/decode measurement.

The original codec and kernel receipts remain tied to their originating
commits. Their historical manifests must not be rewritten to make a later
integrated tree appear identical.
