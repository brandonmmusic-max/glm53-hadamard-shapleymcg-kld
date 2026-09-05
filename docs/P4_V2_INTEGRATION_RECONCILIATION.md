# P4 v2 bridge and GLM-serving reconciliation

Date: 2026-09-04

This record reconciles the independently developed P4 encoder/sidecar bridge
and the native P4 GLM-serving path. The integration is structural evidence
only. It does not claim a CUDA launch, coherent generation, KLD closure,
determinism, graph parity, or speed.

## Inputs preserved

- Native-RNE encoder and TP4 bridge: mainline commits `00e99f2`, `ec9a435`,
  `40d2eb6`, and `f78cb06`; frozen plan SHA-256
  `5e851798bfac202be86cc9d1e056f5f5e7c3bcbb6e12861e417f67febca17101`.
- GLM serving adapter and optimized native source: source commit `af769110`,
  integrated as `4b5c9f1`; plan SHA-256
  `5fa3f9ad31d1694762384f38caf0a295ac109e1869ac4bc9c3a794b6221523b9`.
- Scratch/capture safety amendment: source commit `f469cec`, integrated as
  `0c7bf27`; plan SHA-256
  `285066b8b98307e3f481696c84e34b83821abaf51bd6dce90d5ac908deae6d4e`.
- Source call-path check: local GLM vLLM commit
  `df684ff47dcbf088b41311494fd20347a702e56a`.
- Methodology source: local `rtx6kpro` wiki commit
  `a14383fa1ed0af7aa2f22fd0f5fac95d9cd4c728`.

The independent input receipts remain at their original paths and hashes.
This integration record does not rewrite their historical claims.

## Reconciliation decisions

1. The v2 native law is procedural MCG alpha-1 projected to E2M1 with
   round-to-nearest-even and signed-zero preservation. The kernel was changed
   to that law; the earlier ties-to-lower, positive-zero implementation remains
   historical evidence only.
2. The runtime consumes the bridge's six-tensor TP-rank v2 container. Gate and
   up ordering, TP4 row/column sharding, per-expert global factors, physical
   E4M3/16 scale bytes, and exact file hashes are validated before CUDA
   allocation. Per-layer capture identities must cover the declared layer set
   exactly.
3. Encoder-only state lookup is 64 KiB; the runtime decoder uses no state LUT.
   No LDLQ or BlockLDLQ path was introduced.
4. The actual GLM `FusedMoEFactory` and `RoutedExperts.forward_modular` path is
   opt-in and fail-closed. Router outputs, shared-expert scheduling, and the
   enclosing TP reduction remain owned by the stock runner.
5. Activation packing is reused once per FC stage and the sparse launch bound
   is tightened. Offline assembly contains 15 native E2M1 K64 MMA sites, uses
   62 registers and 1,728 bytes shared memory, and reports zero spills. These
   are compiler facts, not throughput evidence.
6. Scratch is namespaced by model and TP rank, then isolated by shape, stream,
   and CUDA capture sequence. Concurrent/reentrant dispatch, duplicate model
   ownership, DBO, and microbatching fail closed. Captured resources remain
   pinned until worker exit; repeated recapture memory is unqualified.

## Integration validation

After composing both lines on main commit `0c7bf27`, the combined CPU/static
suite passed `185` tests:

```text
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python3 -m pytest -q \
  tests/test_p4_serving_bridge.py tests/test_p4_codec.py \
  tests/test_p4_native_kernel.py tests/test_p4_glm_serving.py
```

The bridge verifier replayed the committed fixture and receipt inputs with
`91` passing bridge tests, four rank files, nine projections, zero TP4-sum
error, all 65,536 decoder states agreeing, and no CUDA-visible device. The
temporary replay receipt was inspected and removed; the immutable source
receipt remains
`evidence/opened/codec-v2/p4-bridge-v2/receipt-layer-capture-pins.json`
(SHA-256
`3b17c5c979f00d2b5021db3d0da326e1fbe3aae4012f722fff21c134c4d9f7e5`).

No protected evaluation role, calibration GPU job, or P8 build GPU was opened
or interrupted during this reconciliation.

## Remaining product gates

After the all-layer P8 build and full-model KLD critical path releases the
GPUs: create real P4 sidecars from the pinned fit capture; prove exact decode
and scale-lane arithmetic on device; produce coherent GLM output; close kernel
KLD against pseudoquant; prove five-run determinism and CUDA-graph parity; and
measure cold-start-matched TP4 prefill and decode throughput. Until those
gates pass, P4 is an integrated structural implementation, not a qualified
NVFP4-speed codec.
