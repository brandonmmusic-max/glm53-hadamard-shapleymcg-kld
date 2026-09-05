# Index-trace import repair and explicit v2 amendment

The v1 diagnostic failed before any evaluation request. All38 source hashes
still verify. GPU/NCCL initialization occurred, but request/capture/trace
directories are empty and the second repeat never started. Owned-container
cleanup and restoration succeeded. This is an instrumentation failure, not
evidence against the codec or the index-order hypothesis.

Preserved root receipt SHA:
`43f91db06306435eaacf4e0969349983bc3ee2b5bb23f26673495b5c2a4d1024`.
Stage receipt SHA:
`77b496198a9bf591ab999f656f8a8bf277f677d641cd997d4feb47e1de69d451`.
Safe snapshot:
`evidence/opened/codec-v2/p8-index-trace-v1-failure/snapshot.json`.
Private runtime settings/logs are referenced by hash, not copied wholesale.

## Demonstrated cause and bounded repair

The import observer was itself compiled with `from __future__ import
annotations`; its nested `compile(source, ...)` inherited that flag into the
original vLLM module, which does not request it. This changed type objects
into strings. Under the actual worker's `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
vLLM custom-op registration then failed to resolve `LayerNameType`.

The old loader reproduced the same error in the exact image on CPU with that
worker setting. The corrected loader imports successfully and preserves the
original function annotations and inferred schema. Both compiler calls now
use `dont_inherit=True`. Explicit future imports in a target module continue
to work. No model, codec, kernel mathematics or numerical decision changed.

Receipts under `evidence/opened/codec-v2/p8-index-trace-import-v2/` preserve
the preliminary annotation mismatch, old worker-mode failure, and final
`new-worker-mode.json` pass. Only the final worker-mode receipt authorizes v2;
it pins the loader, transformation module and import-test script hashes.
The preliminary no-worker-mode schema pass alone was insufficient and is not
used as the gate. The initial v1 CPU import missed this mode-dependent error.

146 focused CPU tests pass, including the preserved-failure snapshot tests,
explicit and non-inherited annotation behavior, invalid preflight rejection,
the existing decode guards and the raw-byte trace analysis. No GPU result is
implied by those tests.

## Decision before another attempt

V2 has a new unit/container namespace, fresh output root and separately sealed
plan, `experiments/p8-index-trace-v2.json`, SHA
`d8e814b5a25ea7d557d2efeb297914aad4c3d98d338e1ef816c68d7c1ecddd2d`.
It authenticates the failed v1 before proceeding, and requires both
the CPU copy/source preflight and exact worker-mode import/schema preflight.
The original two-N128 experiment is unchanged: same CF0056 forced2047 rows,
all11 sparse layers/four ranks, positions255–263, no teacher opening, no KLD,
no throughput claim, no protected roles and no automatic retry. Any failed
stage is preserved and prevents dependent work. The existing maintenance
lock, <=75C startup / >=90C abort and prior-service restoration remain.

The observer's143 copy launches per token can alter scheduling. Positive
same-input order divergence with same-row attention linkage supports the
hypothesis; a null remains inconclusive. Full-model32-window KLD and allocation
remain disabled until the actual numerical issue is diagnosed.

P8 remains native E4M3 `mxf8f6f4`, twice NVFP4's MMA issue count; P4 is separate.
