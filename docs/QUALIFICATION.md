# TrellisMX qualification gates

## CPU gates (ordinary CI)

Run with CUDA disabled:

```bash
CUDA_VISIBLE_DEVICES= pytest -q tests/test_p4_codec.py tests/test_trellismx_package.py tests/test_trellismx_cli.py tests/test_trellismx_protocol.py tests/test_trellismx_workflow.py tests/test_trellismx_p8_fixture.py
```

These gates cover golden P4 packing, every sliding state window, scalar
independent decoding, signed-zero and nearest-even behavior, scale fitting and
swizzling, invalid/truncated inputs, K3/K4/K5 P8 stream round trips, P8 CPU
reference decoding, deterministic fixture creation/loading, architecture
planning, and CLI installation. They do not execute CUDA or the external
Viterbi backend.

## GPU gates — NOT RUN by this package

Every item below remains **NOT RUN — CPU-only task** unless a separately
authorized campaign records an execution receipt:

1. Device decoder bit-exact closure for K3, K4, and K5, including boundaries
   and random access.
2. Reference/pseudoquant numerical closure for each supported shape.
3. Two or more stored rates in one process, with compile specialization keyed
   on rate.
4. M1 decode and grouped prefill independently; no prefix-cache contamination.
5. CUDA-graph parity and repeated determinism.
6. Matched full-model teacher KLD against a stock no-sidecar carrier on the
   same forced-decode `nvfp4_ds_mla` path.
7. Actual resident-memory and prologue-cost accounting.
8. Performance only after correctness, with C1 decode reported separately from
   aggregate throughput.

The existing `scripts/closure-repro/*.sh` files are historical reproduction
aids for authorized device environments. They are not invoked by CI and should
not be started as part of ordinary package validation.

## Preserved recent device diagnostic

The source snapshot includes a grouped M64/N128 real-sidecar closure failure
with four differing post-FC1 payload bytes. The localizer reports each as one
E4M3 code point apart with no UE8M0 scale-byte movement, which is consistent
with accumulate-order rounding rather than a trellis decode defect. This is a
diagnostic classification, not a passed closure gate.
