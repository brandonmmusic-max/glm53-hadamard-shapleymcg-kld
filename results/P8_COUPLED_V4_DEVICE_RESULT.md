# Coupled P8 v4 device result

2026-09-05. Outcome: FAIL under the unchanged frozen M1 protocol.

Image `sha256:a95543586b0e0ed0666e096102f40f1ad7b674a1a882ef7129d0a3a2c0625e7d`
built from `07d54a23e9849f0922dacec71c978a2d372360f0`. Installed manifest
`6d1594ec54f75885f8d08d909fc99d04161c91ee7c7000cb2a37d0c34fce8ac1`.
CPU installed-file/import/ABI/resource verification passed; that did not imply
GPU numerical correctness.

The same frozen rank-0 synthetic sidecar and input seed were run on GPU 0 under
the model-stack lock. Exact carrier result:

| Carrier | Mismatched bytes / total |
|---|---:|
| Input E4M3 | 2 / 4096 |
| Input UE8M0 | 0 / 128 |
| Middle E4M3 | 4058 / 4096 |
| Middle UE8M0 | 59 / 128 |

Input mismatch locations are 1293 and 2000. Shape correction now makes both
payload comparisons shape-compatible. The two confirmed indexing defects were
repaired, but the broad middle mismatch persists; they were not a sufficient
diagnosis. No successful five-repeat, prefill, full-model KLD or speed result
is claimed. Production backend and timer were verified inactive after the run.

## DECISIONS

1. Preserve failure; do not proceed to model encoding or KLD yet.
2. Add an opt-in bounded raw FC1 gate/up capture immediately before coupled
   Hadamard processing. Compare against the reference to separate procedural
   weight decoding/MMA/scales from middle transforms. New instrumentation must
   be pinned before the next run, with no acceptance-threshold change.
3. Audit the remaining two input rounding mismatches separately. Do not assume
   the earlier arithmetic-order change guarantees exact device equivalence.

Exact launch and result receipts are in `evidence/preparation/p8-coupled-m1-v4/`;
build receipt is in `evidence/preparation/p8-coupled-image-v4-build/`.
Raw NPZ remains local with SHA-256
`2bc83c8457e5f726a0cc743bad7c301b6a33e24f0da1c9d8e5843216a8b968e8`.
Post-build Docker/containerd cutoff charge was 88,031,096 bytes and the v4 build
directory 820,483 bytes, within the predeclared small-retry storage reserve.

Regime: synthetic MoE component, no attention/KV, native P8 E4M3 activations,
E4M3 weights with UE8M0/32 scales; K4 payload 4.25 bpw plus coupled metadata.
`mxf8f6f4` uses twice the NVFP4 MMA issue count. This is not a model-quality test.
