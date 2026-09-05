# P8 index-order device v2: UUID receipt amendment

Status: **five new independent GPU0 processes passed the synthetic kernel gate** on 2026-09-05. This is not full-model numerical closure, KLD, or a speed result.

V1's first GPU process returned `passed` after all380 ordinary/negative cases
and80 state-transition checks. Its result SHA256 is
`fe6f56c73dc1d90fff28cc41535b8b4dfd3c4d7c390f017e9804ac8489104574`.
The host then failed its GPU identity check, because PyTorch returned
`4d808f8d-88c8-8d7d-cb31-d077a6fd28ac` and NVIDIA's inventory returned
`GPU-4d808f8d-88c8-8d7d-cb31-d077a6fd28ac`. These encode the same UUID.
V1's root remains failed, SHA256
`23b973a5225823fd228941ae72200e52c67b1148ebc0885058879e17bf7339d3`.
No second process ran; services were restored and the owned container removed.

V2 accepts only a syntactically canonical hexadecimal UUID, optionally prefixed
with NVIDIA's literal `GPU-`. It compares parsed UUID values, not substring
matches. Other GPU UUIDs, malformed identifiers, different compute capability,
missing cases, source drift and every numerical failure still fail. The
original probe and candidate kernel bytes are unchanged.

The old plan/source/result/failure/restoration receipts are authenticated
before the new campaign. This is a fresh five-process namespace/output, not
resumption or replacement of v1. The prior process is not pooled into the new
five-process gate. The [v1 matrix and criteria](P8_INDEX_ORDER_DEVICE_V1.md)
remain unchanged: GPU0, fixed inputs, eager/graph cases, dynamic boundaries,
negative-page cases, exact semantic pairs, reset counters, short-order byte
signatures, no retries,75C start/90C abort and fail-closed cleanup/restoration.
99CPUtests pass, including replay of the stopped receipt with strict UUID
normalization. The exact-image CPU source-introspection gate remains applicable
because its five covered source files are unchanged.

## Terminal results

All five new processes passed 380 case calls and 80 state-transition calls each: **2,300 total checks**. Exact candidate short-order bytes produced the same determinism SHA256 in every process: `d7592d778e06f037de18ae43ad5db4be33ef2f04239ab5c8727db2d564fe6772`. The stopped v1 process is not included in these five repetitions.

| New process | Case calls | Transition calls | Maximum sampled temperature |
| --- | ---: | ---: | ---: |
| 1 | 380 | 80 | 34°C |
| 2 | 380 | 80 | 34°C |
| 3 | 380 | 80 | 35°C |
| 4 | 380 | 80 | 35°C |
| 5 | 380 | 80 | 35°C |

The snapshot replay verified all nine sealed sources, every result hash against the ordered root receipt, exact case inventories and paired-score/counter receipts, five distinct sequential container IDs, unchanged launch arguments, and terminal restoration. Each container exited 0; the unit is inactive/success/MainPID 0, with no owned container remaining. The previously active backend was restored and the previously inactive timer remained inactive. Execution ran 09:20:29–09:20:57 UTC.

These are fresh processes using an isolated shared compiled cache, not five cold-JIT trials. Long legacy rows require the same selected ID/score pairs, not deterministic legacy output order. Only negative-page sentinels were exercised; positive out-of-range addresses remain the original scorer's caller-validity precondition. The synthetic matrix does not establish integrated serving or all-rank/model closure, and no teacher, protected role, KLD or throughput measurement was used. P8 retains E4M3 mxf8f6f4 with twice native NVFP4's MMA issue count; this test does not measure that cost. No allocation stage advances.

Evidence: [snapshot manifest](../evidence/opened/codec-v2/p8-index-order-device-v2-pass/snapshot.json), [root execution](../evidence/opened/codec-v2/p8-index-order-device-v2-pass/execution.json), [replayed report](../evidence/opened/codec-v2/p8-index-order-device-v2-pass/report.json). Full per-process raw results, launch/state/thermal receipts are preserved in the snapshot; logs are hash-referenced only.

Plan SHA256: `fa2b503b9791fd9135240bee7a0b811084e98c2fe95f53bb12d2c53cc205db91`. Root execution SHA256: `251757fd24931fe2ca260fedd63ed1a3819aa5170194b0298dc95036c33390f5`. Frozen image: `sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`; transformed indexer source: `655236be67b31d61a159fcce01ad85fe74b9c2445aedcbac3b474fb664e6de86`.
