# P8 index-order device v2: UUID receipt amendment

Status: preregistered amendment, pending new five-process execution.

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

The successful first process is narrow real-device evidence for this kernel
matrix, not five-run determinism, integrated serving closure, KLD or codec
quality. P8 retains E4M3 mxf8f6f4 with twice NVFP4's MMA issue count. No teacher
or protected role is opened and no allocation stage advances.
