# V4 inherited procedural decoder audit

2026-09-05. CPU-only immutable-image inspection; no GPU or model run.

Question: does the inherited v4 MCG helper differ from the reference source in
a way that explains the broad K4 middle-carrier failure?

Image: `sha256:a95543586b0e0ed0666e096102f40f1ad7b674a1a882ef7129d0a3a2c0625e7d`.
Both installed `b12x/moe/_shared/kernels/w4a8_mcg_decode.py` copies have SHA-256
`8b1e90d3189231ff77b4e3a44aa8d51d4383d565d5e43a7e73faa40cec65b17f`.
Local `runtime_patch/p8_mcg/w4a8_mcg_decode.py` has SHA-256
`baecc8e9f7ef8ea713b04fb5d3a6c80cc661552083756377b0ad380ebc570a9c`.

An exact `diff -u` in the immutable image, with the local file mounted read-only,
finds only the allowed-bit tuple `(3, 4)` versus local `(3, 4, 5)` and its error
message. The K4 bit-window arithmetic, MCG constants, alpha-2 compander,
E4M3 conversion, register arrangement and dispatch code are identical.

Thus a changed helper implementation is not an explanation for the current K4
failure. This does not establish correctness of either implementation or its
caller. It also means K5 must not be claimed supported by this inherited helper;
the separate K5 runtime arm still needs its own pinned integration and closure.

Conditions: synthetic native P8 MoE component, K4 4.25-bpw payload plus coupled
metadata, E4M3 activations/weights and UE8M0/32 scales; attention/KV not tested.
P8 uses `mxf8f6f4`, twice NVFP4's MMA issue count. No KLD/speed claim.
