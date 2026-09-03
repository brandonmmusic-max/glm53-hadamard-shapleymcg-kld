# V3 MXFP6 mixed-loader fail-only diagnostic

Date: 2026-09-03. With the down-projection group-size geometry applied, the repeated one-layer canary progressed through more checkpoint loading and then failed in `_load_w13` with a 128-versus-256 mismatch. It still failed before readiness or inference.

The runtime bridge now wraps the existing routed-expert weight loader only to print the tensor name, shard ID, expert ID, parameter shape, loaded shape, TP size, and TP rank when the original loader raises. Successful calls return the original result unchanged; failures re-raise the original exception. This is diagnostic instrumentation, not a numerical or experimental change, and the same sealed one-layer candidate remains the sole target.
