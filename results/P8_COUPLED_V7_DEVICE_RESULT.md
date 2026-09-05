# P8 coupled v7: M1 device closure passed

2026-09-05. GPU-smoke numerical closure only; no new model KLD or speed result.

Conditions: synthetic layer 3, TP4-local rank 0, M1, eight selected experts,
H512/H128 coupled scales, fixed draw0, Flash capped SiLU. Attention backend and
KV dtype: N/A (standalone MoE). MoE: native coupled P8; activation E4M3 with
UE8M0/32; K4 payload 4.25 bpw plus channel/sign metadata (~4.25398 effective).
P8 mxf8f6f4 has twice NVFP4's MMA issue count; this is not P4 speed qualification.

Immutable image: `sha256:6865c200f6516c18a3f78f50afd21c96e106281da2ef63ddd87feb5675c08774`.
Build source: `c5f7a0ea0218b15b2a63ed76a438d0f5c8d027b5`.
Execution followed `experiments/p8-coupled-v7-execution.json`.

Raw diagnostic passed capture integrity and all 16 per-expert projection checks.
Input payload and scales had zero mismatches. Gate aggregate relative L2 was
0.0000106782; up was 0.00000127636. Seven FP16 bytes differed across gate/up,
so raw MMA output is numerically close, not bit-exact against the reference.

The subsequent ordinary M1 run, with diagnostic mode off, passed the original
frozen protocol. All input and intermediate FP8 payload/scale bytes matched on
all five eager repeats. All five output hashes were identical. Final relative
L2 was 0.0000932578; routed-output relative L2 was 0.0000107316.

Local immutable receipts under
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/`:

- `m1-raw-localization-v7/result.json`: SHA256 `a0ec3add33f4eae1ae362673b75ff59790e0891ae8ab3d9690a91482fc4c4fb2`.
- `m1-device-closure-v7/result.json`: SHA256 `f5e564bdcc0c4d11e7a63eee5d074dcf774a8bfd3fa365d659323b790a039c27`.

This repairs the observed input carrier mismatch and validates the now-live
coupled M1 path on this fixture. Earlier failures remain valid historical
evidence. Prefill closure, CUDA graphs, integrated serving, three-layer KLD,
all-layer KLD and performance remain unproven. Production remains off.
