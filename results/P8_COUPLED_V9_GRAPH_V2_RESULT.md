# Coupled v9 graph/eager parity: v2 passed

Synthetic layer3/rank0 TP4-local, M1/M2/M64/M65, 5 graph replays per case.
Attention/KV: N/A (standalone MoE). MoE: native full-coupled P8; E4M3
activations; K4 E4M3/UE8M0 payload4.25 bpw plus channel metadata (~4.25398).
P8 mxf8f6f4 uses twice NVFP4's MMA issue count; no throughput claim.

Image remains `sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`.
Protocol v2 `77660f7e0d9f11b809695f4aea0966ade03ebbecb94e7398b334f598edeb06a0`.

All cases passed exact CPU-reference activation carrier checks and all six
logical tensor hashes matched fresh eager execution on every replay. Each case
had one unique final-output hash. Routing validity checks remained enabled.

Both original v1 runs remain failed under their whole-receipt rule. The saved
M64 evidence identifies only `schedule.route_to_physical_sha256` variation;
all tensor hashes and other compared fields were identical. V2 excludes only
this valid scratch-placement permutation and repeat number from equality,
retaining the raw hashes. No runtime, fixture, model or numerical tolerance
changed. See `experiments/p8-graph-v2-amendment.json` and the preserved
`evidence/preparation/p8-v9-graph-evidence/` receipts.

V2 receipts: `evidence/preparation/p8-v9-graph-v2/`.
This is component graph closure, not real-sidecar numerical closure,
integrated serving, full-model KLD or speed qualification. Production is off.
