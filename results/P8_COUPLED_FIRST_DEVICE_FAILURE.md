# First coupled P8 M1 device result: fail, diagnosis ongoing

2026-09-05. Synthetic layer-3/rank-0 M1 only. Attention/KV: not exercised.
MoE: coupled P8 native E4M3 weights/activations, UE8M0/32, K4 payload 4.25 bpw
plus channel-scale metadata. P8 requires twice NVFP4's MMA issue count.

After the user canceled remaining baseline repetitions and baseline speed,
the first coupled candidate GPU test ran on GPU 0 with immutable image
`77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64`
and validated synthetic sidecar
`c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433`.
It reached the actual materialized M1/N128 coupled wrapper, but failed the
frozen activation-carrier comparison. No three-layer KLD run was launched.

All attempts are preserved under
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1`:

1. `m1-device-closure-v1`: original gate failure.
2. `m1-device-carrier-diagnostic-v1`: diagnostic itself rejected differing
   array shapes before saving bytes. Preserved, not silently replaced.
3. `m1-device-carrier-diagnostic-v2`: same candidate/reference/gates, diagnostic
   records both shapes and compares flattened bytes without altering the gate.

The diagnostic exposed a harness shape bug: reference input payload is
`[1,128,32]` and intermediate payload `[8,16,32]`, while extracted GPU payloads
are `[1,4096]` and `[8,512]`. Original `torch.equal` rejects these shapes even
if all bytes agree. Thus the first boolean payload failures alone did not
prove numerical disagreement.

Flattened-byte diagnostics from attempt 3 establish actual disagreement too:

| Carrier | Compared bytes | Differing bytes |
|---|---:|---:|
| Input payload | 4,096 | 3 |
| Input scales | 128 | 0 |
| Post-FC1/down-input payload | 4,096 | 4,057 |
| Post-FC1/down-input scales | 128 | 59 |

Input differing packed indices: 800, 1988, 2000. Exploratory route-output
comparison from the saved arrays: cosine 0.03256624349363797 and relative L2
1.4303902714101893. These are diagnostic measurements, not KLD. They do not
meet the preregistered numerical thresholds; fixing only the array-shape check
cannot qualify this candidate.

Input FP32 Hadamard association/normalization rounding is a source-grounded
rival for the sparse input differences. The widespread intermediate mismatch
needs a separate reference/layout/decoder/boundary investigation; it is not
attributed to those three input bytes without evidence. No tolerance has been
relaxed. The existing baseline P8 measurements concern the identity path and
are not invalidated by this new coupled-path failure.

The production backend was started again after the GPU diagnostics; service
state was active. Endpoint readiness is a separate check. Baseline cold/speed
work remains canceled; no automatic restart of that campaign is intended.
