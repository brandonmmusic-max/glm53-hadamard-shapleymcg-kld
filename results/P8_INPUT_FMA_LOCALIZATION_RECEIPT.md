# P8 input FMA localization receipt

Status: demonstrated for the preserved v6 synthetic M1 trace. This is a CPU
analysis of GPU evidence, not device closure, KLD, speed, serving, or quality
qualification.

## Immutable inputs

- Trace archive: `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/m1-raw-localization-v6-evidence/raw-capture.npz`
  (`sha256:1ceefb924c2ea08fad9c1eadbcfabf4cc5c904d017a8e49d5bd4a8d8939f6d61`)
- Synthetic TP4 rank-0 sidecar: `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/rank0-v1/p8-synthetic-layer-003-tp4-rank-0.safetensors`
  (`sha256:c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433`)
- Seed: `20260905128`
- Captured logical K32 blocks: 40 and 62

## Result

The 64 captured raw prequant FP32 values differ from the separately-rounded
CPU reference in 35 positions. The same 35 normalized values differ; both
UE8M0 bytes are 115, so normalization is the exact power-of-two factor 4096.
The quantizer is therefore downstream of the observed discrepancy.

A CPU IEEE `fmaf` transcription that contracts the left `H512*suh` product
into the first H128 butterfly add/sub matches all 64 captured raw FP32 bit
patterns and all 64 normalized bit patterns. The match is exact, not a
tolerance comparison.

- Device raw SHA-256: `e3f99045d4e090c0ba9756fa678762672fa50ccdf723b0f95cd111ce07c88663`
- Contracted-FMA raw SHA-256: `e3f99045d4e090c0ba9756fa678762672fa50ccdf723b0f95cd111ce07c88663`
- Canonical raw SHA-256: `e1c720b971254cf001705fd677205e8a781f7a795b1173e0f4ca2c9851677dfd`

Two decision-relevant rows are:

| logical channel | device raw bits | canonical raw bits | device normalized | canonical normalized |
| ---: | ---: | ---: | ---: | ---: |
| 1287 | `0x3a97ffe9` | `0x3a98000b` | 4.749989032745361 | 4.75000524520874 |
| 2004 | `0x3bf7ffff` | `0x3bf80001` | 30.999998092651367 | 31.000001907348633 |

These values straddle the E4M3 midpoints 4.75 and 31.0 and explain the two
payload bytes observed in v4. On this trace, compiler FMA contraction is the
measured source of the prequant difference.

## Reproduction

Run `scripts/analyze_p8_input_fma_localization.py` with `--trace` and
`--sidecar` set to the immutable paths above. The JSON output includes all 64
device, canonical, and contracted-FMA FP32 values and raw bit patterns, hashes,
scale bytes, and exact mismatch counts.

## Grounded repair

Use an inline PTX `mul.rn.f32` barrier for each of the four `H512*suh`
products before the inner H128 in both the M1 and materialized/prefill owners.
This preserves the existing transform and scale order while forcing the
separately-rounded FP32 multiplication required by the CPU/Luke contract. No
acceptance threshold or quantizer behavior changes.
