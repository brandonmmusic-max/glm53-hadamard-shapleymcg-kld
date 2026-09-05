# P8 narrower FC1 device campaign

Status: **N64/N32 bit-exact on four GPUs; both fail the frozen 35% speed gate.**
No integrated serving or full-model KLD result is claimed for these candidates.

## V2 measured result

| Physical GPU | N128 control ms | N64 ms | N64 reduction | N32 ms | N32 reduction |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.144192 | 0.094976 | 34.13% | 0.099072 | 31.29% |
| 1 | 0.117472 | 0.076832 | 34.60% | 0.080576 | 31.41% |
| 2 | 0.145856 | 0.094880 | 34.95% | 0.098944 | 32.16% |
| 3 | 0.117760 | 0.076832 | 34.76% | 0.080960 | 31.25% |

Medians use 500 CUDA-graph timing samples per arm/device, in five cyclic-order
rounds of 100. These are correlated subsamples from one prepared image and
synthetic layer-3/rank-0 payload, not independent server starts. Both candidates
miss the predeclared requirement of at least 35% lower total routed-block time
on every GPU. The near miss is not rounded into a pass.

All 30 correctness cells per GPU passed: eight single-token inputs plus M2/M3
fallback, three tile requests, five eager executions and five graph replays.
Single-token input, input-scale, intermediate, route-output and final-output
bytes matched the control exactly. Fallback input rows/scales matched after
logical route canonicalization; route and final outputs were exact. All tested
outputs were finite. Cases include high-amplitude clipping stress, zero/tiny
inputs, sparse K boundaries, route permutation and a zero route weight.
This is not exhaustive rounding-boundary testing or model-level KLD closure.

## Preserved V1 failure and V2 amendment

V1 stopped on GPU0 before timing. All eight M1 cases matched, but the unchanged
N128 M3 control itself varied in physical input/scale hashes. Its final and
per-route outputs remained identical. The independent source audit identified
generic routing's atomic per-expert physical-row assignment: token_map retains
the logical route while scratch rows permute. M2 merely happened to retain one
physical order in that run.

V2 did not loosen output tolerances or change the kernel image. It added passive
row-map metadata and canonicalized fallback input/scale rows on CPU before
comparison. Row counts, expert tile prefixes, capacity and exact logical-route
permutation are checked first. Raw physical hashes remain in the receipts.
The resolved fallback dispatch must be the unchanged N128 monolithic path.
V1 remains a failed gate; V2 is an explicitly amended developmental test.

## V3 cropped-copy result

The cropped-copy image also passed all 30 exact-closure cells on all four GPUs.
It did not clear the unchanged speed gate:

| GPU | N128 ms | Cropped N64 ms | Reduction | Cropped N32 ms | Reduction |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.144192 | 0.095008 | 34.11% | 0.096960 | 32.76% |
| 1 | 0.117584 | 0.076896 | 34.60% | 0.078368 | 33.35% |
| 2 | 0.144384 | 0.095200 | 34.06% | 0.096896 | 32.89% |
| 3 | 0.117888 | 0.076960 | 34.72% | 0.079008 | 32.98% |

Removing the duplicate copies did not materially help N64 in this screen;
N32 improved but remained slower than N64. This does not isolate all memory
or decoder costs. V3 image is
`sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141`;
plan and raw evidence are versioned alongside V1/V2.

The next bounded candidate combines the existing zero-initialized scratch
buffers into one aligned arena. It must preserve every initialized byte and
all typed view shapes. The integrated trace's approximately 0.55 ms/token in
fills motivates testing fewer clear nodes; it does not predict the gain.

## Implementation and next test

The B12X-derived standalone FC1 gives N64 64 useful tasks and N32 128 useful
tasks, versus the current dynamic kernel's 32. The candidates launch grid128;
the control uses grid64. This also introduces a separate preparation/FC1
boundary, so the result is not isolated causality for tile width alone.

Each output keeps its complete ordered K4096 accumulation. The port preserves
FP32 clipped SwiGLU, the subsequent BF16 rounding seam, and K32 requantization.
Disjoint byte stores protect neighboring UE8M0 scale bytes. The tested image
inherits FC2 unchanged from the measured control, verified by SHA-256.

The initial candidates still copy an N128 weight carrier for each narrower
owner. The next hypothesis is to copy only the owned B/SFB ranges, retaining
absolute shared-memory addresses, grid geometry, decoder and arithmetic.
Its measured outcome is recorded in the V3 section above.

## Receipts and scope

- Candidate image: `sha256:edb6f347f9b2406479d91b2505684d8c430e6e3dd481b94551edfd68673cb15a`.
- Parent: `sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed`.
- V1 source: `f206d408375a14519cb82fca8d72d6830a4ea2b0`.
- V2 source and plans: see execution receipt and `experiments/p8-fc1-tiles-device-v2.json` with SHA-256 seal.
- Evidence: `evidence/opened/codec-v2/p8-fc1tiles-device-v1/` and `p8-fc1tiles-device-v2/`.
- Reanalysis: `python3 -m glm53_nvfp4.analyze_p8_fc1_tiles --root <v2-evidence> --plan experiments/p8-fc1-tiles-device-v2.json --output <fresh-json>`.

P8 remains K4 procedural MCG to E4M3 with UE8M0/32, 4.25 routed bpw, identity
boundary and no LDLQ. **Its native mxf8f6f4 path has twice NVFP4's MMA issue
count; it is not P4's native NVFP4 speed class.** ExLlamaV3 constants/layout,
KQuant/QSRT lineage and B12X pipeline ports retain the attribution and terms in
`THIRD_PARTY_NOTICES.md`. No protected role was opened, no allocation resumed,
and the prior failed EXL3 serving gate remains unchanged.
