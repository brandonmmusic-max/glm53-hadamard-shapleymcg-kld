# P8 coupled campaign storage-ledger audit

Audit snapshot: 2026-09-05 16:08:36-04:00. This was a CPU/read-only audit:
no fixture, model layer, capture, image, cache, or service was created or
removed by the audit.

## Verdict

**Strict 30,000,000,000-byte gate: UNKNOWN/BLOCKED.** The original storage
receipt is an arithmetic preflight, explicitly not an observed baseline
(`results/P8_COUPLED_PILOT_STORAGE_PREFLIGHT.md:3-5,22-30`). No repository or
campaign receipt records filesystem-used bytes, Docker image/cache bytes, and
campaign-root bytes at the instant the ceiling began. A current total cannot be
subtracted from a missing baseline to prove incremental use.

The upcoming synthetic fixture is **not authorized merely by pointing its
generator at a fresh empty budget root**. The generator counts only regular
files below the caller-provided `--budget-root`
(`scripts/generate_p8_coupled_m1_synthetic_fixture.py:371-407`), while the
campaign rule includes existing growth, image/cache growth, and artifacts on
other NVMe mounts. A locally passing empty-root check would therefore reset the
ledger incorrectly.

The conservative, enumerated arithmetic below is still useful:

- fixture alone: 964,546,858 forecast bytes including its exact 963,497,568-byte
  sidecar, 714-byte design, and 1,048,576-byte receipt allowance;
- three-layer chunk-plus-TP4-sidecar forecast: 23,123,890,176 bytes;
- listed current coupled worktrees, campaign output directories, active quality
  peak, and incremental v2/v3 image layers: 3,536,279,987 bytes;
- listed current artifacts + fixture + three-layer forecast:
  **27,624,717,021 bytes**, leaving only **2,375,282,979 bytes**.

That 2.375 GB is not a safe allowance for the unmeasured shared Git-object
growth, Docker build-cache/container physical growth, safetensors headers,
encoder temporaries, and any simultaneous capture. The three-layer encode is
therefore **not storage-cleared**. The fixture by itself would fit the
enumerated ledger (4,500,826,845 bytes after generation), but strict approval
still requires reconciling the missing campaign baseline and Docker/cache
delta.

## Original ceiling and missing baseline

The controlling preflight was committed as
`23a32edaf8aa98333528fe02af3067b9868baa56` at
2026-09-05 15:30:46-04:00. It records:

| Forecast | Bytes |
|---|---:|
| one layer K4 + UE8M0/32 payload | 3,850,371,072 |
| three-layer payload | 11,551,113,216 |
| three-layer scales across TP4 | 10,813,440 |
| three-layer generated runtime signs | 36,864 |
| two payload copies + two scale copies + signs | **23,123,890,176** |
| nominal remainder below ceiling | **6,876,109,824** |

The same receipt says that headers, codec tables, preparation artifacts,
image/cache growth, receipts, captures, and existing campaign growth all count
and must be measured before launch. The compatibility audit repeats that an
actual aggregate-growth check remains mandatory
(`results/P8_ENCODER_RUNTIME_COMPATIBILITY_AUDIT.md:74-80`). The preparation
code likewise requires all headers, receipts, temporaries, and caches to remain
within the ceiling (`glm53_nvfp4/prepare_p8_coupled_scale_v1.py:220-225`).

Searches of repository `results/` and `experiments/`, and of the campaign
artifact tree, found these arithmetic rules but no timestamped initial byte
inventory. The campaign root currently occupies 376,397,220,352 allocated
bytes, mostly older work. That total proves why a fresh subdirectory cannot be
used as a baseline, but it does not identify how much of the total is new under
this 30 GB campaign.

## Current coupled artifacts

Apparent file sizes are used for the reproducible ledger because both the
existing runner and forecasts use `stat().st_size`. Allocated sizes can be lower
while an active capture is sparse. The five coupled worktrees were measured
with `du --apparent-size -sxB1`:

| worktree | apparent bytes | creation/birth evidence |
|---|---:|---|
| `bmxfp4-glm53-p8-coupled-image-v2` | 442,718,776 | 15:57:19 |
| `bmxfp4-glm53-p8-coupled-scale-kernel-v1` | 443,597,060 | 14:29:00 |
| `bmxfp4-glm53-p8-coupled-prefill-v1` | 443,735,312 | 15:27:00 |
| `bmxfp4-glm53-p8-coupled-integration-v1` | 445,711,868 | 15:31:29 |
| `bmxfp4-glm53-p8-coupled-scale-v1` | 446,590,818 | 14:28:52 |
| **worktree subtotal** | **2,222,353,834** | |

Three worktrees predate the arithmetic preflight. They cannot silently be
excluded because the preflight says existing campaign growth counts and does
not provide a prior baseline. The table also does not include incremental
objects in the shared Git object store; that delta is unknown.

Relevant campaign directories on
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6` were:

| directory | apparent bytes |
|---|---:|
| `p8-coupled-image-preparation-v1` | 742,934 |
| `p8-coupled-image-v2-build` | 803,636 |
| `p8-coupled-image-v3-build` | 817,873 |
| `tail-v2-p8-exl3-cf32-product-validation-v1b` | 1,310,510,246 |
| **directory subtotal** | **1,312,874,689** |

The quality directory began at 14:44:34, before the preflight. At the snapshot
it contained 998 files and one active
`conditional-fit-0060.logits.f32.partial` of 1,268,157,440 apparent bytes; the
remaining persistent metadata was 42,352,806 bytes. The runner retires a
completed raw before the next request
(`glm53_nvfp4/tail_v2_product_runner.py:263-287` and
`glm53_nvfp4/tail_v2_product_validation.py:278-288`), so this is a transient
per-window peak rather than 32 retained raw captures. Its own ceiling check,
however, sums only files under this one product-output root
(`glm53_nvfp4/tail_v2_product_runner.py:193-200`); it is not a global coupled
campaign ledger.

No real layer-3/20/22 coupled chunks or TP4 sidecars were found. The real
three-layer forecast remains wholly future allocation. The synthetic report
also states that the full fixture has not been generated
(`results/P8_COUPLED_M1_SYNTHETIC_FIXTURE_PLAN.md:1-4,98-104`).

## Candidate-image accounting

The immutable parent is
`sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745`.
Charging the entire parent again would be wrong. RootFS inspection proved it is
an exact prefix of both candidates:

| image | status | API size | increment over parent |
|---|---|---:|---:|
| parent `0336113e0fff` | existing parent | 13,814,514,417 | - |
| failed v2 `dcf6c5f6b3b9` | verification failed | 13,814,773,278 | 258,861 |
| v3 `77ae1c0b46c7` | CPU verification complete | 13,815,559,676 | 1,045,259 |

The first 20 added RootFS entries are shared by v2 and v3. Their exact
uncompressed union over the parent is **1,051,464 bytes**: 252,656 shared,
6,205 v2-only, and 792,603 v3-only. This is the image-layer amount included in
the enumerated ledger, not either image's full virtual size.

Docker physical-accounting remains a separate unknown. At 16:05 the daemon
reported 200.1 GB of images, 7.712 GB of containers, and 49.3 GB of build cache
(2.644 GB reclaimable), but no pre-ceiling daemon snapshot exists. The failed
v2 build also left an exited container with a reported 2.6 MB writable layer.
Those global totals cannot be assigned exactly to this campaign from the
available receipts, so they are not falsely presented as incremental bytes.

The v3 build receipt is
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-image-v3-build/receipt.json`;
it records `status=complete`, image
`sha256:77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64`,
and parent `033611...`. It is CPU/source qualification only; it created no
model layer and made no KLD or device-closure claim.

## Filesystem headroom is not campaign allowance

At 16:05 the physical free-byte snapshot was:

| mount | free bytes |
|---|---:|
| `/` (Docker and coupled worktrees) | 82,225,545,216 |
| `/media/brandonmusic/nvme1n1p3` (campaign outputs) | 150,821,601,280 |
| `/media/brandonmusic/klcstore` | 79,915,155,456 |

These mounts have enough raw free space for the fixture, but free space does
not relax the user's 30 GB aggregate-new-data ceiling.

## Required gate before any large write

1. Fix one campaign start time and byte baseline, including the relevant
   worktrees/shared Git objects, campaign directories, and Docker image,
   container, and build-cache deltas. Do not choose a newly empty directory.
2. Reconcile whether the pre-preflight parent/control images and the five
   already-created coupled worktrees are in-scope. In the absence of an older
   receipt, count them conservatively rather than assuming zero.
3. Re-snapshot after the active quality run finishes, retaining its measured
   1,268,157,440-byte simultaneous-capture peak in peak-use arithmetic.
4. Authorize the synthetic fixture only if the reconciled current total plus
   964,546,858 bytes remains below 30,000,000,000.
5. Do not launch the three-layer writer until its actual header/chunk geometry,
   temporary-file peak, and encoder/cache growth fit inside the remaining
   allowance. The current enumerated remainder of 2,375,282,979 bytes is not a
   sufficient evidence-backed bound for those unknowns.

