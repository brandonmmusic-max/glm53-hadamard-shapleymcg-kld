# P8 deterministic short-index order: device gate v1

Status: sealed design pending GPU execution. CPU/static evidence is not device
qualification. This is an attention-indexer correction motivated by the
[two-process trace](P8_INDEX_TRACE_V2.md), not a new trellis law or encoder.

## Decision before result

The hypothesis is that a deterministic placement of all retained short-row
indices eliminates the recorded ordering variation without changing scores or
selection membership. Rivals include a wrong local-carry invariant, missing
synchronization, compiler source/cache substitution, stale cross-launch state,
or another downstream numerical defect. A passing synthetic test does not
eliminate the last rival or establish full-model KLD closure.

The original source is the pinned B12X fused indexer, SHA256
`69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af`.
The candidate source SHA256 is
`655236be67b31d61a159fcce01ad85fe74b9c2445aedcbac3b474fb664e6de86`.
Both execute from image
`sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`.
The original module is separately loaded from the unchanged file; the opt-in
candidate uses inspect-visible transformed source and a unique compile-cache
suffix. CPU preflight checks both class and module source isolation.

The intervention applies only to paged logical output, 32 heads, K512,
cross-CTA merge, and at most 512 compressed pools. Every CTA checks the same
required page entries before choosing the direct arm. Valid local carries
are written to disjoint logical ranges; CTA0 fills the unused suffix. An
explicit local barrier publishes completed carry writes. Empty CTA ranges
are clamped. Negative or missing pages take the old collective relay uniformly.
Positive out-of-range pages remain an existing caller-validity precondition:
the unchanged scorer accesses them before this late guard. They are not
deliberately executed on the GPU.

This refines the earlier proposal before any intervention GPU result: heads32,
uniform page validity, source-introspection isolation and distinct cache keys
are now explicit. The old greater-than-512 relay body remains unchanged, but
the new guard adds one local CTA barrier; zero performance overhead is not
claimed. Tail-token omission is deliberately unchanged to isolate ordering.

## Fixed synthetic matrix

- One physical GPU0, selected by sealed UUID/PCI/power inventory, five fresh
  processes, same deterministic seed20260905. Experimental unit: process;
  cases/replays are correlated subsamples, not independent trials.
- Exact analytic scores:32 heads with one-hot E4M3 queries/keys, weights1,
  per-logical-row FP32 scales `i+1`, yielding score `32*(i+1)`. Page mapping is
  shuffled; key payload and scale arrays use the native split8448-byte page.
- Four CTAs with explicit merge threshold1024 test pool lengths
  0,1,63,64,65,255,256,257,511,512,513,1023,1024,1025. The forced threshold is
  diagnostic, not a new serving policy. A full-SM grid with the default
  threshold resolver separately tests65,512,513 using synthetic capacity.
- Both original and candidate execute five eager and five captured-graph
  replays per cell. Caller-owned scratch is preinitialized once and shared
  across lengths/arms. Both512-to-513 and513-to-512 transition sequences run
  without explicit scratch resets. Safe first/second-page `-1` cases test the
  uniform fallback. Total380 ordinary/negative calls plus80 transition calls
  per process, excluding warmup/capture.
- Require exact selected ID/score-bit pairs, initialized `-1/-inf` suffix,
  reset merge counters and logical ascending candidate order on valid short
  rows. Legacy long-row ordering is not required to repeat; membership and
  score bits are. The canonical short-row output-byte signature must match
  across all five processes. Missing/duplicated cases, wrong GPU or drifted
  source identities fail the host receipt check.
- No exclusions, replacement process, automatic retry, KLD estimate, timing
  metric or confidence interval. Any assertion/process/identity failure stops
  subsequent processes, preserves the attempt and requires a new amendment.

The maintenance lock serializes local GPU use. Start at or below75C, abort at
or above90C, timeout2400seconds per process. No model, sidecar, calibration or
teacher root is mounted; network is disabled. A fresh campaign-local compile
cache is shared across processes: fresh processes do not mean cold JIT caches.
Owned containers are authenticated before cleanup. Original backend/timer
state is restored only after safe container shutdown is verified.

## Validation and continuation boundary

92 CPU tests passed; an independent reviewer checked kernel geometry, source
visibility, matrix coverage, process cleanup and service restoration. Exact
image CPU import passes with GPU access disabled. Those receipts precede the
plan seal. Runtime GPU/kernel behavior remains unproven until execution.

Reproduction entry point:
`python3 -m glm53_nvfp4.p8_index_order_device_runner run --plan <sealed absolute plan>`.
This command is not permission to rerun an opened attempt. Plan and source
hashes, exact image, hardware inventory, raw per-call results and lifecycle
receipts determine the supported claim.

Only after a pass and inspection should a separately sealed all-rank
forced-M1 intervention test compare repeated N128, then N128 versus N64, then
the32 conditional-fit windows. The separate incomplete-tail defect must be
measured/repaired explicitly before a final correctness claim. Full-model
KLD, allocation and optimized-profile gates remain pending.

Attribution: the scorer, original relay, runtime compiler and kernel API are
existing B12X code, not newly invented here. The direct-placement intervention
and test harness are additions. Existing ExLlamaV3/MCG, KQuant, QSRT and
w4a8_trellis attributions remain unchanged. P8 still uses E4M3 `mxf8f6f4`, with
twice NVFP4's MMA issue count for equal K; this test does not measure codec speed.
