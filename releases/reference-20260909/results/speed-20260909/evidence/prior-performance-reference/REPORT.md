# TrellisMX review-fix performance investigation — 2026-09-08

## Finding
The large historical C1 gap is not reproduced as a consistent review-fix regression under matched output caps. Historical208.59 tokens/s used cap512;169.61 used cap8192. Both began with the same nominal zero-context prompt (78 actual tokens), but longer completions accumulate thousands of context tokens and can change MTP acceptance and request turnover. Temperature/thermal conditions also differed historically; the entire historical gap cannot be assigned to one cause.

Matched C1 means: cap512 old190.43/new193.09 (+1.39%); cap8192 old178.58/new174.25 (-2.43%). New cap8192 observations ranged166.94–181.55. These two observations per image/cap do not establish equivalence or rule out a small regression. Prefill8K/32K was essentially unchanged. See COMPARISON.md and every raw preparation directory.

The intended ABBA comparison required independent server starts. Four intended starts became five actual preparations after run3-B stopped at a flagged C4 sample; run3b-B completed missing cells before final A. It is not a completed balanced four-start ABBA experiment. Two C4 short-cap rows are marked underfilled in COMPARISON.md and remain visible. No small-delta significance, tail, or sealed-qualification claim is made.

## Exact regime and scope
Original A image sha256:57a9967b75e649567848c161bd807c9b951bf9955aada924eff5eab41001445c; patched B image sha256:748f8cc08a0307f5dad7c65daac7410df2bf516e3eed1284c7ecb17907ffabca. Four RTX PRO6000 Blackwell Max-Q GPUs,300W each, TP4/DCP4/MTP3, NVFP4 KV, maxseq16, batch4096, maxlen1M. Model glm53-flash-trellismx-p8-k45. Immutable launches, benchmark source hash, telemetry, metrics and checkpoint identity records are retained.

This is speculative serving evidence. Prior target-only MTP-off C4/8K failure reproduced on both images; no target-only speed qualification is inferred here. No new KLD run, model encoding, inference-code edits, tuning, image publication or PR updates occurred in this investigation.

## Kernel launch question
Deleting the dead a0..a3 shared stores and CTA barrier removes work within the reducer. H128 consumes the registers directly, and the post-transform stores remain. The reviewed change adds no hot-path launch call sites; the MCG shared decoder is inline. This code-path finding is not an old-versus-new Nsight launch-count measurement: fresh traces are B only.

CUDA graph CPU submissions and GPU nodes are different counts. C1 traces include1308/1296 graph submissions across four ranks in roughly1.52seconds; the C4 trace includes564. Profiling overhead and parallel ranks make aggregate API duration unsuitable as exclusive latency. NVTX confirms C1 targetM4 and C4 targetM16. The 32K capture includes calibration/setup requests; phases.json separately maps real promptM4096 chunks through runtime correlation IDs and process identity. Unmapped kernels remain reported.

## Nsight Systems
All four reports contain all four GPUs. Captures1/2: C1 cap512/cap8192; capture3: C4 cap8192; capture4: cold prefill32K including setup. cuda-sw,node-level graph tracing and NVTX were enabled. Safetensors replaced InstantTensor only for profiled loading; production configuration remains unchanged. Profiled KV capacity was29,532,160 logical tokens (3605 local blocks×2048×CP4); this is profiler-only capacity, not a replacement production claim.

Representative GPU0 shares of summed kernel durations:

|Category|C1 cap512|C1 cap8192|C4 cap8192|
|---|---:|---:|---:|
|TrellisMX FC1|18.96%|18.92%|29.19%|
|TrellisMX FC2|6.54%|6.63%|14.10%|
|Coupled reducer|0.40%|0.39%|0.20%|
|FillFunctor zero/fill|1.10%|1.08%|0.79%|
|Dense NVFP4 GEMM|15.50%|15.53%|8.66%|
|L2 prefetch|16.57%|16.39%|7.44%|
|NCCL + other collectives|11.63%|12.10%|23.20%|

C4 FC1 spans29–38% across ranks. Rank differences can include communication waits and imbalance. On GPU0, actual promptM4096 chunks show FC1 18.54%, FC2 9.90%, NCCL allreduce32.79%, allgather3.19%, MLA prefill9.09%, reducer2.42%. These are summed, overlapping traced durations, not critical-path shares or expected gains.

L2 prefetch intentionally runs on a side stream to make later projection weights available in cache. Its large duration is not removable overhead by inspection. Test its effects on the critical path before changing it. Likewise, a long collective kernel can include waiting for another rank; do not interpret it as pure link-transfer time.

Analyzer refinement explicitly recognizes FillFunctor instead of matching the substring 'fill', which would incorrectly classify attention/MHC 'prefill' kernels. It also identifies custom Oneshot collectives, prefetch, dense GEMMs and attention. Original analyzer source is preserved as analyze_nsys_v1.py; final summaries use the refined classifier.

## Nsight Compute
Four real-weight, synthetic-input/routing FC1 probes completed: layer3 K5 and layer4 K4, rank0 local TP4 computation, M4/M4096. Warmup twice; one FC1 launch profiled with17 replay passes. All outputs finite; finite output is not numerical equivalence validation. No distributed NCCL or graph replay. These are component diagnostics, not actual request routing or served throughput.

|Shape|K|Duration|Registers/thread|Dynamic shared bytes|Grid CTAs|Theoretical/achieved occupancy|DRAM/SM throughput|
|---|---:|---:|---:|---:|---:|---:|---:|
|M4|5|78.53us|183|27648|384|16.67/8.38%|55.17/38.06%|
|M4|4|68.29us|173|23552|384|16.67/8.40%|50.61/40.37%|
|M4096|5|2.36ms|236|65536|128|8.33/8.33%|18.67/24.19%|
|M4096|4|2.17ms|236|65536|128|8.33/8.33%|16.38/25.26%|

Device has188SM. Direct FC1 is limited to2resident blocks by registers and shared memory. Grouped FC1 is limited to1block by shared memory; its128-block persistent grid is smaller than the SM count. More SMEM would not automatically improve this. Zero spilling requests were reported. Optional PC-sampling metrics were unavailable, so these probes do not establish bank conflicts or the dominant instruction stall. NCU automatic estimated speedups are heuristics, not predictions.

## Candidate experiments, not implemented
1. Grouped FC1: compare capacity128 against188/192 on real routing, preserving persistent task ownership/queue/barrier rules. Verify per-layer output equivalence, graph replay and full-server prefill. The current source policy returns128 for M>16. Extra CTAs can increase contention; no linear gain assumed.
2. FC1 math/dataflow: examine live ranges in reconstruction/Hadamard/quantization and reduce register pressure. Any alternative pipeline/tile must preserve coupled scale/rounding boundaries. Measure resources and duration before spending additional SMEM on stages.
3. Decode: inspect direct persistent-grid work distribution, tail behavior and active expert routes; do not blindly apply NCU wave-size heuristics. Follow with matched output-length controls and full-window MTP accepted/drafted deltas.
4. Analyze communication and prefetch dependencies across all ranks. Consider overlap only with a demonstrated dependency-safe window and an unprofiled controlled comparison.
5. Compact grouped scratch storage/fill fusion is a smaller candidate, given observed fill shares. FC2/reducer fusion requires deterministic router-order FP32 summation and normalized H512 across different CTA ownership; naive atomics are not equivalent. Reducer-only fusion has little decode time to remove in these traces.

## Preserved failures and recovery
Initial Nsight+InstantTensor launch failed allocating MTP staging memory. Retry with safetensors captured four reports. Host Nsight2025.6 could not export reports from container2026.3.1; export succeeded with the matching tool from B in a CPU-only container. First NCU probe failed because audit/profile.py shadowed Python's standard-library profile module; corrected only probe import search path and reran all four probes successfully in ncu2. Earlier ncu logs remain in ncu/.

Canonical production restoration and local reviewer disposition are recorded below after live verification.

## Local cross-review and final recovery
Local GLM glm53-flash-trellismx-p8-k45 completed a read-only review of LOCAL_REVIEW_PACKET.md and a second cross-review response. Its initial claims of statistical indistinguishability and B-only profiles establishing no introduced pathology were challenged and withdrawn. Both reviewers agree: large regression not established by matched samples, small regression unresolved; preserve flagged observations; prioritize grouped FC1 capacity/dataflow and prefill communication dependency analysis; quantify GPU as well as CPU benefit before fusion. No remaining disagreement on those proposals. Full responses and corrections are retained in local-review-initial-findings.json, local-cross-review.txt and local-review-final.json. Agreement is on next experiments, not evidence of a speedup or permission to skip correctness qualification.

Canonical trellismx-r27-dcp4-production is healthy on the pinned B image, /v1/models reports glm53-flash-trellismx-p8-k45, and the TP4/DCP4/MTP3 identity guard passes. Arithmetic smoke passed during restoration. Original Slack, adapter and catchup services were resumed after local review; previously failed unrelated jobs were not started. See final-production-verification.json and final-client-states.txt. No inference implementation changes were made.
