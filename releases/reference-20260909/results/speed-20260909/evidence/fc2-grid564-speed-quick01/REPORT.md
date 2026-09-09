# FC2 grid564 serving viability

Completed successfully; not adopted and no fuller run. Original production and candidate verified stopped after the unit exited0. Production image/config/checkpoint remain unchanged.

| Measurement | Result |
|---|---:|
|32K cold prefill|7912t/s|
|32K TTFT|4.142s|
|8K C1 speculative decode|180.83740t/s|
|8K C4 aggregate speculative decode|305.11200t/s|
|C4 aggregate divided by4|76.278t/s|

One server, one pass;5cold prefill requests,20-second decode windows, output cap8192. No request errors, underfill, warmup timeout or capacity limitation. MTP acceptance fields C10.593607, C40.419355. Duration-window requests are not required to finish naturally.

TP4/DCP4,MTP3probabilistic,NVFP4KV,maxseq24,batch4096,FULL_AND_PIECEWISE verified on4ranks;4x300W unchanged. Startup KV allocation23,411,764tokens. This is startup allocation, distinct from benchmark capacity inference. Image `sha256:7d8f7e2593f49abb0def03cd747fe24b91ae7f913021b51cea3d822e3b94f8f9`. Script SHA `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`.

Changes relative to original production: direct FC2 launch564CTAs and preferred shared carveout100; separate FC2/dynamic module imports and compile-key discriminator. All other kernel math and grouped prefill remain original.172representative exact component comparisons passed; paired local-MoE timing showed roughly1-3%lowerM4 and4-6%lowerM16time, which does not establish served gains here. Source hashes inside running image matched tested files. Original prefill was used; this is not the earlier row-alias376 prefill image. Comparing their prefill numbers cannot isolate FC2 effects.

Server log contains first-use MHCPrefillTf32ProjectTmaKernel JIT warning during inference. All5prefill samples retained with no posthoc exclusion. The aggregate JSON does not isolate compilation cost per request, so no adjusted throughput is claimed. Thermal, cache and MTP variation remain rival explanations for small differences. Historical references only; no production baseline throughput rerun or independent replication. No confidence interval or causal gain claim.

Full-model quality not qualified. Raw smoke preserved in results; component exactness is not full-model KLD. No new KLD, target-only baseline, full suite or publication. Targets300C1/20000prefill remain unmet.

Evidence: SUMMARY.json; sealed plan and hash; results-01/fc2-grid564/rep-1/{prefill,decode-cap8192}.json plus raw logs, metric snapshots, startup, source checks, launch command and telemetry. Component report at ../fc2-carveout100-grid564-component-01/REPORT.md. Further work should seek larger per-step savings rather than expanding this candidate into a full benchmark.
