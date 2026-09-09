# Combined K5 funnel serving screen

Completed successfully. **Promising C1 result, unconfirmed; not adopted.** Original production and candidate verified stopped after unit exit0. No production/client restart.

|Measurement|Previous combination|K5funnel combination|Observed difference|
|---|---:|---:|---:|
|32Kcoldprefill t/s|8196|8207.0|+0.13%|
|64Kcoldprefill t/s|8130|8145.0|+0.18%|
|8KC1speculative decode t/s|181.96215|192.58261|+5.84%|
|8KC4aggregate speculative decode t/s|312.22045|312.41265|+0.06%|

C4aggregate/4=78.103t/s. Prefill TTFT32K3.993s/64K8.046s,5and3cold requests. One server, one pass,20-second cells and decode cap8192. No errors, underfill, warmup timeout or capacity limitation. Current C1/C4MTPacceptance fields0.476852/0.534946; previous0.513889/0.524194. C1acceptance is lower, so a higher reported acceptance field does not explain the higher throughput. This does not by itself prove a kernel-caused gain: prompt, sampling, thermal/cache and inter-run variation remain uncontrolled rivals. No independent replication/confidence interval.

Intended sole change relative to previous combination is directK5FC1 window extraction using32-bit funnels, retaining original MCGconversion/FP8MMA. Grouped rowalias376,FC2grid564/carveout100,overlapthreshold4096 retained. Source verified in running image; composed component passed172exact eager/changed-input graph checks. Prior CPUbasis proof covers all admissible K5window bits, and isolated K5resource audit shows183->181registers, no spills, fewer staticSASSinstructions. Component evidence supports plausibility but is not complete-model attribution.

Image `sha256:e6ae66f1d9f7728100c415bf4ef7b33c53da5f8d33dd9d87d95d02699e0762fb`. TP4/DCP4,MTP3probabilistic,NVFP4KV,maxseq24,batch4096,FULL_AND_PIECEWISE all4ranks,4x300W. Startup KV23,411,764tokens; benchmark metrics-derived KV numbers use a different capacity calculation and are not interchangeable. Full-model quality/KLD remains unqualified; raw smoke preserved. Original production image/config/checkpoint unchanged. No publication, power change or P4 work.

Evidence: plan-01.json/hash; SUMMARY.json; results-01/combo-k5-funnel/rep-1/prefill.json and decode-cap8192.json; source/image/runtime receipts, telemetry, logs and raw metrics. Script SHA `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`. Reference ../combo-rowalias376-fc2grid564-speed-01/REPORT.md; component ../combo-k5-funnel-component-01/REPORT.md; isolated mechanism ../fc1-k5-funnel-component-01/REPORT.md. Targets300C1/20000prefill remain unmet. Narrow confirmation is warranted before broadening performance claims or adopting.
