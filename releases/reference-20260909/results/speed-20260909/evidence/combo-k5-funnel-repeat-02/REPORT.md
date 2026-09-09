# K5 candidate repeatability screen

**The previous C1increase did not repeat.** Same immutable image/runtime, new candidate server and automatically generated benchmark prompt prefixes. No adaptive retries.

|Measurement|First run|Repeat|
|---|---:|---:|
|32Kcoldprefill t/s|8207|8133.0|
|64Kcoldprefill t/s|8145|8051.0|
|8KC1speculative decode t/s|192.58261|177.98882|
|8KC4aggregate speculative decode t/s|312.41265|311.29235|

C1/C4acceptance fields0.523148/0.556452. No errors, underfill, capacity limitation or warmup timeout. Single pass per independent server;5/3cold prefill requests,20-second decode cells,cap8192. No independent reference image rerun; differences are not causal estimates. Do not choose the higher run or present192.6as reliable steady performance.

TP4/DCP4,MTP3probabilistic,NVFP4KV,maxseq24,batch4096,graphs4ranks,threshold4096,4x300W verified. Image `sha256:e6ae66f1d9f7728100c415bf4ef7b33c53da5f8d33dd9d87d95d02699e0762fb`. Startup KV23,274,509tokens. Source hashes matched. Candidate and production both stopped after successful completion; clients not restarted. Full-model quality remains pending; exact component mechanism still passes. Targets unmet, candidate not adopted.

Raw results/logs/metrics/telemetry and commands retained under results-01/combo-k5-funnel. First run remains ../combo-k5-funnel-speed-01; do not overwrite it. Repeat supersedes the tentative served-gain interpretation, not its recorded measurements. Both included in publication handoff.
