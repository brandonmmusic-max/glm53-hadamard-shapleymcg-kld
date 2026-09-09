# Shared-prefill overlap4096 viability screen

Completed successfully. Not adopted; no fuller benchmark justified. Original production and candidate containers both verified stopped after the run. Client services were not restarted.

| Measurement | Earlier row-alias376 screen | Threshold4096 screen | Observed difference |
|---|---:|---:|---:|
|32K cold prefill t/s|8197|8218.0|+0.26%|
|8K C1 speculative decode t/s|188.30534|176.64513|-6.19%|
|8K C4 aggregate speculative decode t/s|310.78385|320.29388|+3.06%|

C4 aggregate divided by4 is80.073t/s per active request. Prefill TTFT3.988s,5 cold requests within one server run. Each decode cell used one20-second measurement window and output cap8192; no request errors, underfill, warmup timeout or capacity limitation. These are speculative served rates, not target-only rates. Duration-window requests need not finish naturally.

Only the shared-expert stream threshold changed from256 to4096 relative to the retained row-alias376 image. TP4/DCP4, MTP3 probabilistic, NVFP4 KV, maxseq24, batch4096, FULL_AND_PIECEWISE graphs on4ranks, and4x300W were verified. Image `sha256:0c12fe9357070c82428109342f4dc3bbfff3ecb9c510100a4a654696fbd6ed62`. Startup KV capacity23,326,797tokens; this is startup allocation, not a benchmark concurrency capacity guarantee. Actual threshold was independently read through installed vLLM envs. Kernel/model sources were hash checked. No source/checkpoint arithmetic edit in this test.

The prefill delta is too small to establish improvement from two historical single-server observations. Decode eligibility under the threshold is unchanged for small shapes; C1 decreased and C4 increased. MTP acceptance fields were0.412037and0.534946, versus earlier0.527778 and0.455556. Do not attribute those speculative decode differences to the scheduling setting. Clock/temperature/cache/run variability remain rival explanations; no independent replication or confidence interval.

Existing synchronization is retained. Extending stream eligibility is verified; achieved overlap and a complete critical path are not measured by this run. Prior Nsight exports lack graph edges and usable event timestamps. No collective/wait removal is warranted by those exports.

Quality is not qualified. Arithmetic-preserving row-alias component evidence remains172exact checks; this config-only run does not replace matched full-model KLD. Smoke returned4 while exposing reasoning text, also seen in the earlier screen; raw response preserved. No new KLD, full suite, target-only baseline or production throughput rerun. Targets300C1/20000prefill remain unmet.

Evidence: `plan-01.json` and its hash; `SUMMARY.json`; `results-01/shared-overlap4096/rep-1/{prefill,decode-cap8192}.json`; launch, startup, source hashes, runtime, hardware telemetry and raw metric snapshots in results directory. Benchmark SHA256 `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`. Plan uses original pinned benchmark and existing production identity; historical reference is `../fc1-row-alias376-speed-quick01/REPORT.md`. This is exploratory viability evidence only.
