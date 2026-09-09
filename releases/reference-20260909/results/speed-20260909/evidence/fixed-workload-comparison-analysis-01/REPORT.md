# Fixed-workload candidate comparison

One fresh server per candidate, preselected order B then A. A is rowalias376 + FC2grid564 + overlap4096. B adds direct FC1 K5 funnel extraction. Both TP4/DCP4, MTP3, CUDA graphs, NVFP4 KV, maxseq24, batch4096, 4 x 300W. Existing pinned llm_decode_bench.py with fixed prefix and temperature0 for measured decode requests. The benchmark warmup still uses model-default temperature. This diagnostic is separate from earlier randomized-prefix/model-default-temperature rates.

| Metric | A tokens/s | B tokens/s | B vs A |
|---|---:|---:|---:|
| 32K cold prefill | 8058.00 | 8197.00 | +1.72% |
| 64K cold prefill | 8022.00 | 8136.00 | +1.42% |
| 8K C1 decode | 193.23 | 184.54 | -4.50% |
| 8K C4 decode aggregate | 332.19 | 332.31 | +0.04% |

The earlier apparent C1 benefit is not supported: B is slower on C1 in this pair, and its preceding model-default repeat also failed to reproduce the first gain. C4 is effectively tied. Small prefill differences do not isolate a decode-only kernel mechanism. Both prefill rates remain well below the 20,000 target; neither C1 reaches 300. Do not adopt or publish B as a demonstrated served speedup.

One ordered pair cannot separate thermal/order/routing effects or establish statistical confidence. Fixed prefix and temperature0 do not prove identical full-model execution. MTP draft settings remain probabilistic; C1 acceptance differed (A 0.56481, B 0.49772). No full-model quality qualification is established. All decode rows report zero errors, no underfill, no warmup timeout and no capacity-limited flag; raw flags and telemetry remain in SUMMARY.json and source JSON.

Server startup reports KV capacity A 23,274,509 and B 23,281,045 tokens. The benchmark separately infers approximately29.17M from block metrics; preserve that discrepancy rather than treating its derived value as authoritative capacity. These are token capacities, not byte sizes or guaranteed concurrency under every workload.

Both benchmark processes exited successfully; both fixed-prefix receipts and temperature arguments were checked. The preservation service completed and the supplemental archive checksum passed. Original production and both serving candidates remain stopped. Next bounded component test extends the exact integer extraction to direct FC2 against B as incremental control; it does not assume the served C1 gain exists. No automatic serving test or adoption follows that component test.
