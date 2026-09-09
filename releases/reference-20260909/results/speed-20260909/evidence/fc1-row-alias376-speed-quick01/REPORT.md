# Row-local scratch plus376-CTA grouped FC1 serving screen

TP4/DCP4/MTP3, NVFP4 KV, FULL_AND_PIECEWISE graphs, maxseq24, batch4096,4x300W; original checkpoint. Separate image and all modified source hashes pinned in plan-01.json. One fresh server, one pass.

| Requested measurement | Tokens/s |
|---|---:|
| Cold32K prefill |8197|
| Speculative8K C1 decode |188.3|
| Speculative8K C4 aggregate decode |310.8|

Prefill:5 requests,32770 reported prompt tokens,4.00s mean TTFT. Decode:8192 output cap,20-second measurement windows. Exact decode values/flags/acceptance are in SUMMARY.json and raw JSON. No production throughput baseline rerun.

Prefill is4.58% above the immediately preceding decode-stream candidate screen7838, and2.46% above the user's approximate8000 reference. These are exploratory comparisons, not a matched baseline speedup estimate. The decode kernel is unchanged by this candidate; do not attribute its higher decode row to this patch. Full-model quality not qualified; smoke returned4 with leaked reasoning text, which is retained in raw response.

Prior component evidence:172 exact comparisons, dynamic shared35/39KiB instead of64KiB, CUDA API occupancy2 instead of1block/SM, approximately16% lower TP-local MoE graph time for synthetic M4096 routes. Nsight achieved-residency profiling is the next test. No adoption or wider suite yet. Production and candidate are stopped after the run.
