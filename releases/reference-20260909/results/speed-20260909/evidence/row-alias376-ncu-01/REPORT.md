# Row-alias376 achieved-residency profile

Pinned candidate image0c12fe9357070c82428109342f4dc3bbfff3ecb9c510100a4a654696fbd6ed62, original checkpoint rank0 K4 layer4 and K5 layer3, synthetic M4096 routes, one profiled grouped FC1 launch per rate. Source probe hashes, commands, weights and Nsight reports retained. No production service or benchmark ran concurrently.

| Metric | K4 | K5 |
|---|---:|---:|
| FC1 CTAs |376|376|
| Threads/CTA |128|128|
| Registers/thread |236|238|
| Dynamic shared bytes |35840|39936|
| Theoretical blocks/SM |2|2|
| Achieved occupancy |16.03%|16.06%|
| Achieved active warps/SM |7.69|7.71|
| Profiled kernel duration |1.40ms|1.49ms|

This supports actual use of near-two-block residency for the synthetic grouped workload, beyond CUDA API capacity alone. The earlier original K4 M4096 profile used128 CTAs and reported8.33% occupancy/4warps and2.17ms. That historical profile is not an independent matched current-run speed control; do not turn its duration ratio into a production speedup claim.

K4 scheduler eligible cycles52.55%, no eligible warp47.45%, eligible warps/scheduler0.71. Compute throughput51.69%, DRAM31.64%. These support latency/scheduling investigation, not a unique bottleneck or an attainable whole-model speedup percentage. PC-sampling counters were not collected; WarpStateStats explicitly warns smsp__pcsamp_sample_count is missing. Obtain source/PC counters before attributing stalls to a particular instruction.

Serving screen:8197tokens/s32K prefill,188.3053tokens/s C1 and310.7838aggregate C4 at8K, one pass. Targets300C1/20000prefill remain unmet; no adoption or full quality qualification. Production remains stopped.
