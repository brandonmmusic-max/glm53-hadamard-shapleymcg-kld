# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 32K prefill | 8424.000 | 92.3 |
| 64K prefill | 8411.000 | 92.3 |
| 8K C1 decode | 203.180 | 92.3 |
| 8K C4 decode aggregate | 314.277 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. Same selected FP8 tile-policy image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. The historical16channel reference did not use this per-cell cooldown and split-invocation protocol; do not attribute differences solely to channels. No adoption or publication. Production and clients remain stopped.
