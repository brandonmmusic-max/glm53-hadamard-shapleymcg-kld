# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 32K prefill | 8409.000 | 92.3 |
| 64K prefill | 8409.000 | 92.3 |
| 8K C1 decode | 212.927 | 92.3 |
| 8K C4 decode aggregate | 326.797 | 92.3 |
| 16K C1 decode | 202.200 | 92.3 |
| 16K C4 decode aggregate | 321.870 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. Same selected FP8 tile-policy image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. The comparison is the prior cooled NCCL8 screen with an84KiB one-shot cutoff. Its first four cells match;16K has no matched prior reference. This is a single-run exploratory comparison; output and MTP variation remain. Plain one-shot cutoff is131072bytes; fused cutoff86016bytes and two-shot maximum786432bytes are verified from runtime logs. No adoption or publication. Production and clients remain stopped.
