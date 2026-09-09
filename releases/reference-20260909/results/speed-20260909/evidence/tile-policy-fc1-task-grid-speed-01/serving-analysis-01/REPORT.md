# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 32K prefill | 8439.000 | 92.3 |
| 64K prefill | 8451.000 | 92.3 |
| 8K C1 decode | 212.098 | 92.3 |
| 8K C4 decode aggregate | 325.083 | 92.3 |
| 16K C1 decode | 208.708 | 92.3 |
| 16K C4 decode aggregate | 353.532 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. Direct-FC1 task-grid derivative of selected FP8 token-map-hoist image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. The comparison is the prior cooled token-map-hoist six-cell run. Only directM32 FC1 launch grid changes; all six benchmark cells match. This is a single-run exploratory comparison; output and MTP variation remain. Plain one-shot cutoff is131072bytes; fused cutoff86016bytes and two-shot maximum786432bytes are verified from runtime logs. No adoption or publication. Production and clients remain stopped.
