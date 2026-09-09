# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 0K C1 decode | 195.372 | 92.3 |
| 0K C4 decode aggregate | 333.729 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. Unchanged selected FP8 token-map-hoist image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. This adds 0K C1/C4 after the expanded matrix, with unchanged image/runtime settings and a fresh server allocation. These cells have no matched control. This is a single-run exploratory comparison; output and MTP variation remain. Plain one-shot cutoff is131072bytes; fused cutoff86016bytes and two-shot maximum786432bytes are verified from runtime logs. No adoption or publication. Production and clients remain stopped.
