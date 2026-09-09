# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 32K prefill | 8423.000 | 92.3 |
| 64K prefill | 8448.000 | 92.3 |
| 8K C1 decode | 214.588 | 92.3 |
| 8K C4 decode aggregate | 326.620 | 92.3 |
| 16K C1 decode | 217.293 | 92.3 |
| 16K C4 decode aggregate | 343.859 | 92.3 |
| 0K C1 decode | 185.422 | 92.3 |
| 0K C4 decode aggregate | 325.106 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. FP8 FC1 grid-floor derivative of selected token-map-hoist image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. This measures 32K/64K prefill and 0K/8K/16K C1/C4 decode. Prior cooled reference measurements are available; separate sessions and MTP variation limit causal claims. This is a single-run exploratory comparison; output and MTP variation remain. Plain one-shot cutoff is131072bytes; fused cutoff86016bytes and two-shot maximum786432bytes are verified from runtime logs. No adoption or publication. Production and clients remain stopped.
