# Actual cooled llm_decode_bench results

| Test | Tokens/sec | Cooldown seconds |
|---|---:|---:|
| 32K prefill | 8457.000 | 92.3 |
| 64K prefill | 8443.000 | 92.3 |
| 128K prefill | 8323.000 | 92.3 |
| 32K C1 decode | 204.930 | 92.3 |
| 32K C4 decode aggregate | 329.343 | 92.3 |
| 64K C1 decode | 199.446 | 92.3 |
| 64K C4 decode aggregate | 330.096 | 92.4 |
| 128K C1 decode | 204.445 | 92.3 |
| 128K C4 decode aggregate | 334.086 | 92.3 |
| 8K C8 decode aggregate | 594.639 | 92.3 |
| 16K C8 decode aggregate | 602.965 | 92.3 |

KV capacity: 23,562,091 tokens. NCCL8 channels verified from logs. Unchanged selected FP8 token-map-hoist image, TP4/DCP4/MTP3, graphs, NVFP4KV, maxseq24, batch4096, four300W GPUs.

Each cell follows at least90seconds idle and all four GPUs at or below55C for30seconds. In-cell temperatures/clocks are retained. This reduces thermal differences but does not guarantee identical sustained clocks or output trajectories. This extends the prior shorter-context token-map-hoist run with unchanged image and runtime. New context cells have no matched control. This is a single-run exploratory comparison; output and MTP variation remain. Plain one-shot cutoff is131072bytes; fused cutoff86016bytes and two-shot maximum786432bytes are verified from runtime logs. No adoption or publication. Production and clients remain stopped.
