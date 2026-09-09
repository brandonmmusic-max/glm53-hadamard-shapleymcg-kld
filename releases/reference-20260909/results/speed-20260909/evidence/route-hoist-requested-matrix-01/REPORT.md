# Requested FP8 token-map-hoist benchmark results

Actual pinned llm_decode_bench measurements. Same image and TP4/DCP4/MTP3 runtime settings, CUDA graphs, NVFP4 KV cache, batch4096, maxseq24, four300W GPUs. At least90seconds cooldown before every cell, with all GPUs at or below55C for30seconds.

Prefill (client prompt tokens / TTFT):

| Context | Tokens/s |
|---|---:|
| 32K | 8457 |
| 64K | 8443 |
| 128K | 8323 |

Speculative sustained decode (C4/C8 values are aggregate):

| Context | C1 tokens/s | C4 tokens/s | C8 tokens/s |
|---|---:|---:|---:|
| 0K | 204.61 | 327.67 | Not tested in this matrix |
| 8K | Not tested in this matrix | Not tested in this matrix | 594.64 |
| 16K | Not tested in this matrix | Not tested in this matrix | 602.96 |
| 32K | 204.93 | 329.34 | Not tested in this matrix |
| 64K | 199.45 | 330.10 | Not tested in this matrix |
| 128K | 204.44 | 334.09 | Not tested in this matrix |

All completed decode cells have zero request errors and no underfill, warmup timeout or capacity-limited flag. This is exploratory performance evidence, not a model-quality qualification or independent replication. Neither the300 tokens/s C1 nor20,000 tokens/s prefill target was achieved.

The expanded matrix ran in one server session. 0K C1 and C4 ran in two subsequent sessions on the identical image. The C1 session completed its benchmark then failed an inappropriate padding-log assertion; its structured workload receipt, raw hash, cooldown and runtime were independently audited. C4 completed after correcting that runner check. Original failure evidence is retained.

Explicit server KV capacities by run:

- tile-policy-route-hoist-expanded-02: 23,562,091 tokens.
- tile-policy-route-hoist-zero-01: 23,568,627 tokens.
- tile-policy-route-hoist-zero-02: 23,568,627 tokens.

The benchmark generic block-times-DCP estimate differs from explicit hybrid-cache capacity; use server kv_cache_size_tokens matching startup logs. See tile-policy-route-hoist-expanded-02/kv-capacity-audit.json. Source summaries and their hashes are in SUMMARY.json. Production and clients remain stopped. No image adoption or publication occurred.
