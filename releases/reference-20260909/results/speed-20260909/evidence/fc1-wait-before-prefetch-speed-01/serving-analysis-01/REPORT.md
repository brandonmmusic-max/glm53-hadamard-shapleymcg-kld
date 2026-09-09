# Actual llm_decode_bench results

| Test | Tokens/sec |
|---|---:|
| 32K prefill | 8294.000 |
| 64K prefill | 8228.000 |
| 8K C1 decode aggregate | 165.399 |
| 8K C4 decode aggregate | 329.109 |

KV capacity: 23,411,764 tokens. TP4/DCP4, MTP3 probabilistic, NVFP4KV, graphs, maxseq24, batch4096, four300W GPUs.

Single FC1 wait-before-prefetch scheduling candidate. Fixed prefix and temperature0 diagnostic; compare with the same prior protocol, not randomized/model-default history. Raw result hashes and source identities verified. No automatic adoption or publication. Full-model quality remains unqualified.
