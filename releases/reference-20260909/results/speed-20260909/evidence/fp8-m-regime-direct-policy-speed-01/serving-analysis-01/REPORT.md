# Actual llm_decode_bench results

| Test | Tokens/sec |
|---|---:|
| 32K prefill | 8274.000 |
| 64K prefill | 8197.000 |
| 8K C1 decode aggregate | 196.589 |
| 8K C4 decode aggregate | 316.221 |

KV capacity: 23,405,228 tokens. TP4/DCP4, MTP3 probabilistic, NVFP4KV, graphs, maxseq24, batch4096, four300W GPUs.

Single corrected M16/M32/M64 candidate. Fixed prefix and temperature0 diagnostic; compare with the same prior protocol, not randomized/model-default history. Raw result hashes and source identities verified. No automatic adoption or publication. Full-model quality remains unqualified.
