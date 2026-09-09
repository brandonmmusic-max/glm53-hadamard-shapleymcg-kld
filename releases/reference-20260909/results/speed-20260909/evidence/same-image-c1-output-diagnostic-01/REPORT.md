# Same-image repeat result

Actual llm_decode_bench 8K C1 decode: **203.731 tokens/sec**, compared with **215.701 tokens/sec** in the preceding recorded run of the same immutable wait-before-prefetch image. Both use the stream recorder; these are diagnostic timings, not replacements for ordinary serving screens. KV capacity:23,418,300tokens. No request errors, underfill, or warmup timeout.

The exact measured request body matches. Captured reasoning text differs at zero-based index331 (character332), after a331-character common prefix. The earlier cross-image pair differed at character242. This demonstrates within-image output variation in two fresh-server runs. It does not prove whether the variation comes from a candidate-specific defect, another runtime component, or floating-point/batching behavior, and it does not itself establish quality degradation. No token-ID/logit equivalence is claimed.

The existing prefill screen measured wait-first at8294/8228tokens/sec for32K/64K versus grid376 at8325/8240. Combined with unstable free-generation decode comparisons, these results do not support adopting wait-first. Retain the source and negative/positive observations without selecting the best single run. No additional automatic repeats under this plan. Resume optimization from the retained FP8 grid376 candidate rather than bundling this unqualified scheduling change into another candidate.

Plan/source hashes and untruncated capture validated. Server stopped at terminal; production and clients remain stopped. No publication, P4/TP2 work, or checkpoint changes.
