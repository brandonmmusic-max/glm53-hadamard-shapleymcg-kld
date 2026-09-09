# Paired 8K C1 output diagnostic

| Image | llm_decode_bench decode tokens/sec |
|---|---:|
| Retained grid376 | 196.789 |
| FC1 wait-before-prefetch | 215.701 |

These are instrumented diagnostics, one fresh server per image, in grid376 then wait-first order. TP4/DCP4, MTP3 probabilistic with standard rejection, NVFP4 KV, CUDA graphs, maxseq24, batch4096, 300 W per GPU. Both completed the 20-second C1 cell without request errors, underfill, capacity limitation, or warmup timeout. No new prefill or C4 measurement was made.

All five captured request bodies match across images by SHA256 and exact body comparison. The measured temperature-zero request (ordinal4, hash13d3648ce95a6d64d422989d4c6b764c44f6728a6c835ef37de6f343d61f392c) has a common reasoning-text prefix of241characters and diverges at zero-based index241. The warmup temperature-zero stream (ordinal2) also diverges, at index336. No captured stream was truncated; no malformed SSE or server error was detected. Duration-limited requests end without a DONE marker as expected from benchmark cancellation; comparisons use their common captured prefixes.

This establishes different generated text for identical request bodies in this pair. It does not identify the first differing token/logit, prove the FC1 change caused divergence, or establish quality degradation. Kernel arithmetic/component equality does not establish full-model greedy trajectory identity. The prior165.399t/s wait-first result did not reproduce here, but this test differs in observer overhead and pre-test history (C1-only rather than prefill followed by C1/C4). Do not attribute the reversal to the kernel or compare it as independent confirmation of a speed gain.

Both containers are stopped; original production and clients were not restored. No candidate is adopted. Sources, launch configuration, raw JSON, SSE, logs, and validation are retained locally.

Next diagnostic decision: distinguish within-image trajectory variation from between-image effects before another optimization. A same-image repeat with captured output can answer reproducibility; an identical-prefix forced-decode comparison can locate numerical divergence if needed. This report does not authorize an unplanned automatic repeat, a systematic sweep, or publication.
