# Relevance of the tile-policy component result to serving

Read existing four Nsight databases only. Verified annotation semantics against gpu_worker.py copied from the exact profiled image: execute_N counts total scheduled tokens, context/generation numbers count requests. CUDA-graph padding can make the model input shape differ; NVTX is not an exact MoE tensor-shape receipt.

| Capture | Scheduled-token regimes (ranges across four workers) |
|---|---|
| 8K C1 | 4: 472 |
| 8K requested C4 | 12: 200 |
| 32K prefill plus setup | 0: 20, 2: 4, 4: 88, 20: 4, 2044: 4, 2048: 12, 4096: 56 |
| 64K prefill plus setup | 0: 20, 2: 4, 4: 88, 20: 4, 2044: 4, 2048: 12, 4096: 92 |

The C1 capture has 4 scheduled tokens and one generation request per range. The requested-C4 capture has 12 scheduled tokens and three generation requests per range. Its full benchmark window separately reports effective concurrency4 and no underfill; these observations cover different intervals. Do not call the short profile a demonstrated sustained-C4 kernel mix. This does not invalidate the separate unprofiled C4 benchmark measurements.

No scheduled128-token regime appears. Grouped32's improvement at component M128 is not evidence of a gain for the observed direct decode or large prefill chunks. Prefill traces include setup, small tails and decode, so raw range counts are not prefill wall-time shares. Existing direct/grouped kernel attribution agrees with this limitation. Do not infer maximum attainable speed from summed overlapping kernel durations.

Decision: keep the requested M16/M32/M64 experiment as negative evidence for small decode and useful but currently unmatched M128 component evidence. Neither broad policy is adopted. No systematic sweep, new GPU profile, production restart or publication performed.
