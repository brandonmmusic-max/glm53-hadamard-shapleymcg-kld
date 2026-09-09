# Decode FC1 streamed-reconstruction screen

One TP4/DCP4/MTP3 server; NVFP4 KV, graphs FULL_AND_PIECEWISE, maxseq24, batch4096, 4x300W. Candidate image and sources are pinned in plan-01.json. No production throughput baseline rerun.

| Measurement | Tokens/s |
|---|---:|
| Cold prefill32K |7838|
| Speculative decode8K C1 |176.5739|
| Speculative decode8K C4 aggregate |297.6777|
| C4 aggregate divided by4 |74.4194|

Decode uses8192 output cap and20-second measurement windows. No errors, underfill, warmup timeouts or capacity limitation in either cell. Acceptance-rate fields are0.49074 C1 and0.43056 C4; raw metrics retained. Prefill has5 requests, TTFT approximately4.18s. Do not compare this8K C1 result to a zero-context or512-output-cap result as a matched delta.

No established useful production speedup. Do not adopt or expand this candidate to a fuller run. Grouped prefill remained on the original kernel in this isolated candidate. Full-model quality is not qualified. Next candidate applies streamed reconstruction specifically to grouped prefill.

KV startup reports23411764 tokens; benchmark metrics derive29343744 (3582*2048*4). These are distinct accounting paths and are not reconciled here.

Production and candidate containers were verified stopped after completion; no production restart. The inherited runner output result.json contains a stale batch16384 status label; the pinned plan, runtime, source hashes and raw commands establish this as FC1-stream with batch4096.
