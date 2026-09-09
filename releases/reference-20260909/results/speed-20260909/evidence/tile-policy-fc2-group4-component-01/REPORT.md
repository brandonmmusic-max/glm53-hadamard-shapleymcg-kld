FC2 groups-of-four candidate rejected for serving testing. All172 exact full-local-MoE and changed-input graph comparisons passed. Both affected compiled K4/K5 FC2 variants contain QMMA.SF.16832.F32.E4M3.E4M3.E8. Only task traversal changed.

Both captures on both stored rates show slower M4096 component execution. These are component milliseconds, not llm_decode_bench tokens/sec. No new serving benchmark was run for this candidate.

| Rate | Capture | Baseline median ms | Candidate median ms | Paired median delta ms |
|---|---:|---:|---:|---:|
| r0-k4-ordinary | 0 | 3.261946 | 3.313443 | 0.023071 |
| r0-k4-ordinary | 1 | 3.268600 | 3.310395 | 0.014538 |
| r0-k5-ordinary | 0 | 3.487119 | 3.518726 | 0.019542 |
| r0-k5-ordinary | 1 | 3.485302 | 3.524110 | 0.032709 |

M4/M16 use unchanged direct paths; their timing variation is control noise and is not evidence of a grouped-kernel speed gain. Preserve this negative result; do not build or deploy a candidate image. Selected token-map-hoist image remains unchanged. Production and clients remain stopped. Broader speed targets are unmet.
