# FC1 expert-local traversal result

All172exact K4/K5 representative comparisons passed, including changed-input CUDA graph replays. CPU tests execute the actual mapping AST for378ragged/empty/tail cases; reversing the scheduling prefix restores the entire reference kernel AST. Both compiled grouped arms contain only QMMA.SF.16832.F32.E4M3.E4M3.E8 MMA instructions.

The M4096 full TP-local MoE graph was slower in both captures for both stored rates. These are component milliseconds on fixed synthetic routing, not serving tokens/sec or independent server replications.

| Arm | Capture | Reference ms | Candidate ms |
|---|---:|---:|---:|
| r0-k4-ordinary | 0 | 3.307010 | 3.346967 |
| r0-k4-ordinary | 1 | 3.316655 | 3.354222 |
| r0-k5-ordinary | 0 | 3.536426 | 3.557682 |
| r0-k5-ordinary | 1 | 3.546112 | 3.570498 |

Decision: preserve as a negative result; do not launch a full-model serving screen or adopt this mapping. Extra metadata/division and altered activation locality are possible explanations, not established causes. No new llm_decode_bench throughput exists for this candidate. The retained FP8 grid376 reference remains unchanged; original production and clients remain stopped. No P4, TP2, checkpoint change or publication.
