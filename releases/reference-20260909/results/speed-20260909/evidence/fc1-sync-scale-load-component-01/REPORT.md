# Synchronous scale-copy result

172exact component comparisons passed for representative K4/K5 real checkpoint layers at M4/M16/M4096, including changed-input CUDA graphs. Two compiled grouped arms retain exclusively FP8 QMMA.SF.16832.F32.E4M3.E4M3.E8. CPU source reversal and aligned32-bit copy checks retained.

M4096 full local-MoE graph median times in two captures: K4 reference3.277986/3.279181ms versus candidate3.317971/3.319925ms; K5 reference3.500955/3.508105ms versus candidate3.526872/3.535696ms. These are component timings, not serving tokens/sec or independent server replicates.

Decision: reject before serving benchmark; preserve negative evidence. No llm_decode_bench result exists for this candidate. Switching the4-byte copy instruction path alone did not help. The source-counter evidence still supports investigating repeated scattered scale traffic, but does not prove that prepacking or caching scales will improve throughput. Retained grid376 source remains unchanged; no checkpoint/image adoption or publication. All test containers stopped, production and clients remain stopped.
