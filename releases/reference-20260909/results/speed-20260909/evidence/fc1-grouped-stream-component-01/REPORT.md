# Grouped-prefill streamed reconstruction

172 representative exact comparisons passed (rank0 K4/K5, M4/M16/M4096, eager and changed-input CUDA graphs). Grouped candidate dispatch is explicit and differs from the earlier decode-only harness.

Paired full TP-local MoE graph times atM4096 increased1.23%/1.17% for K4 and4.92%/5.17% for K5 across two captures. Each capture has five alternating-order samples; these are not independent serving runs or isolated FC1 measurements. Raw timings and compiled resources retained in resource-audit-01.

Decision: no promising speed evidence; do not spend a serving window or full matrix on this candidate. Preserve original image/checkpoint. Proceed to grouped row-local epilogue storage alias, which changes scratch layout rather than reconstruction scheduling. Production remains stopped.
