# Requested M16/M32/M64 policy result

266 exact GPU comparisons passed across K4/K5 and M1/M4/M16/M128/M129/M4096, including canonical route outputs and changed-input CUDA graphs. Raw intermediate byte comparison only at M1 because grouped physical packing differs. CPU policy boundaries, row coverage, scratch extents and kernel-body AST checks passed. All six candidate compiled regimes use QMMA.SF.16832.F32.E4M3.E4M3.E8; no target W4A16 substitution.

| Stored rate | Input M | Median elapsed change, capture 0 | Capture 1 |
|---|---:|---:|---:|
| K4 | 1 | -0.13% | -0.31% |
| K4 | 4 | +111.70% | +113.56% |
| K4 | 16 | +46.20% | +40.45% |
| K4 | 128 | -22.84% | -21.59% |
| K4 | 129 | +0.33% | +0.42% |
| K4 | 4096 | +0.15% | +0.05% |
| K5 | 1 | -0.06% | +0.03% |
| K5 | 4 | +109.33% | +110.50% |
| K5 | 16 | +36.11% | +31.73% |
| K5 | 128 | -19.70% | -19.36% |
| K5 | 129 | +0.73% | +0.39% |
| K5 | 4096 | +0.47% | +0.67% |

Positive means slower. This is full TP-local MoE component time, not served throughput. The policy switches M2..16 from direct routing to grouped expert tiles, introducing grouping and padded work; it is not an isolated dense GEMM tile substitution. M128 improves, while M4/M16 regress substantially. No adoption or serving qualification is justified by these results. M1/M129/M4096 are unchanged-path controls. Keep original production stopped per current user instruction.

The first cubin extraction omitted trailing program headers and failed disassembly; that failed extraction is preserved in resource-audit-01. Corrected extraction in resource-audit-02 succeeded against the intact compiled objects.
