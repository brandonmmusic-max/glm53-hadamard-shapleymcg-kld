# Tile policy with direct routing preserved

Corrects the first policy experiment by retaining direct routing through M16. The requested policy remains M16 for M1, M32 for M2..128 and M64 above128; N128 unchanged.

CPU row/arena checks and inherited arithmetic AST checks passed. 266 exact GPU comparisons passed, including route outputs and changed-input CUDA graph outputs. Raw intermediate bytes compared at M1 only because physical row padding changes elsewhere. All eight compiled K4/K5 direct/grouped candidate arms contain exclusively FP8 E4M3 MMA with FP32 accumulation.

| Rate | Input M | Elapsed change capture0 | Capture1 |
|---|---:|---:|---:|
| K4 | 1 | -0.10% | -0.22% |
| K4 | 4 | +6.47% | +6.39% |
| K4 | 16 | +13.24% | +12.88% |
| K4 | 128 | -22.90% | -21.67% |
| K4 | 129 | +0.12% | +0.60% |
| K4 | 4096 | +0.25% | +0.20% |
| K5 | 1 | -0.00% | -0.02% |
| K5 | 4 | +6.16% | +6.77% |
| K5 | 16 | +5.88% | +7.34% |
| K5 | 128 | -19.90% | -19.24% |
| K5 | 129 | +0.47% | +0.32% |
| K5 | 4096 | +0.58% | +0.61% |

Positive means slower; full TP-local MoE graph times, not model throughput. Direct M32 retains one useful row per route while expanding arithmetic/storage, so the dense expected-M rule does not increase useful rows per expert. The prior large small-batch regression included the extra grouped-routing work; the corrected result isolates that confound and still regresses M4/M16. M128 remains improved; M64 prefill is unchanged. No adoption or new serving baseline. Preserve both experiments.

CPU setup initially caught a missing output arena region; it was corrected before GPU work. A prematurely submitted service failed because runner generation had not finished; no GPU execution occurred for that service. Both setup errors are recorded in SETUP-FAILURE.json.
