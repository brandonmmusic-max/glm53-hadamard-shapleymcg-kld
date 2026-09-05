# Index-order device v1: host identity gate stopped

The first synthetic GPU0 process passed all **380 case calls and 80 state-transition calls**, including eager and CUDA-graph assertions. The enclosing five-process experiment nevertheless **failed**, accepting zero repeats and never starting repeats 2–5. This is not a five-process determinism pass or full-model numerical closure.

The frozen host comparison required NVIDIA's `GPU-4d808f8d-88c8-8d7d-cb31-d077a6fd28ac`; PyTorch reported `4d808f8d-88c8-8d7d-cb31-d077a6fd28ac`. These strings differ only by `GPU-`. The snapshot separately replays the probe inventory with an in-memory normalized UUID to document the formatting issue; it does not alter or retroactively accept the raw failed execution.

The probe container exited 0. The unit terminated failed/MainPID 0; no comparison-owned container remained. Recorded restoration returned the previously active backend and left the previously inactive timer inactive. Maximum sampled temperature was 34°C. GPU work occurred; no model, teacher logits, or protected roles were loaded. No KLD, speed, allocation, or full-model correctness claim follows.

The candidate orders valid short (≤512 pooled-token) outputs logically while preserving paired scores, suffixes and completed-state counters; longer and negative-page cases retain the legacy semantic contract. Positive out-of-range page IDs were not GPU-tested because the original scorer requires valid addresses. The experiment uses fresh processes and an isolated shared compiled cache, not five cold-JIT trials.

Evidence: [snapshot manifest](../evidence/opened/codec-v2/p8-index-order-device-v1-stop/snapshot.json), [failed root execution](../evidence/opened/codec-v2/p8-index-order-device-v1-stop/execution.json), [unchanged passing probe](../evidence/opened/codec-v2/p8-index-order-device-v1-stop/repeat-01/result.json), [bounded report](../evidence/opened/codec-v2/p8-index-order-device-v1-stop/report.json).

Identities: plan `5ed0f8c3268dbdc05d039335b08884f033c6642f2c7f3781fc502bc83873e021`; root execution `23b973a5225823fd228941ae72200e52c67b1148ebc0885058879e17bf7339d3`; raw probe `fe6f56c73dc1d90fff28cc41535b8b4dfd3c4d7c390f017e9804ac8489104574`. All nine sealed sources were verified unchanged. Raw logs are hash-referenced only; the launch is copied only after equality with the exact frozen synthetic-only argument builder.
