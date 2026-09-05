# Full-model P8 completion audit

Pre-result implementation addition, 2026-09-04. The all-layer build is live;
the full-model KLD and speed services are still waiting for their prerequisites.

The KLD analyzer reads window means using pandas, which can skip NaN values.
The run writer already validates teacher identities and the intended window
set, but the follow-on should independently check the retained token records
and full layer/rank dispatch before using the result as a completed native run.

Require all 168 `(layer, rank)` pairs in both `WEIGHTS_READY` and `FORWARD`
events, each naming the expected K4/E4M3/UE8M0/32 contract and design. Require
exactly the 32 conditional-fit windows, eight in each domain, with every ordered
causal position, finite nonnegative token KLD, matching teacher/student/role
metadata, matching record hashes, and agreement with the frozen analyzer.

This audit verifies completeness and execution coverage. It does not change
the encoder, runtime, statistical estimator, decision threshold, evaluation
role, or either previously sealed plan. It does not open additional logits.
The queued speed service now runs this audit before taking the GPU lock.

Validation: eight CPU tests cover complete coverage and corruption of the
dispatch law, design, position sequence, finiteness, domain, teacher identity,
record receipt, and window inventory. A read-only replay of the existing
contextual control verifies 32 windows and 65,504 causal positions.

The contextual decoded-GPTQ control was served TP4/EP4/DCP4, while the new
native P8 candidate uses TP4/no-EP/DCP1. Together with the differing transformed
layer set and rate, this further limits causal interpretation of that contrast.
The absolute teacher KLD remains the primary full-model measurement.
