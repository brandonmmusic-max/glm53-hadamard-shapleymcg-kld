# Direct FC2 K5 funnel extraction

35,104 CPU bit-extraction checks and 172 exact GPU component comparisons passed. GPU comparisons include real representative K4/K5 weights, M4/M16/M4096, changed-input graph replay, intermediate bytes and routed/full component outputs. This is not full-model quality qualification.

Incremental baseline is retained combo-k5-funnel, not original production. Separate kernel modules and a distinct native compile fact isolate the arms. Only direct FC2 dispatch imports the proven K5 funnel helper; FC1 and grouped prefill are unchanged. Hash-verified CUDA ELF exports show FC2 K5 registers165 to161 with zero stack/local storage; FC1 stays181.

Full local MoE paired median elapsed-time changes, two captures: K5 M4 -3.08%/-3.30%; K5 M16 -2.61%/-0.43%. Unchanged K4 and M4096 controls are within0.55%. Samples warm/drift within captures, so this is exploratory timing evidence from one process per rate; do not equate graph replays to independent runs or infer served speedup.

Decision: exactness and the consistent M4 signal justify one narrow served viability screen, using the existing fixed-prefix temperature0 diagnostic and retaining MTP3. No automatic adoption/full-suite follows. Compare to preceding candidate diagnostics with order/thermal/MTP limitations; earlier direct FC1 served benefit is not assumed.
