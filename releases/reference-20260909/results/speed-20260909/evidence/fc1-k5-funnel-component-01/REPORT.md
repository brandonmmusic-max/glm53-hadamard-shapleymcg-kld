# K5 direct FC1 funnel extraction

35,104CPUchecks passed. All32K5lane geometries satisfy the two-word/three-word relation used by the optimized funnel. Zero plus96single-bit basis vectors per lane prove the linear mapping of admissible source words;1000random triples per lane add checks. Two-word alias m=b is enforced. MCG rounding, packing and MMA are unchanged.

172representative GPU comparisons passed for rank0K4/K5M4/M16/M4096: complete TP-local outputs, routed outputs, intermediate regressions, nonfinite eager stress and changed-input captured graphs. No shared-memory barriers or buffers removed. The change simplifies only directK5FC1's read-window extraction. K4, grouped prefill andFC2 remain original. Distinct module imports and compile-key fact isolate candidate artifacts.

Paired median full-component graph time, two captures each:

|Rate|M4|M16|M4096|
|---|---:|---:|---:|
|K4|-0.03%/-0.62%|-0.03%/+0.14%|-0.01%/+0.07%|
|K5|-5.95%/-5.33%|-0.78%/-1.07%|+0.11%/+0.12%|

Negative is faster. K5M4improves about5-6%,K5M16about1%; unchanged arms effectively flat. Fixed synthetic routes, representative real weights, five balanced20-replay samples/capture. These are full TP-local MoE component times, not served rates or full-model results.

Emitted K5FC1registers183->181,STACK0LOCAL0both. Static SASS instruction counts4427->4385; opcode details retained. Static counts are not dynamic timing fractions or speedup bounds. CUDA ELF and parent objects hash-verified; CPU-only cuobjdump exports. This small measured mechanism warrants a serving screen but cannot predict300C1.

Production remains stopped. Component unit finished successfully; no image adopted or serving throughput measured in this test. Next screen should explicitly compose with the user's retained combined candidate, verify source scope, and use32K/64Kprefill and8KC1/C4 only. Full-model quality remains unqualified.
