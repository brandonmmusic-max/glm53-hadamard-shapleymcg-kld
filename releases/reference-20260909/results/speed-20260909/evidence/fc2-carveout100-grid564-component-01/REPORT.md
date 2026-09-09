# FC2 carveout100 and564CTA component test

172 exact comparisons passed across rank0 K4/K5 and M4/M16/M4096. Eager tests cover route-concentrated inputs, signs, zero, NaN, Inf and negative zero. Captured graphs are replayed with changed inputs. Tests compare complete TP-local MoE outputs and routed expert outputs, plus FC1 payload regressions. Old raw `scope` strings say direct/grouped-FC1 because the reusable harness was inherited; actual compared tensors and new candidate are FC2/full-MoE as documented here. Nonfinite standalone reducer input qualification and full-model quality are not covered.

Only direct FC2 launch changes: preferred_smem_carveout=100 and564CTAs instead of2*max_active_clusters. Dynamic backend imports an isolated FC2 module only for the small-M branch. Grouped prefill uses original kernels. Distinct native compile-spec fact separates candidate cache entries. Kernel loops stride by actual grid dimension; route ownership and arithmetic order within each output are unchanged. Grid is qualified only for188SM hardware and harness verifies it.

Paired median full-component graph-time changes (negative faster), two captures with5balanced20-replay samples each:

| Rate | M4 | M16 | M4096 |
|---|---:|---:|---:|
|K4|-0.59%/-0.98%|-6.19%/-5.95%|-0.00%/-0.05%|
|K5|-3.28%/-2.87%|-5.02%/-3.74%|+0.19%/+0.18%|

Modest direct-decode component benefit warrants one narrow serving screen; it does not demonstrate a production speedup. Fixed synthetic routes, one representative layer/rate, one independent process/rate. M4/M16 approximate MTP3 C1/C4 shapes, not complete serving requests.

Extracted FC2 cubins show identical resource counts: K4REG143/K5REG165, STACK0, LOCAL0, static shared1024B. The candidate changes launch metadata and module naming, so entire cubin hashes need not match. Baseline/candidate objects and extracted CUDA ELF are hash-verified; resource exports performed without GPUs. Separate prior NCU confirms the carveout changes shared allocation65536->102400B and theoretical residency2->3blocks; no new achieved-residency measurement of564grid is claimed here.

Source/image/model untouched in original production; component unit exit0, production remains stopped. No publication or full suite. Artifact hashes, raw comparisons, inputs, paired timings and compiler objects retained.
