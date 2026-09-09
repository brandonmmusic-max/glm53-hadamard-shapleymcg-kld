# PC sampling: production decode and row-alias376 grouped FC1

Four profiles completed: rank0 K4/K5, M4 direct decode and M4096 grouped prefill, original real weights and pinned candidate image. No concurrent serving/GPU workloads. Raw Nsight reports, full SASS CSV and summary.json retained. No source-line metadata was added; attribution is to instruction PCs/opcodes, not source lines.

| Profile | Samples | Selected | Dependency wait | Other major reason |
|---|---:|---:|---:|---|
| K4 M4 |4319|52.10%|27.83%|barrier6.23%|
| K5 M4 |4892|48.98%|26.47%|barrier7.79%|
| K4 M4096 |101799|27.54%|24.75%|math-pipe throttle13.86%|
| K5 M4096 |106898|26.84%|25.69%|math-pipe throttle12.02%|

Percentages use all sampled warp states, including selected; not fractions of whole-model runtime. AtM4, dependency-wait PCs frequently contain LOP3, PRMT and SHF used in reconstruction/alignment. AtM4096, QMMA dominates the wait-opcode group with substantial reconstruction and shuffle involvement. A stalled PC can wait for a previous instruction, so these findings do not prove that the named opcode is independently expensive or that deleting it yields its sample share as speedup.

Next bounded hypothesis: replace decode-FC1 MCG state arithmetic with a64KiB global read-only table. Every16-bit state maps to one E4M3 byte. CPU exact-integer rounding validation and native production PTX GPU validation both matched all65536entries atK3/K4/K5. This preserves checkpoint codes, ring extraction, scales, FP8 MMA and transformations; caching/random-load cost might still make it slower. Component A/B running in../fc1-mcg-lut-component-01, no serving adoption or gain claim.

Production remains stopped. Scope remains P8 production optimization; no P4, TP2 or repeated agent-review loop.
