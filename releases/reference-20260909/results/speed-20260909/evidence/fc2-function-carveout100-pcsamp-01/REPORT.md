# FC2 explicit shared carveout diagnostic

Explicit `preferred_smem_carveout=100` at the FC2 CuTe launch changed the measured shared partition from65536 to102400B. K4/K5 remain143/165registers,128threads,384CTAs, unchanged arithmetic and grid. Theoretical occupancy rises from8to12active warps per SM:3resident128-thread blocks, bounded by registers. K4 shared memory alone permits4; K5 permits3.

Achieved occupancy remains about15% because the384-block grid only averages2.04blocks across188SMs. Kernel profile durations were slightly worse: see SUMMARY.json for exact baseline/candidate values. No serving run or adoption justified for carveout alone. NCU replay timing is not paired full-component or serving timing. Finite output and matching printed norms are smoke evidence only, not exact array equality.

The installed CUTLASS runtime accepts a launch-level carveout argument and translates it into a function attribute; this overrides the default inferred from min_blocks_per_mp=2. Context-wide cache preference was previously verified ineffective. Source is a pinned one-line change to p8_small_m.py, read-only mounted in isolated containers with B12X disk-cache loading disabled and fresh CuTe/CUDA caches.

Next bounded candidate: combine FC2-only explicit carveout with564CTAs on188SMs for direct decode. Keep all grouped-prefill launches unchanged. Requires representative exact eager/changed-input graph component checks and paired full TP-local MoE timing; only a promising result proceeds to32Kprefill and8KC1/C4 serving. Increased residency is capacity, not a speed promise; reduced L1 and weight-memory traffic are countervailing effects. No arithmetic, FP8 MMA, checkpoint, MTP or power changes.

Production and clients remain stopped; goals not achieved. Plans, source, raw NCU reports, receipt logs, text exports and hardware snapshot retained here.
