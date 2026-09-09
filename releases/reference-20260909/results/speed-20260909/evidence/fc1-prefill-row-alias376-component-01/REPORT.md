# Grouped row-local scratch plus376 FC1 CTAs

172 representative exact comparisons passed: rank0 K4/K5 real checkpoint layers, M4/M16/M4096, eager and changed-input CUDA graphs. Preparation/FC2/decode grids and original reconstruction/MMA math are unchanged; grouped FC1 scratch layout and its grid are the only functional changes.

CUDA occupancy API reports2 resident blocks/SM for candidate versus1 original, both default and maximum shared carveout. Dynamic shared allocation is35840 bytes K4/39936 K5 versus65536; registers236/238 versus236/236. No stack/local memory reported. API capacity is not measured achieved residency.

Paired full TP-local MoE graph times atM4096 are15.94%/15.97% lower for K4 and15.81%/15.79% lower for K5 across two captures, five alternating-order samples each. These are fixed synthetic routes and within-process samples, not independent server runs or whole-model throughput. Both baseline and candidate were GPU-compiled and checked; extracted cubin resources and raw comparisons are retained.

This warrants only the user-requested narrow serving screen:32K cold prefill,8K C1/C4 speculative decode cap8192, one pass. Separate image/plan at../fc1-row-alias376-speed-quick01. No full suite, adoption, or production restart is authorized by these component results alone. Model quality still needs matched qualification before adoption.
