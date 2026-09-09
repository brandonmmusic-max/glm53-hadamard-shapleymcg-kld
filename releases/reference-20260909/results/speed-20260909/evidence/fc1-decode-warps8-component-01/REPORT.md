# Eight-warp direct FC1 screen

172 representative exact comparisons passed for K4/K5 real rank0 layers, eager and changed-input CUDA graphs. Compiled facts prove8warps for direct decode versus4baseline; grouped prefill is unchanged and shares its original cached kernel.

Paired full TP-local MoE graph timing changes are between-0.58% and+1.90% across K4/K5 M4/M16. No consistent gain. Do not advance to serving based on this result.

Candidate direct FC1 uses128registers/thread,256threads/CTA,23552(K4)/27648(K5)dynamic shared bytes and zero stack/local memory. CUDA API reports2blocks/SM at256threads, versus2blocks at128threads for baseline173/183registers. Thus resident-warp capacity increases, but no component timing gain was established. Occupancy alone does not prove speed.

M32 grouped candidate also rejected:6-7% slower, still register-limited to2blocks. Retain procedural MCG and original4warp decode. M64 rowalias376 remains the measured exploratory prefill candidate:8197prefill188.3C1/310.8aggregateC4, no adoption or full-model quality claim. Production remains stopped.
