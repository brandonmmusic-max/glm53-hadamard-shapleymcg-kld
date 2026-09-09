# Separate loader warp without register directives

172 representative exact comparisons passed on rank0 K4/K5 real checkpoint layers, including changed-input CUDA graphs. Original procedural reconstruction, FP8 MMA accumulation order, Hadamard/scaling/quantization boundaries, decode path and phase0/FC2 are unchanged.

The32-thread loader stages next-buffer A/B/scales while128compute threads consume the current buffer. A160participant named barrier joins eachK64 slice. Four further joins protect pipeline drain, projection publication, transformed-row publication and quantizer completion. This is representative correctness evidence, not a full sanitizer/model-quality qualification.

Removing explicit register budgets eliminated the1232byte stack reported by the prior K4 variant. Compiled grouped K4:250registers/thread,35840dynamic shared bytes,160threads,stack0/local0. K5:242registers,39936bytes,160threads,stack0/local0. CUDA API reports1block/SM for both. Register footprint prevents2blocks.

Paired M4096 local-MoE graph time improved10.67%/10.81% K4 and11.57%/11.40% K5 versus the original component reference. The earlier M64 rowalias376 candidate improved about16% in its separate component run, so this version does not displace the retained candidate. No serving window or wider sweep is scheduled for it. Retain the source and results for future architecture work; stop retuning this prototype now.

The retained candidate's narrow serving screen remains8197prefill188.3C1/310.8aggregateC4, not a qualified production gain. Next focus: whole-model graph dependencies and collective wait/overlap evidence, because individual FC1 improvements have yielded only modest end-to-end changes. Production remains stopped.
