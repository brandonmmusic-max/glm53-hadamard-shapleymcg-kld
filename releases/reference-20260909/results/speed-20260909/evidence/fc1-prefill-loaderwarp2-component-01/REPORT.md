# Loader warp prototype with explicit register budgets

Attempt1 failed CuTeDSL compilation because an unreachable diagnostic early-return branch was nested inside a runtime consumer branch; preserved at../fc1-prefill-loaderwarp-component-01. Attempt2 removes only that unused diagnostic path in this separate grouped-only class.

Attempt2 passed172 representative K4/K5 component comparisons including changed-input CUDA graphs. Its32-thread loader overlaps next-tile copies with128compute threads reading the other buffer. Matching160participant named barriers protect initial publication, per-slice completion, pipeline drain, epilogue publication and final scratch readers. Original arithmetic and decode remain unchanged.

Performance regressed substantially: K4 local MoE graph approximately3.3x baseline, K5 approximately2.75x. The K4 FC1 resource dump reports240registers and1232byte stack. No serving test or adoption. This is not evidence that all warp-specialized kernels are slow.

One diagnostic retry removes the explicit producer64/consumer240 register directives while preserving all scheduling and arithmetic, to distinguish register-budget-induced spilling. No production restart.
