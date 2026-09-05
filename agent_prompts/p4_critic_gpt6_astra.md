# GPT-6 Astra assignment: adversarial P4 ISA/runtime audit

Act as an independent critic in the supplied isolated Git worktree. Commit a
focused audit to the current branch and do not push. Do not implement the
kernel; examine the current repository and write
`docs/P4_RUNTIME_AUDIT.md` plus a compact machine-readable receipt under
`evidence/opened/codec-v2/p4-astra/`.

Audit the proposed product contract:

K4 procedural-MCG sliding-window stream -> per-element prologue decode ->
packed E2M1 nibbles + physical E4M3 residual scales per 16 weights ->
Blackwell `mxf4nvf4` `m16n8k64`, with no dense BF16/FP16 weight
materialization or weight matmul.

Build an evidence matrix that distinguishes source inspection, CPU
bit-closure, GPU arithmetic closure, integrated serving, KLD, determinism,
prefill speed, and decode speed. Identify every condition that must be true for
the phrases "native Tensor Core P4" and "NVFP4 speed class" to be defensible.
Pay particular attention to:

- exact E2M1 and E4M3/16 operand layouts and scale semantics;
- global/shared/register staging, descriptor types, lane permutations, and
  `mxf4nvf4` instruction selection;
- whether the prologue is actually hidden under MMA rather than serialized;
- table size, register pressure, shared-memory use, occupancy, issue count,
  and bytes moved;
- FC1 and FC2 path completeness, routed reduction, CUDA graph parity, TP4,
  and 5-run determinism;
- bit-exact comparison against a separately implemented reference;
- attribution/licensing for ExLlamaV3, KQuant, QSRT, and `w4a8_trellis`.

Predeclare falsifiable device tests and stop conditions. Do not run GPUs,
access selection/confirmation/final roles, inspect the 28 confirmation logits,
download models, modify services, or use LDLQ/BlockLDLQ. Label unknowns and
untested claims explicitly. Run any safe static/unit checks that help the
audit, inspect the diff, and commit.
