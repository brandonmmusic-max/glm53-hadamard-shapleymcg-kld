# GPT-6 Astra assignment: true P4 fused Tensor Core kernel

You are the implementation owner for the true P4 endpoint in the private
`bmxfp4-glm53` experiment repository. Work only in the supplied isolated Git
worktree and commit your changes to its current branch. Do not push.

## Product contract

Implement the speed product, not another P8 approximation:

- Input weights are a K4 procedural-MCG trellis stream using a sliding 16-bit
  state window and the repository's frozen ExLlamaV3-compatible constants.
- Decode per element in the MoE kernel prologue directly into packed E2M1
  nibbles plus the physical E4M3 residual scale plane at one scale per 16
  weights.
- Feed those operands to the Blackwell `mxf4nvf4` `m16n8k64` Tensor Core path.
- Do not materialize a dense BF16/FP16 weight tensor and do not use a
  BF16/FP16 weight matmul.
- The decode must be bit-exact against an independent CPU reference and be
  positioned so it can overlap the MMA pipeline. A fused kernel path is not a
  claim of one machine instruction.
- Keep lookup tables at or below 4 KiB. Prefer the procedural MCG law.
- Do not implement or use LDLQ or BlockLDLQ.

## Evidence and scope

Start by reading `README.md`, `glm53_nvfp4/trellis_nvfp4.py`,
`runtime_patch/p8_native_kernel.py`, and the P8/W4A8 implementation in
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py`. Preserve the
existing P8 endpoint.

Add a separately named P4 runtime endpoint, its launch/probe seam, and focused
tests that establish:

1. sliding-window MCG state decoding and nibble packing;
2. E4M3/16 residual-scale geometry and byte addressing;
3. the exact `mxf4nvf4` operand and MMA selection;
4. absence of a dense BF16/FP16 weight path in the P4 endpoint;
5. deterministic output hashes at the structural/reference level.

Do not run GPUs, open selection/confirmation/final roles, inspect the 28
reserved confirmation logits, download models, or launch/modify production
services. Static and CPU/unit validation only; the parent will run the real
device closure after review. Preserve ExLlamaV3, KQuant, QSRT, and vendored
`w4a8_trellis` attribution and identify what is ported.

Write a short receipt under `evidence/opened/codec-v2/p4-astra/` that states
exactly what is demonstrated, not tested, and still blocks the NVFP4 speed
claim. Run the relevant tests, preserve their output in the receipt, inspect
your diff, and commit with a focused message.
