# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — pre-registration (committed before any candidate is generated)

Date: 2026-09-02 (America/New_York). Hardware: 4x RTX PRO 6000 Blackwell (SM120a, 96 GB). Plan: BMXFP4_plan_v2.md (uploaded by Brandon).

## Model, data, harness
- Teacher: Qwen/Qwen3-30B-A3B @ 4c446470ba0aec43e22ac1128f9ffd915f338ba3 (weights byte-identical to main), BF16, HF transformers, sdpa attention, fp32 logits stored per window.
- Sealed corpus roles: the prior ShapleyMCG Qwen seal (brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility, controls/fixed-hadamard-k34-v1/calibration/qwen-sealed-corpus.json; windows of 2048 tokens; fit 32 / selection 16 / confirmation 16 / final 25; four domains). Token ids are reused directly (vocab/merges identical between the Base and post-trained tokenizers; 88/89 windows round-trip identically, the 89th differs by a merge boundary only and is used as token ids).
- Control panel: the prior WikiText-2 panel (10 x 2048 tokens, causal-arm-v3/turboderp-wiki2-sdpa-teacher/input-token-ids.npz).
- Metric: per-token KL(teacher || student), full vocabulary, positions 0..2046; per-window mean; mean over windows (= token mean at equal window length); p90/p99 per token; top-1 agreement. Paired bootstrap 95% CI over windows (20,000 draws) for arm differences.
- Roles: fit -> Hessians, GPTQ, rotation learning, permutation; selection -> every hyperparameter and the rotation verdict; confirmation -> allocation re-anchor; final -> scored exactly once per reported arm, after all selection decisions are frozen.

## Budgets (exact routed-expert payload bytes, computed from the reference safetensors; filled in at Stage 0)
- E1b: turboderp/Qwen3-30B-A3B-exl3 @ 5.0bpw -> routed-expert payload bytes = [TBD]; total bytes = [TBD]. PRIMARY.
- E1a: turboderp/Qwen3-30B-A3B-exl3 @ 4.0bpw -> routed-expert payload bytes = [TBD]; total bytes = [TBD]. Stress test (forces the sparse tier).
- Backbone (attention, router, shared expert, embeddings, lm_head, norms) stays BF16 in every BMXFP4 arm; the asymmetry vs EXL3 (which quantizes attention) is stated; a total-bytes row (B7t, FP8 backbone) is secondary.

## Rows (all W4A16 unless stated; NVFP4-family rows also scored W4A4 = activations NVFP4 per 16 at expert inputs)
B0 teacher instrument (bitwise repeat); B3 RTN NVFP4 uniform T1; B3' GPTQ + MSE scales + 4-vs-6 + static act-order, no rotation, uniform T1 (CONTROL); B4 B3' + random SO(16) block rotation, 5 seeds; B5 B3' + fixed Hadamard-16; B6 B3' + learned Cayley R16 (init I and Had16), per-layer-per-projection sharing; B6p B6 + Hessian-driven channel permutation; B7 allocation at E1b bytes over {T0,T1,T2,T3,T4} using calibrated values; B7a same at E1a bytes; B7t B7 at equal total bytes with FP8 backbone; B8 B7 with the uncalibrated proxy (raw Hessian loss); E1a/E1b turboderp EXL3 reconstructed to BF16 and scored on the same panels.

## Primary comparison and win criterion
B7 (W4A16) vs E1b on the final role at equal routed-expert payload bytes: mean KLD, paired-bootstrap 95% CI of the difference excluding zero. Expected direction (stated in advance): E1b wins the primary (trellis space-filling gain ~0.2-0.3 bit/dim); the question is the size of the gap and whether the rotation and multi-tier arms close any of it.

## Secondaries (pre-registered)
B7a vs E1a; B7 (W4A4) vs E1b with the activation delta stated; B6 vs B3' and B5 vs B3' on selection (rotation verdict); B7 vs B8 (allocation-value verdict); B7 vs B7t.

## Stop rules
1. If B5-B3' and B6-B3' are both inside the CI on selection: rotation is dead for NVFP4 on this model; B7 uses B3' candidates.
2. If B7 and B8 are inside the CI: the calibrated value adds nothing over the proxy at this scale.
3. If the allocator at E1b puts >=95% of unit-bytes in T1: multi-tier is a negative result; kernel work does not proceed.

## Declared deviations from the plan (for speed; each is reported as such)
- Allocation value: the plan's full Aumann-Shapley path-integrated attribution is replaced, for this overnight pilot, by a calibrated proxy: per-unit GPTQ Hessian loss scaled by a per-layer factor measured on the confirmation role (KLD increase per unit of Hessian loss when only that layer's experts are quantized to T1). B8 is the same allocator with the unscaled proxy. If the repository's causal allocator can be run in the time available it will be added as a further row, not substituted.
- Learned rotation objective: Hessian-weighted per-projection reconstruction through the NVFP4 quantizer with MSE scales (RTN inside the optimizer; GPTQ applied after R is frozen), rather than the full expert-function objective.
- External NVFP4 anchors (RedHatAI, nvidia) are downloaded but scored only if time permits; they validate the harness and are not part of the primary.
