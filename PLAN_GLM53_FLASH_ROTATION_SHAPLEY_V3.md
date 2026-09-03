# GLM-5.3-Flash in-block rotation and 6-bpw Shapley campaign, V3

Date: 2026-09-03. Execution host: Brandon's Pop!_OS workstation with four RTX PRO 6000 Blackwell 96 GB GPUs.

## Question and claim boundary

The first decision is whether a runtime-correct block-16 orthogonal transform of routed MoE inputs and the matching inverse basis in the BF16 expert gate/up weights produces a complete ModelOpt NVFP4 checkpoint with lower end-to-end teacher KLD than stock NVFP4. Both fixed normalized Hadamard-16 and learned orthogonal rotations must be tested. The second decision, made only after freezing the winning rotation recipe, is whether a globally allocated 6.0-bpw routed-expert model improves on the uniform rotated candidate.

The prior V2 result does not answer either question: it calibrated identity-basis GPTQ only and did not use a routed runtime rotation. V2 selection rows are permanently opened and excluded from every V3 role.

Rotation claims are selection-level evidence. The final 6-bpw candidate is evaluated once on V3 confirmation and may support a confirmation-level claim. There is no fresh final holdout in this campaign, so no sealed-qualification claim is permitted.

## Immutable inputs

- BF16 source: `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43`.
- Teacher logits: `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`.
- Teacher panel source revision: `7c378d5f17dba158c4c803eff27c346dd0615660`.
- Stock NVFP4 control/carrier: `/home/brandonmusic/models/GLM-5.3-Flash-NVFP4`, fingerprinted in the experiment record.
- Runtime image: `klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate@sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe`.
- Calibration text: `/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl`, SHA-256 `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`.
- V3 role manifest: `/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-v3.json`.
- KLD implementation: `/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval`, commit `565aca8`, metric-core SHA-256 `752af6740595791f300cf47f15ef81f8ca85245d5282fc600ef26ee57eae41e6`.

## Roles and access

V3 deterministically reuses 64 old V2 fit windows solely for activation/Hessian calibration. Those windows have never been evaluated against teacher logits. It creates fresh, domain-balanced roles from windows not opened by V2:

- fit: 64 windows, 16 per domain; Hessians and rotation fitting only;
- conditional-fit: 16 windows, 4 per domain; runtime wiring, learner settings, and causal diagnostics;
- selection: 48 windows split before execution into three balanced waves of 16; recipe choice and adaptive escalation only;
- confirmation: 28 windows, 7 per domain; opened exactly once for the final frozen 6-bpw candidate and its declared controls;
- final: empty.

Candidate optimization may read fit artifacts and conditional-fit metrics. Each selection wave is opened only after the candidate set for that wave and the analysis code are hashed. Confirmation logits are not evaluated or summarized until the rotation recipe, Shapley estimator, 6-bpw allocation, packed candidate, and declared control list are frozen.

## Rotation ABI and candidates

The transform is inside every 16-channel block of the 4096-wide routed expert input. With row-vector activations, runtime computes `x' = x R`, while BF16 gate/up weights are transformed as `W' = W R`. This preserves the BF16 linear function because the kernel uses `x W^T`. Router logits, shared experts, attention, and down projections remain in the carrier basis. The pinned vLLM runner's `routed_input_transform` contract applies the transform only to routed experts while retaining the original hidden state for the shared expert.

Wave 1 candidates use the same group-16 ModelOpt NVFP4 packer, calibration windows, gate/up scope, carrier, and runtime:

1. stock NVFP4 control;
2. identity-basis calibrated GPTQ control;
3. fixed normalized Hadamard-16, shared across blocks and layers;
4. learned orthogonal 16x16 rotation per MoE layer, initialized and fit from both identity and Hadamard starts, with the lower conditional-fit objective frozen.

Down projections are held byte-identical across the three calibrated controls in Wave 1. A one-layer pilot proves packing, runtime activation transformation, and coherent generation but is not evidence for the model-level decision.

If neither fixed nor learned Wave 1 rotation beats stock, later selection waves are predeclared for (a) per-layer/per-block learned rotations and signed/permuted Hadamard variants, then (b) a gate/up plus down-output rotation that is enabled only if its inverse residual transform is implemented and BF16-invariance-tested. Failed rows remain visible; no candidate is silently replaced.

## Rotation decision rule

On a selection wave, `delta = KLD(candidate) - KLD(stock)` is paired by window. A candidate beats stock only if its mean delta is negative and the paired 20,000-resample bootstrap 95% upper confidence bound is below zero. All tokens and all windows count; non-finite or missing values fail the row. If multiple candidates pass, the lowest mean KLD wins. An adaptive wave may refine only recipes declared above, and its new candidate set must be frozen before opening that wave.

## Six-bpw Shapley allocation

After the rotation recipe is frozen, each routed-expert allocation unit is a `(layer, expert, projection)` group with immutable tensor membership. The precision ladder and its exact packed byte cost are frozen before attribution. The initial ladder is `{NVFP4, MXFP6, FP8, BF16}` with no sparse tier; unsupported formats may be replaced only before selection-wave scoring, with the deviation recorded.

Values are teacher-anchored marginal reductions in end-to-end KLD, not raw Hessian loss. Shapley Monte Carlo uses deterministic seeded permutations, includes full-model and routed-expert baseline terms, reports per-unit standard errors and convergence, and preserves every causal evaluation. The allocator selects upgrades globally under an exact routed-expert payload budget of 6.0 weighted bits per BF16 scalar, including all scale/metadata bytes specified by the chosen packed ABI. The realized assignment is independently byte-audited; exceeding the budget is a failure, and material underspend must be explained.

The frozen 6-bpw candidate is compared on confirmation against stock NVFP4, the uniform winning-rotation NVFP4 candidate, and a uniform 6-bpw control if the same runtime supports it. Primary 6-bpw success requires lower mean KLD than uniform rotated NVFP4 with paired 95% upper confidence bound below zero. Stock comparison is secondary. Runtime throughput is measured only after KLD and correctness pass and is reported by exact regime, never aggregated across speculative and non-speculative runs.

## Gates and stopping

1. Seal the experiment record before target results.
2. Verify source identities, role/hash disjointness, free storage, inactive-service restoration target, and runtime image digest.
3. Pass CPU algebra, orthogonality, Hessian congruence, pack/unpack, and candidate-index tests.
4. Pass a runtime-correct one-layer pilot with the patch-active log marker and a coherent deterministic canary.
5. Freeze and hash every complete candidate before its selection wave.
6. Continue through the declared rotation ladder until a candidate meets the statistical win rule or all three waves are exhausted. Exhaustion is a bounded negative result, not a win.
7. Freeze the Shapley method, ladder, allocation, exact byte receipt, complete checkpoint, and analysis before confirmation.
8. Open confirmation once; preserve and report every row, exclusion, failure, and deviation.
9. Restore the pre-run stopped service state. Upload and publication require separate authorization.
