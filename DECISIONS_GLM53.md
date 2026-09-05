# GLM-5.3-Flash NVFP4 V2 decisions

## 1. 2026-09-02: start and evidence level

Before model downloads or adaptive measurements, execute the uniform identity-GPTQ NVFP4 candidate as a confirmation-level sealed experiment. Treat the existing 25-window panel as opened legacy evidence. Do not claim a new final result without a newly captured post-freeze final panel.

## 2. 2026-09-02: corrected immutable sources

Use the 120-shard BF16 model `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` and teacher dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. The official FP8 checkpoint and full 78-layer GLM teacher are invalid substitutes.

## 3. 2026-09-02: implementation architecture

Use a layer-streamed BF16 loader and a copy-on-write stock NVFP4 carrier. Require a one-layer pack/load/KLD pilot before the 42-layer campaign. This avoids the impossible full-model residency assumption and makes every output layer independently receipted.

## 4. 2026-09-03: capture transport and checkpoint ABI correction

Use the teacher dataset's pinned BF16 hidden-state and router captures as a streamed calibration transport, selecting only the sealed fit-role rows. The resulting claim is capture-calibrated group-16 GPTQ, not causal propagation through a locally resident BF16 model. Store ModelOpt block scales in logical row-major checkpoint order; the 128-by-4 swizzle is performed after load for kernel ingestion. Retain the failed pre-pilot swizzled reconstruction as negative-control evidence.

## 5. 2026-09-03: post-quality benchmark implementation

Pin `local-inference-lab/llm-inference-bench` at commit `d115feee75095081bda2520aa046986a8885f449`. Only after the registered KLD confirmation passes, measure target-only decode, Estonia, and LAVD as three distinct candidate-only regimes. Do not treat their throughput or task scores as part of the primary KLD estimand.

## 6. 2026-09-03 14:24 EDT: restore the Qwen rotation recipe before endpoint KLD

The gate/up-only H16 result is a protocol-deviant diagnostic, not a verdict on the Qwen recipe. The Qwen file of record used independent block-16 transforms for the gate/up input (`R_in`) and the post-SwiGLU down-projection input (`R_mid`), concatenated gate/up rows, a full per-expert Hessian, static act-order within fixed 16-column groups, and sequential GPTQ error propagation across 128-column slabs. Port those elements to the packed GLM path and compare exact identity and H16 at layer 44 before scaling. Use Humming, disable FlashInfer autotuning, and set `NCCL_P2P_LEVEL=SYS`; do not use a CUTLASS MoE backend. Recalibrate gate/up and down activation scales in their corresponding bases. The prior Qwen endpoint was expanded-BF16 pseudo-quant, so the deployable GLM adaptation must constrain every scale to the actual ModelOpt ABI: one FP32 global scale per checkpoint tensor family and E4M3 per-16 block scales. Record this representation constraint rather than silently emulating an unstoreable scale.

## 7. 2026-09-03 14:52 EDT: early-layer follow-up after exact layer-44 H16 failure

The corrected all-projection layer-44 H16 candidate failed conditional-fit against exact identity-GPTQ (mean KLD 0.0726416 versus 0.0382295; paired delta +0.0344121, 95% BCa CI [+0.0248263, +0.0456455]). Stock NVFP4 under the same Humming endpoint measured 0.0388080; identity-GPTQ was directionally better by 0.0005785 but its CI crossed zero. This establishes that the transform ran correctly and that fixed H16 is harmful at layer 44, not that the Qwen recipe is globally ineffective. The Qwen evidence selected rotation primarily in early layers and usually selected identity in late layers. Therefore use layer 3 as the next conditional-fit pilot, preserving the same packed W4A4 ABI, full-Hessian GPTQ, independent `R_in`/`R_mid`, and runtime proof. Fit both identity- and H16-initialized Cayley rotations. Keep the original per-projection Hessian objective as the exact Qwen replication and the later full-expert objective as a separately labeled diagnostic; do not choose between them using protected selection or confirmation data.

## 8. 2026-09-03 16:03 EDT: transform invariance gate before further optimization

The layer-3 family factorial shows H16 is harmful at the packed W4A4 endpoint both for gate/up alone (candidate 0.1415098 versus identity 0.0382913; paired BCa delta CI [+0.0809631, +0.1367601]) and for down alone (candidate 0.3041527).  This is inconsistent with the large improvement in the down-projection packed Hessian surrogate and therefore requires an implementation-invariance gate before interpreting or optimizing another rotation.  Freeze a deterministic within-16 signed permutation with determinant +1, quantize it through the identical full-Hessian packed path, and test it first on gate/up and then on down if needed.  The transform preserves group membership, E2M1 magnitudes, E4M3 scale eligibility, and activation maxima, so a material KLD difference from exact identity is evidence of a lane/layout/packing/runtime defect.  A matched result clears those mechanics and makes H16's endpoint loss credible.  Per-block Cayley matrices remain an exploratory extension, not the Qwen reproduction: the Qwen learner creates one shared 16-by-16 `R_in` for concatenated gate/up and one shared `R_mid` for down per layer.

## 9. 2026-09-03 16:50 EDT: localize the failed exact-packed invariance gate

The exact-packed signed-permutation candidate failed invariance even after moving `R_in` immediately inside Humming before `w13` input quantization.  The stronger sign-only control, `R_in = -I` with all layer-3 gate/up packed values negated and all scale/down tensors byte-identical to identity, also failed (mean conditional-fit KLD 0.8231896 versus identity 0.0382913).  This invalidates the earlier item-7 statement that endpoint logging alone established a correct transform and suspends interpretation of every rotated endpoint KLD result.  Before any further H16 or learned search, run a paired implementation canary from the identity checkpoint: negate the already-loaded `w13_weight` parameter immediately before Humming conversion and pair it with the same inner `-I` input transform.  If runtime-loaded negation restores identity, the copy-on-write checkpoint/load mapping is defective; if it does not, the defect is in sign-code interpretation, Humming conversion, input quantization, or fused GEMM consumption.  This is diagnostic evidence only and must not be used for candidate selection.

## 10. 2026-09-03 17:31 EDT: indexed-loader correction and exact Qwen layer-3 verdict

The runtime-loaded `-I` canary restored identity-level KLD, and source inspection then identified the dominant defect: vLLM's ordinary `safetensors` iterator loads every tensor physically present in each selected carrier shard.  A copy-on-write overlay index can redirect a tensor to a candidate chunk, but a later carrier shard still contains and overwrites the same key.  All earlier sparse-overlay rotation runs were therefore one-sided: the activation rotation executed, while the effective expert weights were stock.  Candidate overlays now declare `candidate-overlay.v2` and require vLLM's `instanttensor` loader, whose indexed iterator loads only the file selected for each key.  `run_kld_v3.sh` fails closed if an overlay is paired with any other load format.  With that correction, a disk `-I` overlay plus inner `-I` activation transform measured 0.0377145 versus identity 0.0382913, paired BCa 95% CI [-0.0017683, +0.0008610], clearing invariance.  The raw signed-permutation builder also now preserves the FP4 signed-zero nibble instead of round-tripping through float.

The corrected layer-3 Qwen-style arm transforms every fixed 16-column group of gate, up, and down; uses one shared `R_in` for concatenated gate/up and a separate `R_mid` for down; rotates the full Hessians as `R^T H R`; preserves within-group static act order; and propagates GPTQ error across 128-column slabs.  A direct differential check against the Qwen reference `rotation.py` produced bitwise-identical H16 and zero maximum error for weight, activation, and Hessian transforms.  The only deliberate endpoint adaptation is required by deployment: Qwen's in-process pseudo-quant scorer rotates the dequantized weights back to dense BF16, whereas the ModelOpt checkpoint keeps `Q(W R)` packed and applies the cancelling activation transform immediately inside Humming.  Inverse-rotating the quantized weights would destroy exact NVFP4 representability.

On the same 16 conditional-fit windows, corrected H16 at layer 3 measured 0.0375717 versus a fresh stock run through the same `instanttensor` loader at 0.0385847: mean delta -0.00101298 (-2.63%), paired BCa 95% CI [-0.00171749, -0.00017658], pass.  The earlier stock anchor at 0.0388080 gives the same decision but is retained only as a cross-loader diagnostic.  Corrected identity-GPTQ measured 0.0390069, so H16 was directionally 3.68% better but that paired interval crossed zero.  The independently calibrated learned `R_in`/`R_mid` candidate measured 0.0385788 and was significantly worse than H16 (paired BCa delta for learned minus H16 [+0.00013188, +0.00196291]); it did not pass versus stock or identity.  Freeze H16 as the scale-up basis.  The pinned Humming build's native `float4e2m1` activation-input path is explicitly disabled/not fully tested upstream and failed aligned execution locally, so these results retain the Qwen-comparable packed-W4/BF16-activation endpoint and do not claim W4A4.

## 11. 2026-09-03 18:02 EDT: exact 256-sample Qwen replication scale-up gate

The item-10 layer-3 arm used up to 10,000 routed samples per expert, which improves the calibration estimate but is not the original Qwen experiment's 256-sample cap.  Repeating the otherwise identical H16 arm with exactly that 256-per-expert cap produced mean conditional-fit KLD 0.0377024 versus same-loader stock 0.0385847: mean delta -0.00088229 (-2.29%), with paired BCa 95% CI [-0.00206005, +0.00058266].  It is directionally better but does not independently pass the registered confidence rule at one layer.  Against the 10,000-sample H16 arm its delta was +0.00013069 with BCa CI [-0.00073320, +0.00109276], so the two calibration caps are not distinguishable on this panel.  Scale the exact 256-per-expert Qwen recipe across all 42 routed layers before deciding it; retain the 10,000-sample result as a labeled calibration-size diagnostic, not as a substitute for the requested reproduction.

## 12. 2026-09-04: window variability becomes a development control

The conditional-fit window confidence interval is no longer a blocking gate for engineering iteration on P8/P4 encoders, scale contracts, device closure, or kernel specialization. Development candidates may advance on a preregistered directional mean, paired window diagnostics, causal expert-level NMSE, exact rate, and arithmetic closure even when a window-level interval crosses zero. Every such result must remain labeled adaptive/developmental, preserve every window and domain without exclusions or rerolls, and report the interval and detectable-effect limit as controls.

This relaxation does not change the scientific claim gate. A statement that a codec beats stock or GPTQ NVFP4 still requires the matched, paired end-to-end teacher-KLD estimand on an eligible role with the BCa upper bound below zero. Later Shapley allocation may select where to spend bits, but it cannot retroactively qualify an inferior or unresolved base arm. The failed W6A8 activation result likewise remains a negative control rather than a blocker on pseudoquant P8 engineering; P8 runtime qualification still requires kernel-versus-pseudoquant KLD closure.

## 13. 2026-09-04: physical P8 scale contract and real-payload device closure

Use the actual UE8M0/32 scale grid in the P8 `mxf8f6f4` MMA. The directive's identity-SFB interpretation was tested directly and regressed geometric-mean routed-output NMSE by 16.6812% versus the physical native-SFB representation, with zero of 16 expert/projection cells nonregressing and a paired BCa mean-log-ratio interval `[+0.12061, +0.19232]`. This is a measured, narrow deviation from the directive: P8 remains a K4 procedural-MCG stream decoded to E4M3, but its per-32 physical scale is consumed by the tensor core instead of being folded into the E4M3 value and replaced by SFB 127.

The corrected monolithic prologue closes on the actual layer-3 K4 artifact for experts 0:8 at full GLM dimensions. Against the exact dense decode of the same bytes it measured cosine `0.9989534`, relative L2 `0.0457535`, and norm ratio `0.9978559`; every registered check passed. Two earlier identity-boundary device failures were implementation diagnostics, not codec results: the monolithic epilogue skipped both H128 and the ordinary gated activation, leaving a raw gate, and a later diagnostic incorrectly removed the mandatory within-K32 MMA lane permutation. Both errors are fixed and the failures remain preserved.

The identity-boundary split-prefill failure was traced to a second input-materialization branch that skipped the mandatory within-K32 MMA lane permutation. After restoring it, the fresh frozen v2 run passed K3/K4 at both M33 and M64: cosine `0.9988962`–`0.9989411`, relative L2 `0.0460121`–`0.0470563`. A separately frozen run using the actual 4.25-bpw layer-3 K4 artifact at full GLM dimensions also passed at both token counts: cosine `0.9989396`–`0.9989480`, relative L2 `0.0459035`–`0.0460787`. Split and monolithic device arithmetic are therefore closed; the preserved v1 failure is an implementation diagnostic, not codec evidence. Integrated end-to-end kernel KLD, speed, and five-run determinism remain unqualified. The existing adaptive identity/H128 policy result remains directional evidence only: mean delta KLD `-0.0005551` (`1.04%`) over 32 conditional-fit windows, BCa interval `[-0.0024023, +0.0011452]`.

## 14. 2026-09-04: physical P8 kernel closes end-to-end within the relaxed engineering margin

The physical K4 identity-boundary stream was sharded without re-encoding into four TP4 sidecars and served at layer 3 by a fused procedural-MCG decode inside `mxf8f6f4`, with physical UE8M0/32 scales and E4M3 activations. Against the exact decoded-BF16 pseudoquant carrier on the same opened domain-balanced n=32 conditional-fit role, the kernel measured mean KLD `0.0539830851` versus `0.0535879281`: paired mean delta `+0.0003951571` (`+0.7374%`), BCa 95% interval `[-0.0018896, +0.0029882]`, and 13/32 window wins. This passes the preregistered decision-12 engineering closure margin of absolute mean delta at most `0.0014` nats. Every window was retained; confirmation logits remain unopened; LDLQ was not used.

This is the first end-to-end teacher-KLD measurement of the physical trellis-to-tensor-core path, but it is not a codec quality win. The interval is wide, general-domain mean delta is `+0.0038225`, and code/agentic mean delta is `-0.0035662`. The initial E=288 small-M specialization emitted non-finite values in a preserved diagnostic, so this KLD run deliberately used the already-closed materialized dynamic arm for both prefill and decode correctness. Native decode speed, five-run bitwise determinism, P4, and a strict comparison against matched NVFP4 remain separate gates.

A post-closure diagnostic paired the same physical-kernel run with the already completed decoded-GPTQ NVFP4 control on the identical 32 windows. Kernel mean KLD `0.0539831` versus control `0.0533314` gives delta `+0.0006517` (a `1.22%` regression), BCa 95% interval `[-0.0025401, +0.0059822]`. This is a null rather than a demonstrated loss, but it is not a quality pass and cannot be repaired statistically by later Shapley allocation.

## 15. 2026-09-04: deterministic P8 closure and relaxed joint-policy verdict

The E=288 small-token failure was not caused by the resident cluster count. Frozen arms at 4, 64, and 188 clusters produced the identical non-finite output hash. Source/ABI tracing found that the monolithic scaled-trellis branch reads the logical `[E,N,K/32]` UE8M0 planes, while the wrapper supplied a one-byte sentinel there and supplied only the separately repacked split-kernel planes. Retaining both representations fixed the defect: the monolithic E=288 arm passed M3/M33 with cosine `0.9995787`/`0.9996932` and relative L2 `0.0291036`/`0.0247810`.

Replacing unordered atomic output scatter with one BF16 row per routed expert plus the fixed-order B12X top-k reduction passed five-run bitwise determinism for both monolithic and split-materialized M3/M33. The promoted serving selector uses monolithic M16 below 36 routed rows per expert and split M64 above it. Its n=32 deterministic kernel-versus-decoded-codec KLD closure passed the relaxed engineering control narrowly: mean delta `+0.00137716`, BCa `[-0.00055423,+0.00397212]`. Against matched decoded GPTQ NVFP4 the point estimate was `3.063%` worse, with BCa `[-0.00091783,+0.00744612]`; this is a null/fail quality result, not a loss claim.

Decision 12 was then applied to the strongest previously blocked codec policy. Its frozen three-state assignment (104 identity, 94 full-H128, 90 FC1-H128/identity-down) had improved held-out joint routed-output NMSE by `24.049%` versus GPTQ with 6/8 window wins, missing only the old CI gate. On the already opened, domain-balanced n=32 KLD role it improved mean KLD by only `0.8104%` (`0.0528992` versus `0.0533314`), 16/32 wins, BCa `[-0.00245954,+0.00126947]`. It therefore fails the preregistered 5% development-promotion threshold and the strict claim gate. Relaxing the local window gate was justified as a diagnostic, but the result proves routed-output NMSE cannot be assumed to be recovered by later Shapley allocation; the next allocator must be KLD-aware.

## 16. 2026-09-05: replace local-error allocation with direct-KLD Shapley

The relaxed three-state policy transferred only `0.8104%` of its `24.049%` routed-output NMSE improvement to end-to-end KLD. Therefore routed-output and weight NMSE are removed as Shapley value functions. The new allocator values a whole routed layer only by its contextual reduction in equal-window teacher KLD. Its native no-FP6 6-bpw ladder uses K4 procedural-MCG P8 at `4.25` bpw and scalar MXFP8 E4M3/UE8M0 at `8.25` bpw; 18 MXFP8 and 24 P8 layers realize `5.9642857` bpw before boundary metadata, leaving `0.0357143` bpw of headroom. Both tiers use `mxf8f6f4`, which has twice NVFP4's MMA issue count. Exact physical bytes including boundaries remain a final gate.

Before building all 84 full-model coalition endpoints, run the frozen four-layer, eight-coalition matched-BF16 interaction pilot over layers 3/19/20/22. Every coalition uses the identical already-opened balanced conditional-fit n=32 role. Confidence intervals are nonblocking controls, but window/domain identity and window-level Shapley efficiency are hard integrity checks. This pilot cannot qualify the codec or substitute for the native P8-to-MXFP8 game. Selection, confirmation, final, and the 28 reserved confirmation logits remain unopened. LDLQ and BlockLDLQ remain excluded.

## 17. 2026-09-05: coalition overlays must merge the quantization metadata ABI

The first three pilot launch attempts opened zero KLD windows and are preserved as preflight failures. The final failure localized a metadata error in the zero-copy coalition builder: it installed full-width BF16 weights for layers 3/19/20/22 but left those layers declared as NVFP4 in `config.json` and `hf_quant_config.json`. ModelOpt consequently allocated a packed width-1024 destination and rejected the width-2048 BF16 `w2`. Generate both coalition config files from the pinned carrier with every BF16 pilot layer removed from `config_groups[*].targets` and `quantized_layers`. This is invariant across coalitions because both the GPTQ base and P8 candidate are decoded BF16 overlays. The layer assignments, weight bytes, run order, role, runtime, estimator, stopping rule, and protected-data boundary remain unchanged.

## 18. 2026-09-05: restore the frozen pilot payload, not the later runtime-rounding variant

Preflight validation found 32 broken candidate links before a fourth endpoint launch. Restore layer 3 by reference-decoding its four surviving historically hashed codec chunks; restore layer 19 from detached encoder revision `2da233d655b237a664a8474111a05471c03ecf78`, before the later native-E4M3 tie-rounding change. The layer-19 reproduction matches the historical codebook hash and all 864 stored per-expert/projection NMSE values exactly. Safetensors container hashes are not claimed to match because regenerated serialization differed. Delete the non-frozen current-rounding data payload to recover 18 GB, but preserve its receipts and logs as a diagnostic. Do not launch the pilot unless every indexed source resolves and quantization metadata excludes all four BF16 layers.

## 19. 2026-09-05: direct-KLD interaction pilot supports allocation, not a quality claim

All eight frozen coalitions completed on the same 32 domain-balanced conditional-fit windows. Both antithetic paths have zero maximum window-level efficiency residual. The decoded-GPTQ empty coalition measured `0.0388388907`; upgrading all four pilot layers to decoded P8 measured `0.0397203962`, a `2.26965%` regression. The lowest observed nonempty coalition was layers 3/20/22 at `0.0376358981`, a `3.09739%` arithmetic-mean improvement over empty with 20/32 window wins and post-hoc paired BCa delta interval `[-0.00352257,+0.00049846]`. This interval crosses zero and the subset was identified after observing the coalition matrix, so it is adaptive developmental evidence only.

The interaction is material: removing layer 19 from the all-four context improves KLD by `5.24794%` relative to all-four, even though layer 19 alone improves the empty baseline by `0.79108%`. Its two contextual marginal reductions are `+0.00030725` and `-0.00208450`; layer 3 likewise ranges from `+0.00003830` to `+0.00081934`. Continue with direct-KLD allocation, but do not treat a two-permutation pilot ranking or local NMSE as a final selector. This pilot used matched BF16 overlays and says nothing about native speed. The physical P8 layer-3 path remains the only end-to-end fused trellis-to-`mxf8f6f4` kernel measurement; full-checkpoint P8, the native P8-to-MXFP8 6-bpw game, P4, and speed remain open. No LDLQ or BlockLDLQ was used.

## 20. 2026-09-05: compact shared middle rotation advances on direct KLD

V6 forced block-local H16 and V7 identity-selective block H16 both improved their fitted local objectives but worsened conditional-fit KLD. Replace those high-cardinality descriptors with one 1,024-byte FP32 SO(16) butterfly shared across every layer-3 expert and every down-projection K16 block. Gate/up remain the stock NVFP4 carrier. Re-encode down with full-Hessian GPTQ at exact ModelOpt NVFP4 rate, apply the shared rotation after SwiGLU in the existing Humming BF16 hook, and rank the fixed angle grid by direct all-causal teacher KLD. Do not fuse the hook into the native prologue until the KLD linkage survives independent tuning.

On the sealed domain-balanced fit/train32 split, stock measured `0.0364503459`. The matched zero-angle full-GPTQ down control measured `0.0375109066`, a `2.9096%` regression. The best nonzero arm, `+pi/16`, measured `0.0356555443`: `2.1805%` better than stock, `4.9462%` better than the matched zero control, and better than stock on 20/32 windows. The paired BCa interval versus stock was `[-0.00306204,+0.00105526]`; it is a nonblocking post-selection development control, not a quality claim. Fixed Hadamard endpoints failed to lead: `-pi/4` was `1.3875%` worse than stock and `+pi/4` was `0.3395%` worse. The fitted reconstruction objective preferred approximately `abs(angle)=pi/8`, confirming that local Hessian error can construct candidates but cannot rank causal KLD reliably.

Advance only `+pi/16`, the matched learned-identity control, and stock to the already declared disjoint tune32 split. The tune gate remains: candidate mean below both controls, delta versus stock at most `-0.0004`, and no domain mean delta above `+0.0005`; paired BCa is reported but nonblocking. This is adaptive fit evidence for a 4.5-bpw native-NVFP4 rotation, not the separate 4.25-bpw P8 product. It does not establish a fused prologue, speed, all-layer allocation, or qualification. Protected roles and reserved confirmation logits remain unopened; no LDLQ or BlockLDLQ was used.

## 21. 2026-09-05: freeze one rotation-by-codec interaction before tune closes

Regardless of the still-running tune32 outcome, give the already selected
`+pi/16` shared middle butterfly exactly one developmental layer-3 interaction
test in the physical 4.25-bpw TrellisMX-P8 codec.  Do not substitute a new
angle or rerun this interaction based on tune.  Gate/up use the ordinary K4
procedural-MCG/E4M3/UE8M0-K32 encoder; W2 is encoded as `W2 @ R` with the same
full-Hessian GPTQ-style inter-group feedback and no LDLQ.  The pseudoquant
runtime must round post-SwiGLU to BF16, execute the four scalar-angle Givens
stages in FP32 registers, round the final rotated vector to BF16, quantize it
to E4M3/UE8M0-K32, and consume the decoded rotated-basis W2.

The standalone tune result controls only claims about the 4.5-bpw ModelOpt
rotation.  The combined test asks a different causal question and will report
its result even if tune fails.  Compare it once against fresh stock and the
unrotated physical-P8 pseudoquant control on the same already opened tune32
windows.  Advancement into the fused device prologue requires the combined
candidate mean to be below unrotated P8 and not worse than stock; the paired
BCa interval is a nonblocking development control.  The kernel must then close
to this exact staged pseudoquant arithmetic before any speed claim.  This does
not open selection, confirmation, final, or the 28 reserved confirmation
logits, and it does not yet qualify an all-42-layer product.

## 22. 2026-09-05: standalone p00625 tune fails; preserve the precommitted interaction

On the disjoint domain-balanced tune32 split, fresh stock measured
`0.0363378694`, matched zero-angle GPTQ down measured `0.0364762391`, and
`+pi/16` measured `0.0371841305`.  Candidate minus stock is
`+0.0008462611` (`2.3289%` worse), with paired BCa 95% interval
`[-0.00110295,+0.00468556]`; candidate minus zero is `+0.0007078914`.
General and legal domain deltas versus stock are `+0.00308741` and
`+0.00292800`, respectively.  All four tune checks fail.  Therefore no claim
that the standalone 4.5-bpw ModelOpt rotation beats stock survives tune.

The candidate KLD manifest completed before the wrapper failed.  The failure
was caused by editing the active `run_kld_v3.sh` file while its long-lived
shell was executing; the shifted lazy-read source resumed at `--config-id` and
exited 127 after scoring.  Do not rerun or replace those windows.  The final
server log was copied bit-for-bit to the expected rotation-log filename, the
existing four-rank runtime verifier passed, and the frozen analyzer was run
once over the three complete manifests.  The recovery receipt preserves those
hashes and labels the event.

Decision 21 predates this result and remains binding: run exactly one
`+pi/16` by TrellisMX-P8 interaction at layer 3.  A combined pass may advance
the codec-specific fused prologue, but it cannot rehabilitate the failed
standalone rotation claim.  No alternative angle, protected role, or LDLQ path
is introduced.
