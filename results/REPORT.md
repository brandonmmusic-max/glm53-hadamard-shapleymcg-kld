# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — results (2026-09-02)

Pre-registration: bmxfp4/PREREGISTRATION.md (committed before candidate generation). Decision log: DECISIONS.md. Method notes and declared deviations: REPORT_NOTES.md. Every arm directory under arms/ holds its receipt.json, per-panel kld-*.json, and per-token KLD arrays.

Budgets (exact routed-expert payload bytes from the EXL3 safetensors headers): E1b = 18,223,202,304 B (5.029 bpw), E1a = 14,599,323,648 B (4.029 bpw). Uniform NVFP4 = 4.500 bpw (n/2 + n/16 + 4 bytes per matrix).

## Headline comparisons (mean per-window KLD, W4A16 unless stated; paired bootstrap over windows)

- PRIMARY (pre-registered): B8 five-tier proxy allocation at E1b bytes vs E1b (EXL3 5.0, full checkpoint), final role: 0.01568 vs 0.00707 (ratio 2.22x; diff +0.00861, 95% CI [+0.00454, +0.01348], excludes 0)
- PRIMARY, selection role: 0.01271 vs 0.00647 (ratio 1.96x; diff +0.00624, 95% CI [+0.00310, +0.00964], excludes 0)
- Bytes-parity row: B8 vs E1b experts-only (attention BF16), final: 0.01568 vs 0.00523 (ratio 3.00x; diff +0.01045, 95% CI [+0.00601, +0.01570], excludes 0)
- Secondary: B8a five-tier at E1a bytes vs E1a (EXL3 4.0 full), final: 0.08777 vs 0.01724 (ratio 5.09x; diff +0.07053, 95% CI [+0.04465, +0.10081], excludes 0)
- Secondary: B8 W4A4 vs E1b, final: 0.02854 vs 0.00707 (ratio 4.04x; diff +0.02147, 95% CI [+0.01448, +0.02993], excludes 0)
- Rotation verdict (selection): B5 Had16 vs B3' control: 0.00978 vs 0.01150 (ratio 0.85x; diff -0.00172, 95% CI [-0.00267, -0.00081], excludes 0)
- Rotation verdict (selection): B6h learned (Had init) vs B3': 0.00962 vs 0.01150 (ratio 0.84x; diff -0.00188, 95% CI [-0.00270, -0.00123], excludes 0)
- Rotation verdict (selection): B6i learned (identity init) vs B3': 0.00977 vs 0.01150 (ratio 0.85x; diff -0.00172, 95% CI [-0.00250, -0.00108], excludes 0)
- Permutation (selection): B5p Had16+diag_band vs B5: 0.01050 vs 0.00978 (ratio 1.07x; diff +0.00072, 95% CI [+0.00011, +0.00145], excludes 0)
- POST-HOC (exploratory): B8c {NVFP4,MXFP6,BF16} at E1b bytes vs E1b full, final: 0.00865 vs 0.00707 (ratio 1.22x; diff +0.00158, 95% CI [+0.00012, +0.00367], excludes 0)
- POST-HOC: B8c vs E1b full, selection: 0.00648 vs 0.00647 (ratio 1.00x; diff +0.00001, 95% CI [-0.00061, +0.00066], includes 0)
- POST-HOC: B8c vs E1b experts-only (bytes parity), final: 0.00865 vs 0.00523 (ratio 1.65x; diff +0.00342, 95% CI [+0.00163, +0.00581], excludes 0)
- POST-HOC: B8c vs B5 uniform NVFP4 Had16 (4.5 bpw), final: 0.00865 vs 0.01242 (ratio 0.70x; diff -0.00377, 95% CI [-0.00511, -0.00260], excludes 0)
- POST-HOC: B8b (route-frequency-weighted) vs B8c, final: 0.00886 vs 0.00865 (ratio 1.02x; diff +0.00021, 95% CI [-0.00026, +0.00077], includes 0)
- POST-HOC: B8r3 (same tier map, random rotation seed 3) vs B8c, final: 0.00834 vs 0.00865 (ratio 0.96x; diff -0.00031, 95% CI [-0.00086, +0.00013], includes 0)
- POST-HOC: B8c W4A4 vs E1b, final: 0.02221 vs 0.00707 (ratio 3.14x; diff +0.01513, 95% CI [+0.01089, +0.02061], excludes 0)
- Uniform NVFP4 GPTQ control B3' (4.5 bpw) vs E1a (EXL3 4.0 full), final: 0.01370 vs 0.01724 (ratio 0.79x; diff -0.00355, 95% CI [-0.00494, -0.00202], excludes 0)
- Uniform NVFP4 Had16 B5 (4.5 bpw) vs E1a full, final: 0.01242 vs 0.01724 (ratio 0.72x; diff -0.00482, 95% CI [-0.00619, -0.00349], excludes 0)

## Allocation histograms (units = expert x projection, 18,432 total)

- B8: five tiers, uncalibrated full-expert residual, E1b bytes (pre-registered): used 5.029 bpw of 5.029; units per tier {'T0_sparse_nvfp4': 1561, 'T1_nvfp4': 11459, 'T2_mxfp6': 5224, 'T4_bf16': 188}; per projection {'gate_proj': {'T0_sparse_nvfp4': 700, 'T1_nvfp4': 4137, 'T2_mxfp6': 1296, 'T4_bf16': 11}, 'up_proj': {'T0_sparse_nvfp4': 672, 'T1_nvfp4': 4127, 'T2_mxfp6': 1316, 'T4_bf16': 29}, 'down_proj': {'T0_sparse_nvfp4': 189, 'T1_nvfp4': 3195, 'T2_mxfp6': 2612, 'T4_bf16': 148}}
- B8a: five tiers, E1a bytes (pre-registered stress test): used 4.029 bpw of 4.029; units per tier {'T0_sparse_nvfp4': 9783, 'T1_nvfp4': 8024, 'T2_mxfp6': 625}; per projection {'gate_proj': {'T0_sparse_nvfp4': 3830, 'T1_nvfp4': 2237, 'T2_mxfp6': 77}, 'up_proj': {'T0_sparse_nvfp4': 3709, 'T1_nvfp4': 2329, 'T2_mxfp6': 106}, 'down_proj': {'T0_sparse_nvfp4': 2244, 'T1_nvfp4': 3458, 'T2_mxfp6': 442}}
- B8c: {NVFP4, MXFP6, BF16}, E1b bytes (post-hoc): used 5.029 bpw of 5.029; units per tier {'T1_nvfp4': 13606, 'T2_mxfp6': 4693, 'T4_bf16': 133}; per projection {'gate_proj': {'T1_nvfp4': 4973, 'T2_mxfp6': 1160, 'T4_bf16': 11}, 'up_proj': {'T1_nvfp4': 4942, 'T2_mxfp6': 1183, 'T4_bf16': 19}, 'down_proj': {'T1_nvfp4': 3691, 'T2_mxfp6': 2350, 'T4_bf16': 103}}
- B8b: same ladder, route-frequency-weighted values (post-hoc): used 5.029 bpw of 5.029; units per tier {'T1_nvfp4': 14012, 'T2_mxfp6': 4214, 'T4_bf16': 206}; per projection {'gate_proj': {'T1_nvfp4': 5075, 'T2_mxfp6': 1046, 'T4_bf16': 23}, 'up_proj': {'T1_nvfp4': 5021, 'T2_mxfp6': 1093, 'T4_bf16': 30}, 'down_proj': {'T1_nvfp4': 3916, 'T2_mxfp6': 2075, 'T4_bf16': 153}}

## Rotation learning (Cayley, 80 steps, fit-role Hessians; objective = Hessian-weighted RTN error)

- init identity: mean objective gain vs identity — gate/up -0.95%, down -2.52% (negative = worse than identity on the proxy objective)
- init had16: mean objective gain vs identity — gate/up -2.36%, down -15.10% (negative = worse than identity on the proxy objective)

# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — baseline ladder (mean per-window KLD, lower is better)

## Panel: selection

| arm | act | mean KLD | top-1 | p99 token KLD | windows | vs | diff [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| B3 | W4A16 | 0.01191 | 0.9616 | 0.128 | 16 | B3p | +0.00042 [-0.00096, +0.00165] |
| B3 | W4A4 | 0.02200 | 0.9455 | 0.239 | 16 | B3p | +0.01051 [+0.00755, +0.01326] * |
| B3p | W4A16 | 0.01150 | 0.9650 | 0.129 | 16 |  |  |
| B3p | W4A4 | 0.02109 | 0.9475 | 0.243 | 16 |  |  |
| B4s1 | W4A16 | 0.01001 | 0.9668 | 0.125 | 16 | B3p | -0.00148 [-0.00239, -0.00064] * |
| B4s2 | W4A16 | 0.00979 | 0.9668 | 0.119 | 16 | B3p | -0.00171 [-0.00279, -0.00071] * |
| B4s3 | W4A16 | 0.00960 | 0.9666 | 0.119 | 16 | B3p | -0.00190 [-0.00277, -0.00112] * |
| B5 | W4A16 | 0.00978 | 0.9656 | 0.116 | 16 | B3p | -0.00172 [-0.00267, -0.00081] * |
| B5 | W4A4 | 0.02185 | 0.9440 | 0.256 | 16 | B3p | +0.01036 [+0.00739, +0.01312] * |
| B5p | W4A16 | 0.01050 | 0.9639 | 0.129 | 16 | B5 | +0.00072 [+0.00011, +0.00145] * |
| B6h | W4A16 | 0.00962 | 0.9668 | 0.119 | 16 | B3p | -0.00188 [-0.00270, -0.00123] * |
| B6i | W4A16 | 0.00977 | 0.9670 | 0.118 | 16 | B3p | -0.00172 [-0.00250, -0.00108] * |
| B8 | W4A16 | 0.01271 | 0.9635 | 0.167 | 16 | E1b | +0.00624 [+0.00310, +0.00964] * |
| B8 | W4A4 | 0.02359 | 0.9435 | 0.300 | 16 | E1b | +0.01713 [+0.01164, +0.02270] * |
| B8a | W4A16 | 0.06541 | 0.9230 | 0.874 | 16 | E1a | +0.05100 [+0.03159, +0.07062] * |
| B8a | W4A4 | 0.07843 | 0.9125 | 1.009 | 16 | E1a | +0.06403 [+0.04121, +0.08731] * |
| B8b | W4A16 | 0.00685 | 0.9719 | 0.086 | 16 | E1b | +0.00038 [-0.00051, +0.00127] |
| B8b | W4A4 | 0.01937 | 0.9472 | 0.214 | 16 | E1b | +0.01291 [+0.00908, +0.01682] * |
| B8c | W4A16 | 0.00648 | 0.9714 | 0.083 | 16 | E1b | +0.00001 [-0.00061, +0.00066] |
| B8c | W4A4 | 0.01827 | 0.9490 | 0.214 | 16 | E1b | +0.01180 [+0.00842, +0.01515] * |
| B8r3 | W4A16 | 0.00699 | 0.9712 | 0.082 | 16 |  |  |
| E1a | W4A16 | 0.01440 | 0.9591 | 0.161 | 16 |  |  |
| E1a-experts | W4A16 | 0.00974 | 0.9670 | 0.107 | 16 | E1a | -0.00466 [-0.00600, -0.00347] * |
| E1b | W4A16 | 0.00647 | 0.9718 | 0.075 | 16 |  |  |
| E1b-experts | W4A16 | 0.00493 | 0.9765 | 0.059 | 16 | E1b | -0.00154 [-0.00207, -0.00101] * |

## Panel: final

| arm | act | mean KLD | top-1 | p99 token KLD | windows | vs | diff [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| B3p | W4A16 | 0.01370 | 0.9651 | 0.194 | 25 |  |  |
| B3p | W4A4 | 0.02621 | 0.9485 | 0.364 | 25 |  |  |
| B5-final | W4A16 | 0.01242 | 0.9665 | 0.184 | 25 | B3p | -0.00127 [-0.00201, -0.00064] * |
| B5-final | W4A4 | 0.02590 | 0.9462 | 0.367 | 25 | B3p | +0.01220 [+0.00907, +0.01601] * |
| B8 | W4A16 | 0.01568 | 0.9651 | 0.251 | 25 | E1b | +0.00861 [+0.00454, +0.01348] * |
| B8 | W4A4 | 0.02854 | 0.9442 | 0.407 | 25 | E1b | +0.02147 [+0.01448, +0.02993] * |
| B8a | W4A16 | 0.08777 | 0.9210 | 1.495 | 25 | E1a | +0.07053 [+0.04465, +0.10081] * |
| B8a | W4A4 | 0.10188 | 0.9091 | 1.734 | 25 | E1a | +0.08464 [+0.05577, +0.11815] * |
| B8b | W4A16 | 0.00886 | 0.9712 | 0.122 | 25 | E1b | +0.00179 [+0.00008, +0.00432] * |
| B8b | W4A4 | 0.02146 | 0.9482 | 0.299 | 25 | E1b | +0.01438 [+0.01040, +0.01929] * |
| B8c | W4A16 | 0.00865 | 0.9717 | 0.121 | 25 | E1b | +0.00158 [+0.00012, +0.00367] * |
| B8c | W4A4 | 0.02221 | 0.9481 | 0.298 | 25 | E1b | +0.01513 [+0.01089, +0.02061] * |
| B8r3 | W4A16 | 0.00834 | 0.9705 | 0.120 | 25 |  |  |
| E1a | W4A16 | 0.01724 | 0.9579 | 0.231 | 25 |  |  |
| E1a-experts | W4A16 | 0.01123 | 0.9662 | 0.146 | 25 | E1a | -0.00602 [-0.00751, -0.00460] * |
| E1b | W4A16 | 0.00707 | 0.9712 | 0.093 | 25 |  |  |
| E1b-experts | W4A16 | 0.00523 | 0.9757 | 0.069 | 25 | E1b | -0.00184 [-0.00238, -0.00136] * |

## Panel: wikitext

| arm | act | mean KLD | top-1 | p99 token KLD | windows | vs | diff [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| B3p | W4A16 | 0.02337 | 0.9461 | 0.292 | 10 |  |  |
| B3p | W4A4 | 0.03880 | 0.9224 | 0.472 | 10 |  |  |
| B5-final | W4A16 | 0.02193 | 0.9466 | 0.284 | 10 | B3p | -0.00144 [-0.00286, -0.00021] * |
| B5-final | W4A4 | 0.04174 | 0.9193 | 0.514 | 10 | B3p | +0.01837 [+0.01641, +0.02028] * |
| B8 | W4A16 | 0.03332 | 0.9380 | 0.463 | 10 | E1b | +0.02397 [+0.01711, +0.03241] * |
| B8 | W4A4 | 0.05206 | 0.9133 | 0.624 | 10 | E1b | +0.04272 [+0.03447, +0.05286] * |
| B8b | W4A16 | 0.01466 | 0.9527 | 0.174 | 10 | E1b | +0.00532 [+0.00376, +0.00705] * |
| B8b | W4A4 | 0.03478 | 0.9259 | 0.417 | 10 | E1b | +0.02544 [+0.02314, +0.02781] * |
| B8c | W4A16 | 0.01374 | 0.9560 | 0.169 | 10 | E1b | +0.00440 [+0.00316, +0.00564] * |
| B8c | W4A4 | 0.03507 | 0.9255 | 0.420 | 10 | E1b | +0.02573 [+0.02241, +0.02917] * |
| E1a | W4A16 | 0.02111 | 0.9432 | 0.221 | 10 |  |  |
| E1b | W4A16 | 0.00934 | 0.9623 | 0.097 | 10 |  |  |
| E1b-experts | W4A16 | 0.00684 | 0.9689 | 0.068 | 10 | E1b | -0.00250 [-0.00320, -0.00194] * |



## Decisions and deviations

# Decision log — BMXFP4 pilot, 2026-09-02 (times America/New_York)

All decisions below were made on the selection role (16 x 2048) or on process grounds; the final role (25 x 2048) was scored only for arms named here, once each.

1. 01:20 Stage 0. Teacher captured for selection/final/confirmation/WikiText; B0 bitwise-identical 3/3. Hessians: route-weight^2, fit 32 windows; 256 routed samples per expert stored. EXL3 5.0/4.0 reconstructed to BF16 in the pinned K3 image; routed-expert payload bytes 18,223,202,304 (5.029 bpw) and 14,599,323,648 (4.029 bpw).
2. 05:23 Rotation verdict (pre-registered secondary, selection role, W4A16): B5 Had16 0.00978 and every random seed (0.00960-0.01001) beat the B3' control (0.01150), all CIs excluding zero. Rotation is NOT dead for NVFP4 on this MoE model. Per the pre-registration ("best of {B3', B5, B6, B6p} by selection"), the candidate ladder was generated in the Had16 basis; B6 (learned) and B6p (permutation) were still running at that time and are compared afterwards.
3. 05:27 Candidate ladder observation: FP8 per-output-channel (T3) is dominated by MXFP6 (T2) on both bytes and error for every unit, so T3 is never selected (Pareto). 2:4-sparse NVFP4 (T0) carries ~25x the full-expert residual of dense NVFP4.
4. 05:34 Pre-registered allocation arm B8 (uncalibrated proxy, five tiers, E1b bytes): selection 0.01271 W4A16 — WORSE than uniform NVFP4 Had16 (0.00978) and than the identity control. B8a (E1a bytes; 53% of units forced to T0): 0.06541. Pre-registered reading: at these budgets the proxy-driven five-tier allocation loses to uniform; the sparse tier is disqualified at 4.0 bpw. Diagnosis from the allocation itself: the proxy ignored routing frequency and let rarely-routed experts be sold into T0 for free.
5. 05:36 POST-HOC (exploratory, NOT pre-registered; decided after seeing B8 on selection): B8b = ladder {T1, T2, T4} with route-frequency-weighted values; B8c = same ladder without frequency weighting. Both at E1b bytes. Because they were chosen after looking at selection-role KLD of a related arm, their final-role numbers are reported as exploratory, not confirmatory.
6. Not run (time): the calibrated B7 (per-layer KLD factors), B7t (FP8 backbone at equal total bytes), the external NVFP4 anchors B1/B2, and rotation sharing-granularity sweeps beyond per-layer-per-projection.


# BMXFP4 pilot on Qwen/Qwen3-30B-A3B — methodology notes (to accompany REPORT.md)

Generated 2026-09-02 overnight on 4x RTX PRO 6000 Blackwell (SM120a). Plan: BMXFP4_plan_v2.md; pre-registration: bmxfp4 repo PREREGISTRATION.md (committed before any candidate).

## What was measured, and how
- Teacher: Qwen/Qwen3-30B-A3B @ 4c446470 (weights identical to main), BF16, HF transformers 4.52, sdpa, use_cache off, fp32 logits per window stored (teacher/<panel>/row-NNN.safetensors). B0 = the first selection window recomputed 3x and compared bitwise (teacher/b0-receipt.json).
- Panels (token ids reused from the prior ShapleyMCG Qwen seal in brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility): selection 16 x 2048, final 25 x 2048, confirmation 16 x 2048 (four domains: general / legal / code-agentic / reasoning), plus the WikiText-2 10 x 2048 control panel. Fit (32 x 2048) was used only for Hessians, routed samples, and rotation learning.
- Metric: per-token KL(teacher || student), full vocabulary, float32 log-softmax, positions 0..2046; per-window mean; mean over windows; p90/p99 of per-token KLD; top-1 agreement; paired bootstrap 95% CI (20,000 draws over windows) of the difference vs the named comparator. Same definition as the repository's scoring/kld.py (which reports p95/p99 rather than p90).
- Pseudo-quantization: every arm is scored as an expanded-BF16 reconstruction installed into the HF model (routed experts only for BMXFP4 arms; the EXL3 anchors are scored both as the deployable full reconstruction and as an experts-only variant). W4A4 rows add NVFP4 activation quantization at every routed-expert input (per-token blocks of 16, dynamic E4M3 block scales, dynamic FP32 tensor scale) in the same transform basis as the weights.
- Byte budgets: exact routed-expert payload bytes of turboderp/Qwen3-30B-A3B-exl3 5.0bpw (E1b, primary) and 4.0bpw (E1a, stress) from their safetensors headers (seals/bytes-exl3-*.json). BMXFP4 tier bytes: NVFP4 = n/2 + n/16 + 4; 2:4-sparse NVFP4 = n/4 + n/8 (2-bit metadata per stored nonzero) + n/16 + 4 (one UE4M3 per 16 logical elements; the ISA question in the plan is recorded as this constant); MXFP6 = 6n/8 + n/32; FP8 = n + 4·rows; BF16 = 2n.

## Codec details (bmxfp4/nvfp4.py, gptq.py, rotation.py)
- NVFP4: E2M1 elements, UE4M3 per-16 block scale, per-tensor FP32 scale (ModelOpt convention amax/(6*448)), MSE-optimal block scale search (12 shrink factors) with 4-vs-6 range-max choice, two rounds of alternating per-tensor refit; straight-through roundings.
- GPTQ: batched over the 128 experts of a layer with per-expert route-weight^2 Hessians (fit role, 32 windows), 1% damping, column blocks of 128, in-group static act-order (descending Hessian diagonal inside each fixed 16-group, implemented as a permutation that leaves the storage layout unchanged), block scale searched once per group on the error-updated values.
- 2:4: SparseGPT-style saliency w^2/[H^-1]_ii^2 on the original 4-column layout, mask decided at group start, pruning error fed forward like rounding error.
- Rotations: block-diagonal 16x16 on the input dimension; W' = W R, H' = R^T H R, W_eff = Q(W')R^T; gate and up share one R per layer (they share the input), down has its own. Fixed Hadamard-16, seeded random SO(16) (five seeds), and Cayley-learned R = R0 (I-A)(I+A)^-1 with A = 0 at init, R0 in {I, Had16}.

## Declared deviations from the plan (for speed; each is a caveat on the corresponding row)
1. Allocation value: not the repository's Aumann-Shapley path integration. Each unit's value under each tier is the route-weight^2-weighted mean squared full-expert output error on 256 stored routed samples per expert when only that unit is quantized (others BF16). B8/B7-uncal use it directly; if per-layer KLD calibration runs were completed, B7 scales it by the measured per-layer KLD-per-unit-error factor.
2. Learned rotation objective: Hessian-weighted per-projection reconstruction through the NVFP4 quantizer with MSE scales (RTN inside the optimizer, 100 Adam steps on 16-expert batches), then GPTQ with the frozen R; not the full expert-function objective.
3. The codec is NOT integrated into the shapleymcg competitive ledger (its attestation gate is hard-bound to the EXL3/MCG codec and integer bits {3,4,5}); in the repository's terms this pilot codec is a diagnostic oracle. The results are therefore comparable in method (roles, metric, pseudo-quant scoring, exact bytes) but do not carry the repository's sealed-ledger receipts.
4. External NVFP4 anchors (RedHatAI, nvidia) were not scored.
5. The 89th sealed window (confirmation, code-agentic) re-tokenizes to 2047 tokens under the post-trained tokenizer; token ids were used directly, so all windows are 2048 tokens.
