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
| B5-prov | W4A16 | 0.00978 | 0.9656 | 0.116 | 16 | B3p | -0.00172 [-0.00267, -0.00081] * |
| B5-prov2 | W4A16 | 0.00923 | 0.9671 | 0.107 | 16 | B3p | -0.00227 [-0.00351, -0.00115] * |
| B5L | W4A16 | 0.01085 | 0.9653 | 0.124 | 16 | B5 | +0.00107 [+0.00036, +0.00182] * |
| B5L2 | W4A16 | 0.01001 | 0.9664 | 0.119 | 16 | B5 | +0.00023 [-0.00058, +0.00118] |
| B5p | W4A16 | 0.01050 | 0.9639 | 0.129 | 16 | B5 | +0.00072 [+0.00011, +0.00145] * |
| B6h | W4A16 | 0.00962 | 0.9668 | 0.119 | 16 | B3p | -0.00188 [-0.00270, -0.00123] * |
| B6i | W4A16 | 0.00977 | 0.9670 | 0.118 | 16 | B3p | -0.00172 [-0.00250, -0.00108] * |
| B7 | W4A16 | 0.00841 | 0.9711 | 0.108 | 16 | E1b | +0.00194 [+0.00050, +0.00352] * |
| B7 | W4A4 | 0.02066 | 0.9467 | 0.241 | 16 | E1b | +0.01419 [+0.00975, +0.01886] * |
| B7a | W4A16 | 0.05264 | 0.9323 | 0.775 | 16 | E1a | +0.03824 [+0.01911, +0.06040] * |
| B7a | W4A4 | 0.06393 | 0.9182 | 0.889 | 16 | E1a | +0.04953 [+0.02839, +0.07340] * |
| B7n | W4A16 | 0.00690 | 0.9729 | 0.084 | 16 | E1b | +0.00044 [-0.00044, +0.00141] |
| B7n | W4A4 | 0.01874 | 0.9496 | 0.217 | 16 | E1b | +0.01227 [+0.00879, +0.01571] * |
| B7u | W4A16 | 0.01408 | 0.9638 | 0.189 | 16 | E1b | +0.00761 [+0.00426, +0.01136] * |
| B7u | W4A4 | 0.02566 | 0.9414 | 0.304 | 16 | E1b | +0.01919 [+0.01314, +0.02563] * |
| B7ua | W4A16 | 0.08597 | 0.9140 | 1.296 | 16 | E1a | +0.07157 [+0.04407, +0.10016] * |
| B7ua | W4A4 | 0.09822 | 0.9009 | 1.385 | 16 | E1a | +0.08382 [+0.05332, +0.11518] * |
| B7un | W4A16 | 0.00728 | 0.9713 | 0.086 | 16 | E1b | +0.00081 [+0.00003, +0.00161] * |
| B7un | W4A4 | 0.01860 | 0.9481 | 0.210 | 16 | E1b | +0.01213 [+0.00880, +0.01530] * |
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
| B7 | W4A16 | 0.01372 | 0.9676 | 0.206 | 25 | E1b | +0.00665 [+0.00239, +0.01231] * |
| B7 | W4A4 | 0.02675 | 0.9455 | 0.399 | 25 | E1b | +0.01967 [+0.01300, +0.02834] * |
| B7a | W4A16 | 0.07540 | 0.9263 | 1.315 | 25 | E1a | +0.05815 [+0.03388, +0.08651] * |
| B7a | W4A4 | 0.08857 | 0.9146 | 1.492 | 25 | E1a | +0.07133 [+0.04471, +0.10268] * |
| B7n | W4A16 | 0.00900 | 0.9700 | 0.130 | 25 | E1b | +0.00193 [+0.00018, +0.00438] * |
| B7n | W4A4 | 0.02308 | 0.9469 | 0.320 | 25 | E1b | +0.01601 [+0.01080, +0.02278] * |
| B7u | W4A16 | 0.01643 | 0.9633 | 0.246 | 25 | E1b | +0.00936 [+0.00542, +0.01403] * |
| B7u | W4A4 | 0.02990 | 0.9457 | 0.432 | 25 | E1b | +0.02282 [+0.01556, +0.03186] * |
| B7ua | W4A16 | 0.11761 | 0.9077 | 2.081 | 25 | E1a | +0.10037 [+0.06376, +0.14268] * |
| B7ua | W4A4 | 0.13231 | 0.8962 | 2.230 | 25 | E1a | +0.11507 [+0.07610, +0.16019] * |
| B7un | W4A16 | 0.00883 | 0.9705 | 0.127 | 25 | E1b | +0.00176 [+0.00018, +0.00381] * |
| B7un | W4A4 | 0.02225 | 0.9482 | 0.306 | 25 | E1b | +0.01518 [+0.01054, +0.02098] * |
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
| B7 | W4A16 | 0.02843 | 0.9376 | 0.393 | 10 | E1b | +0.01909 [+0.01499, +0.02345] * |
| B7 | W4A4 | 0.04788 | 0.9160 | 0.600 | 10 | E1b | +0.03853 [+0.03341, +0.04423] * |
| B7a | W4A16 | 0.16821 | 0.8577 | 2.445 | 10 | E1a | +0.14711 [+0.12643, +0.17279] * |
| B7a | W4A4 | 0.19119 | 0.8463 | 2.707 | 10 | E1a | +0.17008 [+0.15035, +0.19525] * |
| B7n | W4A16 | 0.01579 | 0.9520 | 0.216 | 10 | E1b | +0.00645 [+0.00525, +0.00774] * |
| B7n | W4A4 | 0.03719 | 0.9238 | 0.437 | 10 | E1b | +0.02785 [+0.02496, +0.03083] * |
| B7u | W4A16 | 0.03853 | 0.9348 | 0.535 | 10 | E1b | +0.02919 [+0.01994, +0.04052] * |
| B7u | W4A4 | 0.05807 | 0.9104 | 0.731 | 10 | E1b | +0.04873 [+0.03882, +0.06070] * |
| B7ua | W4A16 | 0.26713 | 0.8191 | 3.424 | 10 | E1a | +0.24603 [+0.22033, +0.26950] * |
| B7ua | W4A4 | 0.28570 | 0.8089 | 3.503 | 10 | E1a | +0.26460 [+0.23836, +0.28906] * |
| B7un | W4A16 | 0.01434 | 0.9550 | 0.178 | 10 | E1b | +0.00500 [+0.00365, +0.00630] * |
| B7un | W4A4 | 0.03502 | 0.9280 | 0.381 | 10 | E1b | +0.02568 [+0.02254, +0.02911] * |
| B8 | W4A16 | 0.03332 | 0.9380 | 0.463 | 10 | E1b | +0.02397 [+0.01711, +0.03241] * |
| B8 | W4A4 | 0.05206 | 0.9133 | 0.624 | 10 | E1b | +0.04272 [+0.03447, +0.05286] * |
| B8b | W4A16 | 0.01466 | 0.9527 | 0.174 | 10 | E1b | +0.00532 [+0.00376, +0.00705] * |
| B8b | W4A4 | 0.03478 | 0.9259 | 0.417 | 10 | E1b | +0.02544 [+0.02314, +0.02781] * |
| B8c | W4A16 | 0.01374 | 0.9560 | 0.169 | 10 | E1b | +0.00440 [+0.00316, +0.00564] * |
| B8c | W4A4 | 0.03507 | 0.9255 | 0.420 | 10 | E1b | +0.02573 [+0.02241, +0.02917] * |
| E1a | W4A16 | 0.02111 | 0.9432 | 0.221 | 10 |  |  |
| E1b | W4A16 | 0.00934 | 0.9623 | 0.097 | 10 |  |  |
| E1b-experts | W4A16 | 0.00684 | 0.9689 | 0.068 | 10 | E1b | -0.00250 [-0.00320, -0.00194] * |

---

# Round 2 — ShapleyMCG path (2026-09-02, after Brandon's 06:09 correction)

Calibration data = Brandon's REAP calibration corpus (reap_recall_calib.jsonl, 12,228 samples, 4 balanced axes) packed into 64 x 2048-token windows (role calib; first 32 = calib-attrib) — NOT the sealed evaluation windows (DECISIONS #15). Path: Hessians + routed samples on calib (64 windows) -> calibrated block-16 rotation (calib samples) vs Had16 decided on the selection role -> provisional endpoint B5-prov2 (uniform NVFP4, Had16, GPTQ) -> path-integrated Aumann-Shapley attribution of measured end-to-end KLD (calib-attrib, 32 windows, 5 Gauss-Legendre nodes) -> calibrated exact-byte allocation with the 4:8 pair-structured sparse tier (PTX ISA 9.3 verified) -> causal layer-by-layer re-encode (Hessians recaptured on calib-attrib per layer) with confirmation-role re-anchors -> final/selection/WikiText KLD scored exactly as the EXL3 controls. Declared in DECISIONS.md items 7-16 before any allocation ran.

## Rotation calibration verdict (selection role, W4A16, uniform NVFP4, GPTQ)
- B5L2 (calibrated-learned per layer, REAP calib) vs B5 (Had16): 0.01001 vs 0.00978 (ratio 1.02x; diff +0.00023, 95% CI [-0.00058, +0.00118], includes 0)
- per-layer basis chosen by the calibrated objective: {'learned-identity': 5, 'learned-had16': 1, 'had16': 1, 'identity': 41}

## Attribution reconciliation (calib-attrib windows, REAP corpus)
- KLD(src)=0.000000, KLD(end)=0.004662, sum of unit shares=0.001731, remainder=+0.002931 (+62.9% of the endpoint KLD), windows=32, nodes=5
- shares: negative 7236 of 18432; top-1% of units carry 31.3% of the total

## Allocations (18,432 units; T0 = 4:8 pair-structured sparse NVFP4)
- alloc-B7: 5.029 bpw used of 5.029; units T0 359, T1 12226, T2 5842, T4 5; attribution: sum 0.00173 of KLD_end 0.00466, remainder +0.00293, negative shares 7236, fallback units 7236
- alloc-B7a: 4.029 bpw used of 4.029; units T0 5814, T1 12600, T2 18; attribution: sum 0.00173 of KLD_end 0.00466, remainder +0.00293, negative shares 7236, fallback units 7236
- alloc-B7u: 5.029 bpw used of 5.029; units T0 1029, T1 11947, T2 5276, T3 2, T4 178; uncalibrated proxy
- alloc-B7ua: 4.029 bpw used of 4.029; units T0 7059, T1 10310, T2 1059, T4 4; uncalibrated proxy
- alloc-B7n: 5.029 bpw used of 5.029; units T1 12893, T2 5534, T4 5; attribution: sum 0.00173 of KLD_end 0.00466, remainder +0.00293, negative shares 7236, fallback units 7236
- alloc-B7un: 5.029 bpw used of 5.029; units T1 13545, T2 4764, T3 1, T4 122; uncalibrated proxy

## Causal re-encode re-anchors (confirmation role, W4A16, cumulative)
- B7: L3: 0.00269, L7: 0.00329, L11: 0.00372, L15: 0.00393, L19: 0.00445, L23: 0.00477, L27: 0.00498, L31: 0.00535, L35: 0.00577, L39: 0.00608, L43: 0.00684, L47: 0.00763
- B7a: L3: 0.00577, L7: 0.00935, L11: 0.01119, L15: 0.01294, L19: 0.01614, L23: 0.01961, L27: 0.02169, L31: 0.02540, L35: 0.02871, L39: 0.03211, L43: 0.03828, L47: 0.04922
- B7u: L3: 0.00828, L7: 0.00928, L11: 0.00974, L15: 0.00986, L19: 0.01021, L23: 0.01050, L27: 0.01068, L31: 0.01112, L35: 0.01132, L39: 0.01153, L43: 0.01155, L47: 0.01170
- B7ua: L3: 0.02680, L7: 0.05442, L11: 0.06631, L15: 0.07105, L19: 0.07708, L23: 0.08049, L27: 0.08156, L31: 0.08337, L35: 0.08379, L39: 0.08493, L43: 0.08575, L47: 0.08621
- B7n: L3: 0.00263, L7: 0.00297, L11: 0.00314, L15: 0.00332, L19: 0.00352, L23: 0.00386, L27: 0.00413, L31: 0.00439, L35: 0.00463, L39: 0.00492, L43: 0.00558, L47: 0.00646
- B7un: L3: 0.00285, L7: 0.00338, L11: 0.00368, L15: 0.00391, L19: 0.00431, L23: 0.00479, L27: 0.00506, L31: 0.00536, L35: 0.00549, L39: 0.00572, L43: 0.00581, L47: 0.00595

## Headline comparisons (final role, 25 x 2048; paired bootstrap over windows)
- PRIMARY B7 (Shapley-calibrated, E1b bytes) vs E1b (EXL3 5.0 bpw, full): 0.01372 vs 0.00707 (ratio 1.94x; diff +0.00665, 95% CI [+0.00239, +0.01231], excludes 0)
- B7 vs E1b-experts (EXL3 experts only, attention BF16): 0.01372 vs 0.00523 (ratio 2.62x; diff +0.00850, 95% CI [+0.00392, +0.01446], excludes 0)
- B7a (Shapley-calibrated, E1a bytes) vs E1a (EXL3 4.0 bpw, full): 0.07540 vs 0.01724 (ratio 4.37x; diff +0.05815, 95% CI [+0.03388, +0.08651], excludes 0)
- B7a vs E1a-experts: 0.07540 vs 0.01123 (ratio 6.72x; diff +0.06417, 95% CI [+0.03890, +0.09374], excludes 0)
- B7u (uncalibrated proxy control, E1b bytes) vs E1b: 0.01643 vs 0.00707 (ratio 2.32x; diff +0.00936, 95% CI [+0.00542, +0.01403], excludes 0)
- B7ua (uncalibrated proxy control, E1a bytes) vs E1a: 0.11761 vs 0.01724 (ratio 6.82x; diff +0.10037, 95% CI [+0.06376, +0.14268], excludes 0)
- Calibration effect B7 vs B7u (same candidates, same bytes): 0.01372 vs 0.01643 (ratio 0.84x; diff -0.00271, 95% CI [-0.00452, -0.00089], excludes 0)
- Calibration effect B7a vs B7ua: 0.07540 vs 0.11761 (ratio 0.64x; diff -0.04222, 95% CI [-0.06118, -0.02544], excludes 0)
- B7 vs B5-final (uniform NVFP4 Had16, 4.5 bpw): 0.01372 vs 0.01242 (ratio 1.10x; diff +0.00130, 95% CI [-0.00072, +0.00388], includes 0)
- B7 W4A4 vs E1b: 0.02675 vs 0.00707 (ratio 3.78x; diff +0.01967, 95% CI [+0.01300, +0.02834], excludes 0)

## POST-HOC ablation (DECISIONS #18, decided after the results above): no 4:8 tier, ladder {NVFP4, MXFP6, FP8, BF16}, E1b bytes, causal
- B7n (Shapley-calibrated, no T0) vs E1b: 0.00900 vs 0.00707 (ratio 1.27x; diff +0.00193, 95% CI [+0.00018, +0.00438], excludes 0)
- B7n vs E1b-experts: 0.00900 vs 0.00523 (ratio 1.72x; diff +0.00377, 95% CI [+0.00171, +0.00654], excludes 0)
- B7un (proxy, no T0) vs E1b: 0.00883 vs 0.00707 (ratio 1.25x; diff +0.00176, 95% CI [+0.00018, +0.00381], excludes 0)
- B7n vs B7un (calibration effect without the sparse tier): 0.00900 vs 0.00883 (ratio 1.02x; diff +0.00017, 95% CI [-0.00046, +0.00089], includes 0)
- B7n vs B7 (cost of the 4:8 tier under Shapley allocation): 0.00900 vs 0.01372 (ratio 0.66x; diff -0.00472, 95% CI [-0.00799, -0.00213], excludes 0)
- B7n vs B8c (pilot post-hoc: proxy, no T0, one-shot, fit-window Hessians): 0.00900 vs 0.00865 (ratio 1.04x; diff +0.00035, 95% CI [-0.00007, +0.00083], includes 0)
- B7n vs B5-final (uniform NVFP4 Had16 4.5 bpw): 0.00900 vs 0.01242 (ratio 0.72x; diff -0.00342, 95% CI [-0.00447, -0.00245], excludes 0)
- B7n W4A4 vs E1b: 0.02308 vs 0.00707 (ratio 3.26x; diff +0.01601, 95% CI [+0.01080, +0.02278], excludes 0)
- [selection] B7n vs E1b: 0.00690 vs 0.00647 (ratio 1.07x; diff +0.00044, 95% CI [-0.00044, +0.00141], includes 0); B7un vs E1b: 0.00728 vs 0.00647 (ratio 1.13x; diff +0.00081, 95% CI [+0.00003, +0.00161], excludes 0)
- B7a W4A4 vs E1a: 0.08857 vs 0.01724 (ratio 5.14x; diff +0.07133, 95% CI [+0.04471, +0.10268], excludes 0)

## Same comparisons on the selection role (16 x 2048) and WikiText (10 x 2048)
- [selection] B7 vs E1b: 0.00841 vs 0.00647 (ratio 1.30x; diff +0.00194, 95% CI [+0.00050, +0.00352], excludes 0); B7a vs E1a: 0.05264 vs 0.01440 (ratio 3.66x; diff +0.03824, 95% CI [+0.01911, +0.06040], excludes 0); B7u vs E1b: 0.01408 vs 0.00647 (ratio 2.18x; diff +0.00761, 95% CI [+0.00426, +0.01136], excludes 0); B7ua vs E1a: 0.08597 vs 0.01440 (ratio 5.97x; diff +0.07157, 95% CI [+0.04407, +0.10016], excludes 0)
- [wikitext] B7 vs E1b: 0.02843 vs 0.00934 (ratio 3.04x; diff +0.01909, 95% CI [+0.01499, +0.02345], excludes 0); B7a vs E1a: 0.16821 vs 0.02111 (ratio 7.97x; diff +0.14711, 95% CI [+0.12643, +0.17279], excludes 0); B7u vs E1b: 0.03853 vs 0.00934 (ratio 4.12x; diff +0.02919, 95% CI [+0.01994, +0.04052], excludes 0); B7ua vs E1a: 0.26713 vs 0.02111 (ratio 12.66x; diff +0.24603, 95% CI [+0.22033, +0.26950], excludes 0)

