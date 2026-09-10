# Local quality comparison: TrellisMX and TR3 / EXL3 4bpw

CPU-only exploratory analysis of retained scores, September 9, 2026. No models were run or restarted. The same 32 already-opened conditional-fit windows supply 65,472 aligned next-token positions per arm (2,046 per window, excluding prediction row 0). Input score hashes and teacher/token bindings were verified. MX means the TrellisMX reference stack. FP8/NVFP4 refer to MLA KV cache precision, not weight format.

## Reference fidelity and text prediction

| Model/cache | Full BF16→student KL ↓ | BF16 top-1 agreement ↑ | Actual-text top-1 hit ↑ | Text perplexity ↓ | Actual-token probability MAE vs BF16 ↓ |
|---|---:|---:|---:|---:|---:|
| MX_fp8 | 0.03194517 | 94.7443% | 76.4113% | 2.763034 | 0.018921 |
| MX_nvfp4 | 0.03545622 | 94.3273% | 76.4113% | 2.770716 | 0.019874 |
| TR3_fp8 | 0.02818993 | 94.9612% | 76.4174% | 2.766834 | 0.017494 |
| TR3_nvfp4 | 0.03047854 | 94.8100% | 76.3563% | 2.768238 | 0.018555 |

BF16 teacher actual-text top-1 hit: 76.3807%; text perplexity: 2.762712.

### What each measurement means

- **Full BF16→student KL:** average divergence over all 154,880 vocabulary entries, in nats per prediction. Zero means identical distributions. These are the previously captured full-distribution scores, reaggregated here; lower means closer to BF16, not a percentage of answer-quality loss.
- **BF16 top-1 agreement:** how often the student and BF16 choose the same highest-probability next token on identical supplied histories. Higher means closer greedy choices; the teacher can still be wrong.
- **Actual-text top-1 hit:** how often the student’s preferred token equals the next token in the source text. Other continuations may also be valid, so this is not answer correctness.
- **NLL / perplexity:** NLL is the average negative natural log probability assigned to the actual next token; perplexity is exp(NLL). Lower means better prediction of this particular text. Tiny differences do not establish general superiority.
- **Actual-token probability MAE:** mean absolute probability difference on the source-text next token. It ignores redistribution among all the other tokens.
- **High-confidence agreement:** agreement restricted to positions where BF16 gives its preferred token at least 90% probability. This checks preservation of decisive reference choices without claiming that other disagreements are harmless.
- **Token KL quantiles:** median/p95/p99 summarize ordinary and unusually divergent positions. The maximum is one extreme token, not a typical result.
- **Student-pair top-1 agreement:** how often two student configurations choose the same next token; it does not identify which is better.
- **Two-bucket KL A→B / B→A:** collapse each distribution into [probability of the actual next token, probability of everything else]. Direction matters. This is a mathematically valid lower bound on the corresponding full-vocabulary KL, not an estimate or substitute for it.
- **Two-bucket Jensen–Shannon (JS):** symmetric comparison of those same two probabilities through their midpoint, in nats; zero means identical buckets. It also omits all distinctions within the other-token bucket.
- **Cache flips / lost / gained:** preferred-token changes between cache modes; lost/gained count positions where agreement with BF16 disappears/appears. These compare two captured runs, not independently replicated cache effects.

## Confident teacher choices

| Model/cache | Matches when BF16 p(top1) ≥ 0.9 |
|---|---:|
| MX_fp8 | 38,297 / 38,309 |
| MX_nvfp4 | 38,284 / 38,309 |
| TR3_fp8 | 38,297 / 38,309 |
| TR3_nvfp4 | 38,286 / 38,309 |

## All pairwise compact-distribution comparisons

All KL/JS values in this table are **two-bucket**, not full-vocabulary. BF16 self-comparison would be zero by identity and is not an independent BF16 replication.

| A vs B | KL A→B ↓ | KL B→A ↓ | JS ↓ | Actual-token probability MAE | Student top-1 agreement |
|---|---:|---:|---:|---:|---:|
| BF16 vs MX_fp8 | 0.00806666 | 0.00778317 | 0.00179985 | 0.018921 | See BF16 agreement above |
| BF16 vs MX_nvfp4 | 0.00876047 | 0.00849363 | 0.00195346 | 0.019874 | See BF16 agreement above |
| BF16 vs TR3_fp8 | 0.00688095 | 0.00657354 | 0.00154380 | 0.017494 | See BF16 agreement above |
| BF16 vs TR3_nvfp4 | 0.00758435 | 0.00735375 | 0.00171437 | 0.018555 | See BF16 agreement above |
| MX_fp8 vs MX_nvfp4 | 0.00600238 | 0.00617460 | 0.00140794 | 0.016761 | 95.2086% |
| MX_fp8 vs TR3_fp8 | 0.00967078 | 0.00968288 | 0.00220712 | 0.021221 | 94.1303% |
| MX_fp8 vs TR3_nvfp4 | 0.01024512 | 0.01028084 | 0.00231470 | 0.021680 | 94.0173% |
| MX_nvfp4 vs TR3_fp8 | 0.01072742 | 0.01058300 | 0.00240906 | 0.022260 | 93.8691% |
| MX_nvfp4 vs TR3_nvfp4 | 0.01078728 | 0.01080192 | 0.00243779 | 0.022446 | 93.7592% |
| TR3_fp8 vs TR3_nvfp4 | 0.00449169 | 0.00457515 | 0.00107893 | 0.014885 | 95.6409% |

## Cache sensitivity: FP8 → NVFP4

| Model | Top-1 flips / 65,472 | BF16 matches lost | BF16 matches gained | Increase in full BF16 KL |
|---|---:|---:|---:|---:|
| MX | 3,137 | 1,496 | 1,223 | 0.00351105 |
| TR3 | 2,854 | 1,282 | 1,183 | 0.00228861 |

## Divergence tails and NLL

| Model/cache | Mean NLL | KL median | KL p90 | KL p95 | KL p99 | KL max |
|---|---:|---:|---:|---:|---:|---:|
| MX_fp8 | 1.016329352 | 0.002004 | 0.065998 | 0.128497 | 0.453726 | 8.511914 |
| MX_nvfp4 | 1.019105940 | 0.002222 | 0.074602 | 0.148159 | 0.497530 | 9.321305 |
| TR3_fp8 | 1.017703650 | 0.001811 | 0.057711 | 0.115854 | 0.398845 | 10.970403 |
| TR3_nvfp4 | 1.018211176 | 0.002029 | 0.063497 | 0.125117 | 0.435940 | 14.413848 |

## Interpretation and limits

TR3 has lower mean full-reference KL and slightly higher reference top-1 agreement in both cache modes. TrellisMX FP8 has marginally lower text perplexity on these windows. The earlier paired-window NLL intervals cross zero for both cache comparisons, so the perplexity difference is not an established win. At ≥90% teacher confidence, FP8 configurations have exactly the same match count; NVFP4 differs by two matches.

For MX-minus-TR3 teacher top-1 agreement, paired-window BCa95 intervals are −0.5896 to +0.2337 percentage points (FP8) and −1.0768 to −0.1100 points (NVFP4). Full-KL TR3-minus-MX intervals are −0.00743989 to +0.00034560 (FP8) and −0.01067139 to −0.00081853 (NVFP4). These describe resampling the 32 windows, not independently prepared server-run variability. No equivalence margin or multiplicity-adjusted superiority claim is made.

This is a system comparison: TrellisMX r27 TP4/DCP4 without EP uses native FP8 expert math; TR3 r10 TP4/EP4/DCP4 uses W4A16 expert math and differs in nonrouted-weight policy and cache layout. Quality captures used MTP off, one sequence, batch 4096, and zero prefix hits. Each cache arm has one server preparation. The data were already opened development windows, not an untouched quality test. Forced histories do not test free-running answer divergence or factual correctness.

Full student logits/top-k distributions were retired after scoring. Therefore full student→student KL, full reverse student→BF16 KL, full JS, and top-k overlap/rank were **not measured**. Existing full teacher→student KL cannot reconstruct them. Two-bucket reverse and student-pair results above are explicitly lower-resolution measurements. No new capture was launched.

Reproducibility: [analysis script](analyze.py), [declared exploratory plan](plan.json), [per-window results and input hashes](results.json), and [previous top-1 analysis and paired intervals](prior-top1/REPORT.md).
