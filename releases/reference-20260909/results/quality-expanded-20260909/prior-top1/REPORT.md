# Teacher agreement and original-text likelihood

Post-hoc CPU analysis of retained scores from the same 32 conditional-fit windows; 65,472 true-decode positions per model/cache mode. No new GPU runs or service changes. Every source score-array hash and teacher/token binding was checked.

| Model | KV cache | Teacher top-1 agreement | Matches / positions | Original-text perplexity |
|---|---|---:|---:|---:|
| TR3 / EXL3 4bpw | fp8 | 94.9612% | 62,173 / 65,472 | 2.766834 |
| TR3 / EXL3 4bpw | nvfp4 | 94.8100% | 62,074 / 65,472 | 2.768238 |
| TrellisMX reference | fp8 | 94.7443% | 62,031 / 65,472 | 2.763034 |
| TrellisMX reference | nvfp4 | 94.3273% | 61,758 / 65,472 | 2.770716 |

BF16 teacher perplexity on this exact text: **2.762712**. Lower perplexity means greater likelihood assigned to the observed continuation; it is not the same objective as minimizing full-distribution KL to the teacher.

| Model | Cache | Teacher p(top1) ≥ 0.5: agreement | Teacher p(top1) ≥ 0.9: agreement |
|---|---|---:|---:|
| TR3 | fp8 | 99.1085% | 99.9687% |
| TR3 | nvfp4 | 98.9980% | 99.9400% |
| TrellisMX | fp8 | 98.9866% | 99.9687% |
| TrellisMX | nvfp4 | 98.9351% | 99.9347% |

The confidence subsets contain 52,495 and 38,309 positions respectively and were defined by the same teacher probabilities for both students. Thresholds were specified before this analysis read the score arrays; this remains an exploratory follow-up on already-opened data.

## Paired differences

- fp8: TrellisMX minus TR3 teacher-agreement difference **-0.2169 percentage points**, paired-window BCa95 [-0.5896, +0.2337]. The two students agree with one another on **94.1303%** of positions.
- nvfp4: TrellisMX minus TR3 teacher-agreement difference **-0.4826 percentage points**, paired-window BCa95 [-1.0768, -0.1100]. The two students agree with one another on **93.7592%** of positions.

BCa uses 20,000 resamples of the 32 paired window means, seed 20260902. Confidence-subset percentages are pooled over eligible positions. No independent server replication or quality-equivalence margin was established.

## What the measurements mean

- Top-1 agreement asks whether the student chooses the same highest-probability next token as the BF16 teacher, given identical forced histories. It ignores probability shifts when the winner stays the same.
- Teacher-confidence subsets distinguish disagreement on decisive teacher choices from disagreement on less decisive positions. They do not establish that other disagreements are harmless.
- NLL and perplexity ask how well each model predicts the actual reference text. They complement KLD; a student may fit that text slightly better while matching the teacher distribution less closely. Paired excess-NLL intervals cross zero for both cache comparisons, so these small perplexity differences are not a clear system-level winner.
- Teacher agreement is fidelity, not answer correctness. The teacher may be wrong, and free-running generation can diverge from forced-history measurements. Runtime, EP, activation math and nonrouted-weight differences remain as in the KLD comparison.

## Additional compact-logit metrics to collect

A further capture could record k=5,10,50,100 and compute: top-k set overlap; whether the teacher top-1 remains in the student top-k; rank of the teacher top-1; and probability mass placed on the teacher top-k.

For top-k divergence, preserve an OTHER bucket containing all omitted probability mass. Renormalizing just the retained k tokens can hide probability shifted into the tail. A union of teacher/student top-k sets plus OTHER permits a coarse-grained KL or Jensen-Shannon comparison; it is a lower-resolution metric and cannot replace full-vocabulary KL.

The retained score files do not contain student top-k IDs/probabilities. Full student logits were retired after durable scoring, so these top-k metrics have NOT been measured here. A new capture can save compact top-k statistics before retirement without keeping the large raw files. Teacher logits remain available; the current BF16 teacher, rather than model size alone, supplies the pinned reference.

For actual answer quality beyond token fidelity, use a small blinded paired review of real prompts with verifiable answers or source-grounded requirements; score factual mistakes and missing requirements separately from wording preferences. That measures a different property from all logit comparisons.

Machine-readable values, per-window results, input hashes and the analysis plan are in results.json and analysis-plan.json.
