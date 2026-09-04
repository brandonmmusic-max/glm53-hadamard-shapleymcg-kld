# GLM-5.3-Flash in-block H16 and ShapleyMCG KLD experiment

This repository is the reproducible code-and-evidence record for a local
GLM-5.3-Flash BF16-to-NVFP4 experiment on four RTX PRO 6000 Blackwell GPUs.
It tests block-local 16-by-16 Hadamard transforms inside every routed-expert
projection, calibrated GPTQ, and a later 6-bpw ShapleyMCG global allocation.

The campaign is still in progress. The strongest conditional-fit result as of
2026-09-03 is a **5.493% reduction in end-to-end KLD** for routed MoE layers
3, 19, and 20, but that adaptive result did not generalize: the pre-frozen
candidate was 0.339% worse on its untouched selection wave. The independent
pre-frozen runner-up (layers 19 and 20) improved its untouched wave by only
0.407%, with a confidence interval crossing zero. Therefore **no selective or
uniform H16 candidate has a qualified protected-selection win on GLM**.

### Codec redesign status (2026-09-04)

The no-LDLQ P8 redesign now has a matched-path numerical gate. Its encoder
keeps the native 16x16 trellis layout, uses full-Hessian GPTQ-style error
feedback between groups with static within-group activation order, and refits
K32 UE8M0 scales with the Viterbi codes. On 16 held-out experts it improved
routed gate/up projection NMSE by **14.95%** versus GPTQ NVFP4 (16/16 wins),
and the causal gate/up/SwiGLU/down screen improved full-expert NMSE by
**11.87%** (15/16 wins). Layers 19 and 20 independently reproduced the local
causal NMSE effect at **16.52%** and **16.74%**, each with 16/16 expert wins.
Layer 22 produced the strongest local result: **17.07%** lower causal
full-expert NMSE with 16/16 wins after an exact full-activation recapture.

An earlier KLD comparison was invalid for encoder attribution because it put
the BF16 candidate against a packed-NVFP4/Humming control. After decoding the
GPTQ control to BF16 and putting both arms through the identical
InstantTensor/framework path, per-layer conditional-fit attribution found a
**4.157% KLD reduction at layer 3** and **3.086% at layer 20**; both
Bonferroni-adjusted intervals excluded zero. Layer 19 was a null. The combined
layer-3/20 subset then reduced mean KLD by only **1.383%** on all 63 balanced
V5 windows, with BCa 95% CI `[-0.002496,+0.000661]`; legal and code/agentic
domain means slightly regressed. It therefore failed the preregistered 3%
external gate. These are useful developmental KLD effects, not a generalized
or protected codec win.

The preregistered matched-path end-to-end test of the strong layer-22
candidate also failed. On the domain-balanced conditional-fit n=32 panel,
candidate mean KLD was `0.0391435578` versus `0.0387275135` for decoded GPTQ:
delta `+0.0004160443`, or a **1.074% regression**, with paired BCa 95% CI
`[-0.000249,+0.001168]`. This directly shows that the large local causal-NMSE
gain is not a reliable end-to-end selection surrogate for this codec.

The subsequent scaled-H128 procedural-MCG build also failed its frozen
layer-3 gate. Its physical payload is 4.251302 bpw and its 16-expert screen
improved causal routed-output NMSE by 26.189%, but on all 32 conditional-fit
windows its matched pseudoquant mean KLD was `0.0534951744` versus
`0.0533314176` for decoded GPTQ NVFP4: delta `+0.0001637568`, or a **0.307%
regression**, with paired BCa 95% CI `[-0.002134,+0.005994]`. It improved
22/32 windows and three of four domain means; the legal domain regressed by
`+0.0058845`, dominated by `conditional-fit-0074` at `+0.0487102`. This is a
developmental null/fail, not evidence that the physical codec is worse on a
protected holdout. Confirmation logits remain unopened, and neither LDLQ nor
BlockLDLQ was used.

![Matched-path codec-v2 KLD effects](figures/codec-v2-matched-kld.png)

The P4/P8 kernel closure gates remain blocked. P8 first failed E4M3 activation
carrier closure—even near-lossless MXFP6 weights do not serve within CI of
the BF16 path—and the redesigned encoder now also fails its matched-path
end-to-end KLD gate. Consequently no native prologue, speed, or determinism
claim is made.
A checkpoint-family learned 4 KiB T12 law also failed on 16 disjoint experts
(0/16 wins versus MCG), so it was stopped before another full-layer build. No
LDLQ or BlockLDLQ path is implemented or used.

| Endpoint | Mean KLD | Change vs same-loader stock | Paired BCa 95% CI for candidate minus stock | Status |
|---|---:|---:|---:|---|
| Stock NVFP4, InstantTensor/Humming | 0.0385846920 | reference | — | valid control |
| Fixed H16, all projections, 10k calibration cap, layer 3 only | 0.0375717123 | **-2.625%** | [-0.00171749, -0.00017658] | qualified conditional-fit pass |
| Exact Qwen recipe, 256 samples/expert, layer 3 only | 0.0377024045 | -2.287% | [-0.00206005, +0.00058266] | mean improved; interval crosses zero |
| Learned block rotation, all projections, layer 3 only | 0.0385788029 | -0.590% against its stock anchor | [-0.00229439, +0.00114439] | did not qualify |
| Exact Qwen fixed H16, all 42 routed layers | 0.0380729712 | -1.326% | [-0.00217155, +0.00185182] | mean improved; conditional-fit fail |

Lower KLD is better. These are 16-window conditional-fit measurements with
20,000 paired bootstrap replicates. The full-42-layer row is a completed
full-model conditional-fit measurement, but its confidence interval crosses
zero; it is not a qualified improvement or a final-holdout claim. All valid
GLM NVFP4 rows use the runtime-qualified packed-W4
with BF16 expert activations (`W4A16_NVFP4`) at an exact 4.5000076294-bpw
routed-weight payload; no W4A4 result is claimed.

![Current KLD comparison](figures/current-kld.png)

### Selective-placement generalization

| Pre-frozen candidate | Conditional-fit change | Protected selection change | Selection BCa 95% CI for candidate minus stock | Decision |
|---|---:|---:|---:|---|
| Layers 3, 19, 20 | -5.493% | **+0.339% worse** | [-0.00099953, +0.00269914] | fail |
| Layers 19, 20 | -2.956% | -0.407% | [-0.00110568, +0.00066555] | fail |

The conditional-fit rows were used adaptively and are not claims. All three
predeclared selection waves are now opened, so further layer-subset tuning on
V3 would be leakage. A new hypothesis requires a new sealed campaign.

![Selective H16 generalization](figures/selective-h16-generalization.png)

## What is established

- The exact Qwen block-H16 transformation is implemented per tensor family:
  `R_in` is shared by gate/up, `R_mid` is separate for down, weights use
  `W R`, Hessians use `R^T H R`, and runtime activations receive the matching
  transform inside the MoE execution path.
- Fixed H16 is the only rotation family with a statistically qualified GLM
  improvement so far.
- Uniform H16 across all 42 routed layers reduced mean KLD by 1.326%, but the
  paired BCa interval crossed zero. This shows that forcing the same transform
  everywhere dilutes the qualified layer-3 gain.
- Selective placement also failed to generalize. Layers 3/19/20 moved from
  +5.493% on conditional-fit to -0.339% on selection; layers 19/20 moved from
  +2.956% to +0.407%, with its selection interval crossing zero.
- The learned candidate was significantly worse than fixed H16 in the direct
  comparison: learned minus H16 KLD was +0.00100709 with BCa 95% CI
  [+0.00013188, +0.00196291].
- Increasing the calibration cap from the exact-Qwen 256 samples/expert to
  10,000 did not produce a distinguishable difference on this layer.
- The earlier catastrophic H16 result was invalid. A sparse overlay index was
  used with the ordinary safetensors loader, allowing later carrier shards to
  overwrite redirected tensors. Version-2 overlays now fail closed unless
  loaded by InstantTensor.

## Why this is not yet the Qwen-sized gain

The earlier Qwen headline was a **full-model recipe compared with a weak
shipped RTN NVFP4 routed-expert control**: -29% on the final panel, -36% on
selection, and -9% on WikiText at the same routed-weight rate. It combined
REAP calibration, GPTQ, fixed H16, and every routed layer. H16 was not
responsible for that entire gap: against the already calibrated GPTQ control,
fixed H16 contributed about -9% on final, -15% on selection, and -6% on
WikiText. GPTQ alone improved Qwen selection KLD by about 20% versus the stock
expert-only control. On GLM, however, the corrected fixed-loader identity-GPTQ
comparison was statistically indistinguishable from stock (`t = 0.24`; point
estimate `+1.09%` KLD, where lower is better). It rules out a Qwen-sized GPTQ
gain at the measured interval bound, but it is a null result and does not
establish a causal "GPTQ regression."

The calibration data itself is not the mismatch. The Hub corpus receipt's
source hash exactly matches the local REAP JSONL, and the Hub tokenizer receipt
exactly matches the GLM tokenizer. Those public receipts and the local
verification are under `evidence/provenance/glm-token-panel/`. Actual KLD
windows and absolute baseline values differ between Qwen and GLM, so raw
percentage improvements remain model- and panel-dependent. Exact Qwen inputs
and source hashes are recorded in
`evidence/qwen-reference/comparison.json`.

The next scientifically valid tests are:

1. create a new sealed campaign for a materially different hypothesis rather
   than continuing to tune layer subsets on exhausted V3 selection data; and
2. test a router-stability-aware or
   layer-specific transform objective before spending the 6-bpw confirmation
   holdout.

The executable 6-bpw ladder mixes whole layers of NVFP4 W4A16 and MXFP6 W6A8.
Its 5.9583409627-bpw budget is exact routed-weight storage. The Shapley and
uniform-depth controls have identical 35/7 format counts, so their comparison
isolates allocation value despite the different native activation formats.

No 10-30% end-to-end GLM KLD improvement has been measured. A 10-30% effect
has been measured for routed projection and causal full-expert NMSE. The best
matched-path per-layer KLD effect is 4.157% on adaptive conditional-fit data;
its layer-3/20 subset generalized to only 1.383% on V5 with an interval that
crossed zero, so a larger KLD claim remains unsupported.

## Repository map

- `glm53_nvfp4/`: GLM quantization, rotation, evaluation, freezing, and
  attribution implementation.
- `runtime_patch/`: fail-closed Humming/InstantTensor and B12X MXFP6 runtime
  transforms.
- `scripts/`: exact launch and campaign scripts.
- `tests/`: unit and receipt-verifier tests.
- `evidence/opened/`: raw JSON for already opened conditional-fit results.
- `evidence/opened/exact-v10-selective-h16/`: every selective build, adaptive
  conditional-fit comparison, freeze, and protected selection comparison.
- `evidence/historical/`: complete KLD run and runtime-session receipts,
  including superseded and invalidated diagnostic attempts.
- `results/current-kld-summary.json`: compact machine-readable result table.
- `results/selective-h16-summary.json`: machine-readable adaptive screen and
  protected generalization table.
- `docs/REPRODUCIBILITY.md`: exact identities, replay commands, and exclusions.
- `evidence/MANIFEST.sha256`: content hashes for the published evidence set.

## Quick verification

```bash
python3 -m pytest -q
python3 scripts/verify_public_evidence.py
python3 scripts/plot_current_kld.py
python3 scripts/build_selective_h16_summary.py
```

The current published snapshot passes 45 tests. Large model weights, teacher
logits, JIT caches, quantized checkpoints, raw calibration records, and
unopened holdout inputs are deliberately not stored in Git. Their immutable
identities and acquisition/replay requirements are documented instead.

## License and attribution

This work is **source-available**, not OSI open source. It is distributed under
the ShapleyMCG License 1.0 in `LICENSE`. Required attribution:

> ShapleyMCG was created by Brandon M. Music. Canonical source:
> https://github.com/brandonmmusic-max/shapleymcg

Third-party models, runtimes, and libraries remain under their own licenses.
See `THIRD_PARTY_NOTICES.md`.

## Related pinned context

The surrounding Local Inference Lab documentation was inspected at
`local-inference-lab/rtx6kpro@704e7974d0e052ac4d7427f328fc7bd412cb632d`:

- [GLM-5.3-Flash model hub](https://github.com/local-inference-lab/rtx6kpro/blob/704e7974d0e052ac4d7427f328fc7bd412cb632d/models/glm-5.3-flash.md)
- [BF16-to-NVFP4 KLD record](https://github.com/local-inference-lab/rtx6kpro/blob/704e7974d0e052ac4d7427f328fc7bd412cb632d/kld/glm-5.3-flash-bf16-nvfp4.md)
- [NVFP4 quantization comparison](https://github.com/local-inference-lab/rtx6kpro/blob/704e7974d0e052ac4d7427f328fc7bd412cb632d/benchmarks/nvfp4-quantization-comparison.md)

The earlier `glm52-sqg-mcg-experiments` repository is related prior work but
answers a different model/codec question. The canonical `shapleymcg`
repository remains the method and license authority.
