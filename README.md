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
expert-only control. In contrast, GLM's completed uniform identity-GPTQ V2
candidate was 5.05% worse than stock on its protected selection panel. This is
the strongest measured explanation for the missing 10-30% GLM gain: H16 is
not being added to a large GPTQ improvement on GLM.

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
2. isolate GLM's GPTQ regression and test a router-stability-aware or
   layer-specific transform objective before spending the 6-bpw confirmation
   holdout.

The executable 6-bpw ladder mixes whole layers of NVFP4 W4A16 and MXFP6 W6A8.
Its 5.9583409627-bpw budget is exact routed-weight storage. The Shapley and
uniform-depth controls have identical 35/7 format counts, so their comparison
isolates allocation value despite the different native activation formats.

No 10-30% GLM improvement has been measured. The completed uniform and
selective protected tests did not reproduce that range, so it is unsupported
for the candidates tested here.

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
