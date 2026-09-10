# TrellisMX — GLM-5.3-Flash quantization toolkit and campaign record

**September 9 selected serving reference:** [image, launch configuration and complete speed/KLD evidence](releases/reference-20260909/README.md). Matched32window development KLD: FP8 KV **0.0319451732**, NVFP4 MLA KV **0.0354562238**, TP4/DCP4/MTPoff. [HF model card](https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8).


## Additional local quality measurements (September 9, 2026)

The same 32 windows / 65,472 prediction positions now include top-1 agreement, text likelihood, cache sensitivity, and compact student-to-student divergence. These are CPU analyses of retained scores; no new model execution.

| Model/cache | Full BF16→student KL ↓ | BF16 top-1 agreement ↑ | Actual-text top-1 hit ↑ | Text perplexity ↓ | Actual-token probability MAE vs BF16 ↓ |
|---|---:|---:|---:|---:|---:|
| MX_fp8 | 0.03194517 | 94.7443% | 76.4113% | 2.763034 | 0.018921 |
| MX_nvfp4 | 0.03545622 | 94.3273% | 76.4113% | 2.770716 | 0.019874 |
| TR3_fp8 | 0.02818993 | 94.9612% | 76.4174% | 2.766834 | 0.017494 |
| TR3_nvfp4 | 0.03047854 | 94.8100% | 76.3563% | 2.768238 | 0.018555 |

Top-1 agreement measures the same preferred token as BF16. Perplexity measures likelihood of the actual text. Probability MAE measures the average difference on that text’s next token. Full KL measures the whole distribution; none is a direct answer-correctness score. See the report for definitions of every metric.

When BF16 assigns its preferred token ≥90% probability, both FP8 students match 38,297/38,309 positions; NVFP4 matches are 38,286 for TR3 and 38,284 for TrellisMX. Changing FP8→NVFP4 cache flips top-1 at 3,137 positions for TrellisMX and 2,854 for TR3.

All six student-pair comparisons and four teacher/student comparisons include **two-bucket KL in both directions and Jensen–Shannon divergence**, grouping the actual next token versus all other tokens. These are lower-resolution divergences, not full-vocabulary student-pair KL. Full student logits were retired, so full student-pair KL and top-k overlap cannot be recovered without another capture.

TR3 has lower measured full-reference KL; TrellisMX FP8’s slightly lower text perplexity does not establish superiority. These are exploratory results on already-opened conditional-fit windows, with runtime/EP/math differences and one preparation per cache arm. They do not establish answer-quality equivalence.

[Full tables, metric explanations, limitations and reproducible analysis](releases/reference-20260909/results/quality-expanded-20260909/README.md).



Reproducible code and evidence for a local GLM-5.3-Flash quantization campaign on
four RTX PRO 6000 Blackwell GPUs. It covers block-local 16×16 Hadamard rotation
(H16) inside routed-expert projections, calibrated GPTQ, the native P8/TrellisMX
codec, and 6-bpw ShapleyMCG allocation.

The campaign is ongoing. This page states what is currently true. The dated
narrative of how each result was reached is in
**[docs/CAMPAIGN_HISTORY.md](docs/CAMPAIGN_HISTORY.md)**.

## How to read the results

Every number below carries one of four labels. They are not interchangeable.

| Label | Meaning |
| --- | --- |
| **Qualified (per-layer)** | Passed its preregistered per-layer gate with a BCa interval excluding zero. Scope is the layer and role stated in the row |
| **Developmental** | A real measurement on already-opened or adaptively-used data. Informative, not a claim |
| **Failed gate** | Ran against a predeclared threshold and did not meet it |
| **Superseded** | Replaced by a later measurement; kept for provenance |

A developmental number is not a weaker version of a qualified one. It answers a
different question, and no amount of them adds up to qualification.

> **Claim boundary, verbatim from `results/current-kld-summary.json`:** the
> full-42-layer fixed-H16 mean improved by 1.326%, but its interval crossed
> zero. **No statistically qualified full-model or final-holdout GLM improvement
> is claimed yet.**

## Status board

### End-to-end KLD, mean KL(teacher ‖ student), lower is better

| Result | KLD | BCa 95% CI | Routed bpw | Label |
| --- | ---: | :---: | ---: | --- |
| Coupled P8, 17×K5 / 25×K4 | **0.034181** | [0.029148, 0.040978] | 4.6587 | Developmental |
| Coupled P8, uniform K4 | 0.036967 | — | 4.2540 | Developmental |
| Uniform all-42 native P8 | 0.040021 | [0.033778, 0.048253] | 4.25 | Developmental |
| Fixed H16, layer 3 only | **0.037572** vs stock 0.038585 | [-0.001717, -0.000177] | — | **Qualified (per-layer)** |

The coupled 17-K5 checkpoint is 7.54% below uniform coupled K4, but at a higher
bitrate. It is not an equal-size comparison, not a matched stock-NVFP4 win, and
not a measurement of the later speed-optimized runtime.

### H16 placement

Signs below are **KLD change: negative is better**. Note that
`results/selective-h16-summary.json` stores the opposite convention
(`relative_improvement`, positive is better); the values agree.

| Candidate | Conditional-fit | Protected selection | CI (candidate − stock) | Label |
| --- | ---: | ---: | :---: | --- |
| Fixed H16, layer 3 | −2.63% | not tested at full model | [-0.001717, -0.000177] | **Qualified (per-layer)** |
| Layers 3, 19, 20 | −5.493% | **+0.339% worse** | [-0.00100, +0.00270] | Failed gate |
| Layers 19, 20 | −2.956% | −0.407% | [-0.00111, +0.00067] | Failed gate |
| Uniform H16, all 42 layers | −1.326% | — | crosses zero | Failed gate |
| Learned rotation vs fixed H16 | +0.001007 worse | — | [+0.000132, +0.001963] | Failed gate |

**No selective or uniform H16 candidate has a qualified protected-selection win
on GLM.** Layer 3 alone is the single qualified rotation result.

### RC5 release, 7 September

| Benchmark | Score | Notes |
| --- | --- | --- |
| Estonia | **30/30 PASS** | — |
| LAVD | **28/30 credited** | 22 exact + 6 near; 28/28 finished answers credited. Two 100,000-token-limit truncations, no scored FAIL |
| Hotel Lights | **28/30 exact** | Two scored failures, neither token-limited |

| MTP3 sustained decode | tok/s | Client prefill | tok/s |
| --- | ---: | --- | ---: |
| C1 | 178.34 | at 8k | ~8,060 |
| C2 | 235.61 | at 32k | ~8,000 |
| C4 aggregate | 300.87 | | |

**Label: Developmental.** These are one-run measurements. They are not a
200 tok/s claim and not 1M-context qualification. No KLD was measured on the
RC5 optimized runtime; the 0.034181 coupled figure above predates it.

[Results, conditions and raw receipts](results/RC5_RELEASE_RESULTS_20260907.md)

**Published runtime:** [Docker Hub `verdictai/trellismx`](https://hub.docker.com/r/verdictai/trellismx)
with a [digest-pinned Compose and serving script](deploy/trellismx/README.md).
The default maximum model length is 1,000,000. That launch limit is **not** a
claim of tested 1M retrieval quality.

### Serving speed, P8 versus the existing EXL3 topology

| Metric | P8 | EXL3 | Delta | Label |
| --- | ---: | ---: | ---: | --- |
| C1 decode, tok/s | 100.5501 | 91.2200 | **+10.23%** | Passed both gates |
| 32K prefill, tok/s | 6,959 | 6,239 | **+11.54%** | Passed both gates |

Five fresh cold runs per arm; historical samples excluded. This is
TP4/noEP/DCP1 P8 against TP4/EP4/DCP4 EXL3 — a serving-system comparison, not
isolated codec causality. P8 uses E4M3 MMA at twice NVFP4's issue count, so it is
not the P4/NVFP4 speed class. An earlier comparison on the previous runtime
failed the same gate (−77.33% C1 decode) and is retained as **Superseded**.

## What is established

- The exact Qwen block-H16 transformation is implemented per tensor family:
  `R_in` shared by gate/up, `R_mid` separate for down, weights use `W R`,
  Hessians use `Rᵀ H R`, and runtime activations receive the matching transform
  inside the MoE execution path.
- Fixed H16 is the only rotation family with a statistically qualified GLM
  improvement, and only at layer 3, on the 16-window conditional-fit role.
- Uniform H16 across all 42 routed layers dilutes that layer-3 gain rather than
  compounding it.
- Local NMSE gains do not predict end-to-end KLD. A 17.07% causal full-expert
  NMSE gain at layer 22 produced a 1.07% end-to-end KLD *regression*.
- Learned rotation was significantly worse than fixed H16 in direct comparison.
- Raising the calibration cap from 256 to 10,000 samples/expert produced no
  distinguishable difference on the measured layer.

## What is not established

- No 10–30% end-to-end GLM KLD improvement. That range has been measured only
  for routed-projection and causal full-expert NMSE, which is a different
  quantity.
- No matched-bitrate NVFP4 win. The coupled checkpoints spend more bits.
- No protected-holdout qualification for any multi-layer H16 placement.
- No qualified KLD for the speed-optimized decode path. The KLD measurements
  used the earlier eager/FP8 MLA KV regime.
- All three predeclared selection waves are now opened. Further layer-subset
  tuning on V3 data would be leakage; a new hypothesis needs a new sealed
  campaign.

## Why this is not the Qwen-sized gain

The Qwen headline compared a full recipe against a weak shipped RTN NVFP4
control: −29% final, −36% selection, −9% WikiText. H16 was not responsible for
that whole gap. Against an already-calibrated GPTQ control, fixed H16
contributed about −9% final, −15% selection, −6% WikiText.

On GLM the corrected fixed-loader identity-GPTQ comparison was statistically
indistinguishable from stock (`t = 0.24`, point estimate +1.09%). That rules out
a Qwen-sized GPTQ gain at the measured interval bound. It is a null result, not
evidence of a GPTQ regression.

The calibration data is not the mismatch: the Hub corpus receipt hash matches the
local REAP JSONL and the Hub tokenizer matches the GLM tokenizer, both verified
under `evidence/provenance/glm-token-panel/`.

## Toolkit

```bash
python -m pip install -e '.[test]'
trellismx --help
trellismx capabilities
```

The `trellismx` package exposes versioned P4/P8 formats, K3/K4/K5 CPU reference
decoding, checkpoint inspection and planning, resumable orchestration,
role/provenance validation, and an architecture-adapter contract.

**Support boundary.** This is not a ModelOpt-equivalent arbitrary-model BF16→P8
converter. GLM-5.3-Flash is the bundled adapter; other architectures need their
own adapter plus model and runtime validation. The CUDA coupled encoder is
research source, and ordinary CLI execution does not encode a complete model.

[Workflow](docs/TRELLISMX_WORKFLOW.md) ·
[Adapter template](configs/architecture-adapter-contract.template.json) ·
[Implementation and boundaries](docs/IMPLEMENTATION_REPORT.md)

## Repository map

| Path | Contents |
| --- | --- |
| `trellismx/` | Toolkit package and CLI |
| `glm53_nvfp4/` | Quantization, rotation, evaluation, freezing, attribution |
| `runtime_patch/` | Fail-closed Humming/InstantTensor and B12X MXFP6 transforms |
| `deploy/trellismx/` | Published image, digest-pinned Compose, serving script |
| `results/INDEX.md` | **Index of all 124 result files, grouped, with the current ones marked** |
| `results/` | Per-experiment reports and receipts |
| `results/current-kld-summary.json` | Machine-readable current result table |
| `results/P8_LATEST_KLD.md` | Latest coupled checkpoint and exact receipts |
| `evidence/opened/` | Raw JSON for opened conditional-fit results |
| `evidence/historical/` | Superseded and invalidated attempts, retained |
| `evidence/MANIFEST.sha256` | Content hashes for the published evidence set |
| `docs/REPRODUCIBILITY.md` | Exact identities, replay commands, exclusions |
| `docs/CAMPAIGN_HISTORY.md` | Dated narrative of every experiment |

## Verification

```bash
python3 -m pytest -q                        # 45 tests on the published snapshot
python3 scripts/verify_public_evidence.py
python3 scripts/plot_current_kld.py
python3 scripts/build_selective_h16_summary.py
```

Model weights, teacher logits, JIT caches, quantized checkpoints, raw calibration
records and unopened holdout inputs are deliberately not in Git. Their immutable
identities and replay requirements are documented instead.

## License and attribution

**Source-available, not OSI open source.** Distributed under the ShapleyMCG
License 1.0 in `LICENSE`. Required attribution:

> ShapleyMCG was created by Brandon M. Music. Canonical source:
> https://github.com/brandonmmusic-max/shapleymcg

Third-party models, runtimes and libraries keep their own licenses. See
`THIRD_PARTY_NOTICES.md`.

### Method lineage

QTIP (Tseng et al., NeurIPS 2024) → ExLlamaV3/EXL3 by turboderp → Luke Alonso's
KQuant/QSRT and the b12x W4A8 trellis path, with Martin Vit on the GLM-5.2 stack
→ the P8/TrellisMX endpoint here, which replaces the lookup table with the
procedural law, consumes a physical UE8M0/32 MX scale plane in `mxf8f6f4`, and
adds MoE small-M scheduling.

Error-feedback encoding follows GPTQ (Frantar et al., ICLR 2023) and the
LDLQ/incoherence analysis of QuIP and QuIP# (Chee et al. 2023; Tseng et al.
2024). Rotation experiments follow QuaRot (Ashkboos et al., 2024) and SpinQuant
(Liu et al., 2024). The MX scale plane follows the OCP Microscaling
specification (Rouhani et al., 2023) and NVIDIA NVFP4. Shapley allocation follows
Shapley (1953) and Castro, Gómez and Tejada (2009); CoopQ (Zhao et al., 2025) is
the closest published precedent and is cited as such. KLD is Kullback and Leibler
(1951); intervals are Efron's BCa (1987); sealed-role practice follows Nosek et
al. (PNAS 2018). Calibration descends from the REAP corpus family (Lasby et al.,
Cerebras, 2025). Full references are in `CITATIONS.md`.

## Related work

The earlier `glm52-sqg-mcg-experiments` repository answers a different
model/codec question. The canonical
[`shapleymcg`](https://github.com/brandonmmusic-max/shapleymcg) repository is the
method and license authority.
