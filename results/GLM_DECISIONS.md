# GLM-5.3-Flash decision log

Times and numerical values in this file are backed by the JSON under
`evidence/`. Lower KLD is better. Conditional-fit was an adaptive development
role; selection waves were untouched until their candidates were frozen.

## 2026-09-03: uniform rotation

- The original KLD 2.93661 result was invalidated. A sparse overlay was loaded
  by the ordinary safetensors loader and redirected tensors were overwritten
  by later carrier shards.
- The corrected runtime uses a fail-closed InstantTensor overlay and records
  first-forward proofs for both `R_in` (gate/up input) and `R_mid` (down input).
- Fixed H16 on layer 3 improved conditional-fit KLD by 2.625% and passed the
  paired interval gate. Learned rotation improved only 0.590% against its
  stock anchor and lost directly to H16.
- Exact-Qwen H16 on all 42 routed layers improved mean conditional-fit KLD by
  1.326%, but its BCa interval crossed zero. Uniform H16 therefore failed.

## 2026-09-03: selective rotation

- Adaptive conditional-fit localized heterogeneous effects. Layers 3-9 and
  32-44 worsened KLD, while the middle region contained the positive rows.
- The best adaptive subset, layers 3/19/20, improved conditional-fit by 5.493%
  and passed there. It was frozen before selection wave 2, where it was 0.339%
  worse than stock and failed.
- The pre-ranked runner-up, layers 19/20, improved conditional-fit by 2.956%.
  It was frozen before selection wave 3, where it improved by only 0.407%; its
  BCa 95% interval `[-0.00110568, +0.00066555]` crossed zero, so it failed.
- All three V3 selection waves are now opened. No more V3 layer-subset tuning
  is permitted. There is no qualified GLM rotation win from this campaign.

## Interpretation boundary

- The calibration capture is not evaluation leakage or a substituted corpus.
  The Hub corpus receipt points to the exact local REAP JSONL SHA-256
  `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`,
  and the Hub tokenizer receipt matches the local GLM tokenizer.
- Qwen's 29-36% headline combined GPTQ, REAP calibration, and H16 against a
  shipped RTN expert-only control. Its same-calibration H16-only increment was
  about 6-15%, depending on panel.
- GLM's uniform identity-GPTQ V2 candidate was 5.05% worse than stock on its
  protected selection role. This measured control difference, plus the strong
  layer-sign heterogeneity, is the current evidence-based explanation for why
  the Qwen-sized gain did not transfer.
- Architecture is a plausible mechanism, not yet a proven cause: GLM has 288
  routed experts per layer and sigmoid/noaux top-8 routing with scaling 2.5,
  versus Qwen's smaller routed-expert system. Quantization errors can therefore
  change later routing decisions differently. A causal router-flip experiment
  is required to establish that mechanism.
