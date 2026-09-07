# Latest measured coupled P8 checkpoint

Documentation updated September 7, 2026; measurement from September 6.
No new GPU measurement was performed for this documentation update.

## Result

**Mean true-decode KL(teacher || student): 0.03418114591027796.**
Window BCa 95% interval: **[0.02914835179457108, 0.040978425681327625]**.
Mean including the prefill row: **0.03531762204580294**. These are distinct
metrics; do not interchange them.

The checkpoint covers all 42 routed layers (3–44), with 17 at K5 and 25 at
K4. Boundary: `coupled-h512-h128-suh-svh-v1`. Four TP rank files per layer.
Routed stored rate including metadata: **4.658741764290623 bpw**.
Routed file bytes: **177,269,057,440**; this is not the entire served model size.

| Condition | Recorded value |
| --- | --- |
| Attention | B12X_MLA_SPARSE |
| KV cache | nvfp4_ds_mla |
| MoE | native-p8-mxf8f6f4-n128-coupled-h512-h128-suh-svh-v1 |
| Activation precision | E4M3, UE8M0 K32 at native P8 MMA boundaries |
| Stored rates | K4/K5 |
| Image | `sha256:c121590d3371b406c1e456076b1fc9cf9b596aa425b4401c692d142cbc425cd4` |
| Runtime manifest SHA256 | `06e6d21c6c86b81a680c56e6aee50b7351711866ffe4c8d932f6b5372560f4f1` |

Do not infer the later MTP3/graph speed-run configuration from this quality
receipt. Only conditions established by the measurement artifacts are claimed.
P8 decodes to E4M3 tensor-core MMA and uses twice NVFP4's MMA issue count for
equal dimensions; this is not native FP4-speed-class arithmetic.

## Protocol and comparison

The analysis has 32 already-opened conditional-fit windows, balanced eight
per domain, and 2,047 true-decode positions per window. Row 0 (prefill) is
excluded from true-decode KLD. The campaign report documents stored BF16
teacher logits and CPU FP64 KL accumulation. The analysis records a
window-level BCa bootstrap with 20,000 resamples and seed 20260902.

Uniform coupled K4 measured **0.036967452439595275**, at 4.2539798595 routed
stored bpw. Upgrades-only improves the mean by **0.0027863065293173145**,
or **7.54%**, and wins 25/32 windows. The campaign report records the paired
percentile-bootstrap interval **[-0.004723, -0.000961]** (20,000 resamples,
seed 20260906). This difference interval is not the individual-arm BCa interval.

This is a larger-bit-budget checkpoint, not a same-size improvement. The
uniform and upgrades-only runtime image revisions differ; the comparison
is not a single-variable attribution to bitrate alone. The raw analysis
explicitly states: **already-opened CF32 development evidence, not final
qualification**. No protected holdout was opened for this update. No matched
stock-NVFP4 superiority or later optimized-runtime KLD is established here.

## Receipts

- [Upgrades-only analysis: all 32 window scores and BCa interval](../evidence/p8-full-coupled-campaign-20260906/phase1/k5-only-cf32-analysis.json)
- [Checkpoint manifest: exact layer allocation and stored bytes](../evidence/p8-full-coupled-campaign-20260906/phase1/k5-only-checkpoint-manifest.json)
- [Pre-execution seal: image, runtime manifest and window order](../evidence/p8-full-coupled-campaign-20260906/phase1/k5-only-cf32-seal.json)
- [Uniform coupled K4 analysis](../evidence/p8-full-coupled-campaign-20260906/uniform-coupled-cf32-analysis.json)
- [Three-arm rate-swap evidence](P8_RATE_SWAP_EVIDENCE_V1.json)
- [Full historical campaign report, measured results in section 9](P8_FULL_COUPLED_CAMPAIGN_RESULTS_20260906.md#9-phase-1-measured-results)

## Why this was missing from the default page

Before this update, default `main` was `b607528b5a657882e546bb6f41f29175cebeacc0`.
The pushed `fable-p8-full-coupled-v1` branch at
`6c7664a96809c2f4af1fedf6c26b7152e47fac0a` was 25 commits ahead and seven
behind it. The measurements were published on that branch by
`414e86bab834881acf69cb296716d8ef23c5000a`; the rate-swap table followed in
`18fe026af453e88c3bb0d4651e90fe0a869bbb05`. They had not landed on `main`.

This update brings only documentation and existing scrubbed receipts to
the current default branch, preserving its separate cleanup commits. It
does not merge the divergent runtime/encoder branch or upload encoding scripts.
