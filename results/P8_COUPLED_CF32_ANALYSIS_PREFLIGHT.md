# Coupled CF32 analysis preflight

1. Before any new three-arm result, implement the existing proposal's fixed
   equal-window analysis in `glm53_nvfp4/analyze_p8_coupled_cf32.py`.
   The manifest must contain exactly 32 unique IDs in four equal domains;
   stock, identity P8 and coupled P8 must each cover that exact set.
2. Primary KLD excludes the first prefill row via the caller's scored
   `true_decode_mean_kld`. Including-prefill KLD is secondary. This helper
   does not itself prove causal row masking, teacher identity or native
   dispatch: those remain capture/installation receipt requirements.
3. Preserve the proposal decision: mean coupled-minus-identity below zero is
   a development pass; paired BCa is nonblocking. Use the existing 20,000
   bootstrap replicates and seed 20260902, paired by manifest ID. Report
   stock difference, window wins and domain means without selecting layers.
   A constant delta distribution is explicitly flagged as degenerate.
4. Every arm/result carries attention, KV dtype, MoE backend, activation
   precision and bpw labels. P8 has twice the NVFP4 MMA issue count. CF32 is
   already-opened development data, not final qualification.

CPU synthetic tests: `python3 -m pytest -q tests/test_p8_coupled_cf32_analysis.py`
— 9 passed. Cases cover a nonblocking CI spanning zero, row permutation,
prefill-only improvement not satisfying the primary gate, constant deltas,
missing/duplicate rows, wrong domains, NaN/negative metrics, missing runtime
labels and unbalanced manifest. These are analysis tests, not KLD results.
No GPU or teacher logits were used by these tests.
