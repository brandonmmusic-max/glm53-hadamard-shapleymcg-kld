# Tail V2 CF32 repeat-1 paired result

Recorded 2026-09-05. Status: provisional controlled product comparison.
Both repeat-1 slots completed successfully. The five-cold determinism gate
and capture-disabled speed campaign are incomplete. These measurements do
not qualify the coupled-scale candidate, which has not run on device.

Common conditions: B12X_MLA_SPARSE attention, nvfp4_ds_mla KV, tail V2,
TP4, CUDA graphs, MTP disabled. Conditional-fit has 32 windows, eight per
domain. P8 uses noEP/DCP1 and EXL3 uses EP4/DCP4; this is a product comparison.

| Arm | MoE backend | MoE activation | Stored payload bpw | Mean KLD including first prefill row | True-decode KLD |
|---|---|---|---:|---:|---:|
| EXL3 | B12X EXL3 full-expert | BF16 | 4.0 | 0.033492298040408444 | 0.031611840268931456 |
| P8 identity | native P8 mxf8f6f4 N64 fused scratch | E4M3, UE8M0/32 | 4.25 | 0.038271610648036664 | 0.03730955956731441 |

Primary protocol metric P8 minus EXL3: +0.004779312607628223
(+14.2698856% relative); paired 95% BCa interval
[+0.0001459667599048755, +0.010449762511746886].

True-decode diagnostic P8 minus EXL3: +0.005697719298382958
(+18.0240038% relative); paired 95% BCa interval
[+0.0013815042390767573, +0.011103104189582385]. The true-decode-only
interval is a supplemental diagnostic; the sealed runner uses mean_kld.

Both calculations use manifest window order, NumPy default_rng seed 20260902,
20,000 paired window bootstrap samples, and the existing
glm53_nvfp4.paired_role_analysis.bca_mean_interval. The earlier conversational
quick calculation sorted IDs, producing different finite bootstrap draws and
slightly different endpoints. Means and conclusions are unchanged; the
manifest-order values above are the reproducible provisional values.

Evidence root:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/tail-v2-p8-exl3-cf32-product-validation-v1b/quality`

| Evidence | SHA-256 |
|---|---|
| Sealed v1b plan | d055d4db3563fa112326979c70da9ddc058abd5de6f67213aeee3c7308820beb |
| round-01-exl3/execution.json | 406ce1fa0a00404e049f1da84a93f173d62fc1f835474dbd084f6eb2ffeeffc8 |
| round-01-p8/execution.json | 6bdc7a314ffc09a45a5a742aeb209d2893288e70a9ac5b0c458d533865c8abaa |

Per-window scores and retirement receipts reside under scores/repeat-01/ARM.
The excluded v1a EXL3 attempt has 0/32 matching raw hashes versus v1b and
a v1b-minus-v1a true-decode mean shift of +0.0004534678078313516.
This is an observed cross-start discrepancy, not a diagnosis. Scheduled
v1b cold repetitions determine the formal bitwise gate. No v1a score is
included in the paired v1b estimates.

P8 uses mxf8f6f4, with twice the MMA issue count of NVFP4. No throughput
or coupled-scale improvement is established by this result.

## Cold-repeat checkpoint

P8 repeat 2 completed all 32 windows with exit code 0. Every raw-logit SHA
matches its repeat-1 counterpart. Conditions are the P8 identity row above:
B12X_MLA_SPARSE / nvfp4_ds_mla / native P8 mxf8f6f4 N64 / E4M3 activation /
4.25 bpw, TP4/noEP/DCP1, graphs on, MTP off, tail V2.

`round-02-p8/execution.json` SHA-256:
`aa9d7fe6227f0104a7cbd23f69ce4ac389489869018cba4a1569e6b53c03821e`.
This establishes a full-panel two-start match for P8. The remaining three
P8 repeats and EXL3 repeats are still needed for the five-start gate.
