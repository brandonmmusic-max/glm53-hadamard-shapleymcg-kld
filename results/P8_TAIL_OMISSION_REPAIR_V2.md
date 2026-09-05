# P8 fixed-order KPool tail repair V2

Decision: **incomplete-tail omission is a material cause of the prior P8
B12X/NVFP4 forced-decode regression.**

The corrected intervention emits all complete four-token pools, immediately
followed by the current request's `L mod 4` tail tokens, then `-1` padding. An
exhaustive device test of the actual Triton expander over `L=1..2047` proved
that only rows with `L mod 4 in {1,2,3}` changed: mod-4 counts were exactly
`{0:0,1:512,2:512,3:512}`. All 511 complete-pool rows were unchanged.

Every KLD row below used attention backend `B12X_MLA_SPARSE`, KV dtype
`nvfp4_ds_mla`, native P8 MoE (`mxf8f6f4`, N64 fused scratch), E4M3
activations, TP4/DCP1/no-EP, CUDA graphs on, MTP off, and exact 4.25 physical
bpw. The only changed variable was the tail selector.

| Window/domain | Baseline KLD | Tail-V2 KLD | Delta |
|---|---:|---:|---:|
| conditional-fit-0032 / general | 0.0974280 | 0.0774279 | -0.0200001 |
| conditional-fit-0021 / legal | 0.2015925 | 0.0461773 | -0.1554152 |
| conditional-fit-0090 / code | 0.0981727 | 0.0523391 | -0.0458336 |
| conditional-fit-0003 / reasoning | 0.0727031 | 0.0303301 | -0.0423730 |
| Equal-window mean | **0.1174741** | **0.0515686** | **-0.0659055** |

The mean improved by **56.1021%**, all four windows improved, and the paired
four-window percentile interval for candidate-minus-baseline was
`[-0.1271547,-0.0264585]`. This passes the preregistered material-support rule.
It is a four-window conditional-fit serving diagnostic, not full32 product
qualification and not a speed measurement.

The repaired B12X/NVFP4 endpoint is only `0.003177` above the independently
measured P8 FlashInfer/FP8 mean `0.048392` on the same four windows. Thus tail
omission explains most of the former approximately-0.11 regression; scale or
reader effects remain residual hypotheses.

Receipts:

- preregistration: `experiments/p8-tail-omission-repair-v2-prereg.json`
- sealed execution: `experiments/p8-tail-omission-repair-v2.json`
- evidence: `evidence/opened/codec-v2/p8-tail-omission-repair-v2/`
- raw logits: retained on NVMe; 5,072,629,760 bytes; nothing written to klcstore
