# Matched forced-decode control V2

Decision: **the approximately 0.11 KLD endpoint is shared by stock NVFP4,
production EXL3, and TrellisMX-P8. The codec is exonerated as its cause, and a
shared production serving-path defect is supported.**

This was the predeclared four-window, one-per-domain diagnostic. All arms used
the same forced-token, pre-mask FP32 capture protocol, TP4/EP4/DCP4,
`B12X_MLA_SPARSE`, NVFP4 MLA KV, CUDA graphs, and MTP off. Stock used its
clamp-correct Humming MoE backend; EXL3 used its production B12X MoE backend.

| Arm | All-row mean KLD | True-decode mean KLD | Relative to P8 |
|---|---:|---:|---:|
| TrellisMX-P8 frozen reference | 0.1174741 | not re-estimated | reference |
| Stock NVFP4 | 0.1250571 | 0.1230734 | 6.455% worse |
| Production EXL3 | 0.1128498 | 0.1109032 | 3.936% better |

Both controls are inside the preregistered `[0.09, 0.15]` interval. Neither
improves at least 20% over P8, so both frozen rules pass. This identifies the
high endpoint as a serving/capture-path problem shared by production baselines;
it does not identify the exact faulty component and does not replace the
full-32-window quality estimate.

The eight raw captures occupy exactly 10,145,259,520 bytes on
`/media/brandonmusic/nvme1n1p3`. No raw data was written to 96%-full
`klcstore`. The repository snapshot includes terminal execution, aggregate and
per-window JSON receipts; raw logits remain external and hash-bound.

Receipts:

- plan: `experiments/decode-path-matched-control-v2.json`
- pre-registration: `experiments/decode-path-matched-control-v1-prereg.json`
- amendment: `experiments/decode-path-matched-control-v2-amendment.json`
- evidence: `evidence/opened/codec-v2/decode-path-matched-control-v2/`
