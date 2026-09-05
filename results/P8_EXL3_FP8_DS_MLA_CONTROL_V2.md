# P8 and EXL3 FP8-DS-MLA forced-decode control V2

Decision: **the approximately 0.11 forced-decode endpoint does not persist on
the established FP8-compatible serving path in either product. The shared
B12X/NVFP4 production MLA stack is strongly implicated; the codec is
exonerated as the shared cause.**

| Product | B12X/NVFP4 KLD | FlashInfer/FP8 KLD | Relative decrease | Wins | Paired delta 95% CI |
|---|---:|---:|---:|---:|---:|
| Native TrellisMX-P8, exact 4.25 payload bpw | 0.1174741 | 0.0483920 | 58.81% | 4/4 | [-0.13010, -0.03021] |
| Production EXL3, 4.0 stored bpw | 0.1128498 | 0.0411688 | 63.52% | 4/4 | [-0.11892, -0.04414] |

The corresponding true-decode-only means were `0.0467729` for P8 and
`0.0397559` for EXL3. Both products passed the predeclared rule of at least a
20% reduction and an FP8-path mean below `0.09`; every domain window improved.
These are MTP-off results.

V1 failed safely before capture because `B12X_MLA_SPARSE` explicitly rejects
GLM's RoPE-less head size 512 with `fp8_ds_mla`. The sealed V2 amendment used
the established `FLASHINFER_MLA_SPARSE_SM120` FP8 path for both products while
retaining P8's TP4/DCP1/no-EP/native-N64/fused-scratch topology and EXL3's
TP4/DCP4/EP4/B12X-MoE production topology. Therefore this evidence identifies
the combined attention-plus-KV serving path, not KV dtype in isolation. It is
a four-window diagnostic, not a full32 quality or speed qualification.

The eight raw captures occupy exactly 10,145,259,520 bytes on the fast NVMe;
none were written to 96%-full `klcstore`. The repository snapshot contains the
terminal execution, aggregate analysis, and all per-window JSON receipts. Raw
logits and private runtime logs remain local and hash-bound.

Receipts:

- plan: `experiments/p8-exl3-fp8-ds-mla-forced-decode-v2.json`
- original preregistration: `experiments/p8-exl3-fp8-ds-mla-forced-decode-v1-prereg.json`
- compatibility amendment: `experiments/p8-exl3-fp8-ds-mla-forced-decode-v2-amendment.json`
- snapshot: `evidence/opened/codec-v2/p8-exl3-fp8-ds-mla-control-v2/`
