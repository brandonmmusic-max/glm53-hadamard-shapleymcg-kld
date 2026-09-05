# Rotation V8: shared middle-butterfly train32 search

Status: **adaptive fit search passed; `p00625` advances to tune32**.

V8 replaces the per-block selector with one compact SO(16) butterfly shared by
all 288 layer-3 experts and all down-projection K16 blocks. Gate/up remain the
stock NVFP4 carrier. Each down candidate is physically packed ModelOpt NVFP4 at
`4.500003814697266` payload bpw and the complete rotation table is 1,024 bytes.
The runtime applies the matrix after SwiGLU in BF16 on every TP4 rank; it is not
yet fused into the MMA prologue. LDLQ and BlockLDLQ are excluded.

## End-to-end teacher KLD

Every arm used the exact same 32 domain-balanced fit/train windows, all causal
rows, the same pinned runtime image, TP4/EP4/DCP4, Humming, BF16 activations,
FP8 MLA KV, eager execution, and no MTP.

| Arm | Angle | Mean KLD | Improvement vs stock |
| --- | ---: | ---: | ---: |
| Stock ModelOpt NVFP4 | n/a | 0.0364503459 | baseline |
| Matched GPTQ/identity | 0 | 0.0375109066 | -2.9096% |
| m003125 | -pi/32 | 0.0360164582 | +1.1904% |
| p003125 | +pi/32 | 0.0360568702 | +1.0795% |
| m00625 | -pi/16 | 0.0362063570 | +0.6694% |
| **p00625** | **+pi/16** | **0.0356555443** | **+2.1805%** |
| m0125 | -pi/8 | 0.0363996770 | +0.1390% |
| p0125 | +pi/8 | 0.0359050243 | +1.4961% |
| m0250 | -pi/4 | 0.0369560954 | -1.3875% |
| p0250 | +pi/4 | 0.0365740900 | -0.3395% |

For `p00625`, the mean delta versus stock is `-0.0007948015`; it wins 20/32
windows. Its paired BCa 95% interval is `[-0.00306204,+0.00105526]`. The
interval is a nonblocking development control after adaptive grid selection
and does not establish a qualified improvement. The same arm is 4.9462%
better than the matched zero-angle encoder, showing that the rotation more than
repays the down re-encoding penalty.

All four domain means are within the prospective tune guard versus stock. The
only numerically worse domain is legal by `+0.00000279`, far below the
`+0.0005` limit. The independent tune32 run remains decisive.

## Interpretation boundary

The local fitted reconstruction ratio was lowest near `abs(angle)=pi/8`, while
causal KLD selected `+pi/16`. This is evidence that the fitted Hessian remains
useful for constructing candidates but is not a sufficient end-to-end ranking
objective. It is not evidence to restart the project or discard rotation.

This is a 4.5-bpw native-NVFP4 rotation experiment, not the separate 4.25-bpw
P8 trellis product. It does not yet establish a fused prologue, speed, an
all-layer checkpoint, Shapley allocation, or protected-role quality. Selection,
confirmation, final, and the reserved confirmation logits remain unopened.

## Primary receipts

- `train32-analysis-v2.json`: `d8f470e22841417c21c41b108ef656c45ea35513a11831c1bc066c18fc451dc0`
- `train32-execution-v2.json`: `02b2cee4d47a7d567c3c2c08fbd4ba06c121f97624b6f516161824fd6088b994`
- Sealed execution plan: `ecd06ebc9959ba2c966fdfe2eb3f9954232ef9c93cf44fe13e126cff84ebba64`
- Verified 32-file teacher subset: `5a9ae6c070007dc1d2ac9dbf92fa194ab6ef938b95b10409d0474073ad5448c2`
