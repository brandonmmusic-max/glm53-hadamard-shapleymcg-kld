# Rotation V8: shared middle-butterfly train32 search

Status: **fit/train32 selected `p00625`; the disjoint tune32 gate failed**.

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

All four train32 domain means were within the prospective tune guard versus
stock. The independent tune32 result below is the authoritative transport
check.

## Disjoint tune32 result

| Arm | Mean KLD | Delta vs stock | Relative change vs stock |
| --- | ---: | ---: | ---: |
| Fresh stock ModelOpt NVFP4 | 0.0363378694 | baseline | baseline |
| Matched GPTQ/identity | 0.0364762391 | +0.0001383697 | +0.3808% worse |
| `p00625` | 0.0371841305 | +0.0008462611 | +2.3289% worse |

`p00625` still wins 20/32 windows, but the paired BCa interval versus stock is
`[-0.00110295,+0.00468556]` and both general (`+0.00308741`) and legal
(`+0.00292800`) violate the prospective `+0.0005` domain guard. It fails all
four registered checks. Thus the train improvement did not transport as a
standalone 4.5-bpw NVFP4 rotation claim.

All three KLD manifests completed. After the candidate manifest was written,
the wrapper exited 127 because `run_kld_v3.sh` was edited while its long-lived
shell was still reading it. No KLD row was rerun or modified. The preserved
final server log was copied bit-for-bit to the expected rotation-log filename,
the existing runtime verifier passed all four ranks, and the frozen analyzer
was run once over the completed manifests. `tune32-execution-v1.json` records
that recovery and every relevant hash.

## Interpretation boundary

The local fitted reconstruction ratio was lowest near `abs(angle)=pi/8`, while
train32 causal KLD selected `+pi/16`, and tune32 reversed that result. This is
evidence that neither fitted Hessian error nor one opened 32-window split is a
sufficient end-to-end ranking objective. It does not establish a rotation
quality gain.

Decision 21 was committed before tune closed: the fixed `+pi/16` arm receives
one combined 4.25-bpw TrellisMX-P8 interaction test regardless of this outcome.
That test asks whether the rotation is useful inside the codec, not whether
the standalone 4.5-bpw result transported. No replacement angle or reroll is
allowed.

This is a 4.5-bpw native-NVFP4 rotation experiment, not the separate 4.25-bpw
P8 trellis product. It does not yet establish a fused prologue, speed, an
all-layer checkpoint, Shapley allocation, or protected-role quality. Selection,
confirmation, final, and the reserved confirmation logits remain unopened.

## Primary receipts

- `train32-analysis-v2.json`: `d8f470e22841417c21c41b108ef656c45ea35513a11831c1bc066c18fc451dc0`
- `train32-execution-v2.json`: `02b2cee4d47a7d567c3c2c08fbd4ba06c121f97624b6f516161824fd6088b994`
- Sealed execution plan: `ecd06ebc9959ba2c966fdfe2eb3f9954232ef9c93cf44fe13e126cff84ebba64`
- Verified 32-file teacher subset: `5a9ae6c070007dc1d2ac9dbf92fa194ab6ef938b95b10409d0474073ad5448c2`
- `tune32-analysis-v1.json`: `105c54116f1230225c1cdb26714497a456880c21de866be63d26526446df467c`
- `tune32-execution-v1.json`: `86c4b9958135e409284e2061f60817ff1bd5322549c811149f00b4636dfa6d65`
