# Rotation v6: all-expert block-local H16

Status: **adaptive conditional-fit failure; do not promote**.

This experiment materialized layer 3 for all 288 routed experts with an
independently selected D-P-H16 transform for every projection and physical K16
block. Gate, up, and down were not forced to share transform descriptors. The
encoder used domain-balanced fit captures, a block-Hessian GPTQ objective,
static within-group activation order, two global-scale refits, and no LDLQ or
BlockLDLQ.

## Physical closure

- Logical weights: 7,247,757,312
- Physical payload: 4,077,051,264 bytes
- Physical rate: 4.500207265218099 bpw
- Descriptor payload: 184,320 bytes (one uint8 index per physical K16 block)
- Coverage: 288 experts x 3 projections
- Saved-payload verification: four of four expert slices passed bitwise BF16
  equality after NVFP4 decode and indexed inverse rotation
- Runtime boundary: the KLD measurement used BF16 effective weights
  `Q(W R) R.T`; a projection-specific compact native prologue was not tested

The first overlay startup exposed misuse of the reserved safetensors `format`
metadata key. `FULL_BUILD_PT.json` and `repair-receipts/` preserve the repair
lineage: the original files remain intact and each corrected file records a
byte-identical tensor-payload copy plus new full-file and header hashes.

## End-to-end BF16-teacher KLD

The final matched v3 run used the same 32 conditional-fit windows, image digest,
InstantTensor loader, TP4/EP4/DCP4 topology, Humming backend, BF16 activations,
FP8 MLA KV cache, eager execution, and no MTP for both arms.

| Arm | Mean KLD |
| --- | ---: |
| Stock ModelOpt NVFP4 | 0.03892642607132914 |
| Block-local H16 candidate | 0.03923395104120979 |

- Paired mean delta (candidate - stock): +0.000307524969880657
- Relative improvement: -0.7900159375462401% (candidate is worse)
- Paired BCa 95% interval: [-0.0007695032366529342,
  +0.0013890440843033775]
- Decision: fail; do not open protected selection

The candidate improved 13 of 32 windows. Post-hoc domain means of the paired
delta were:

| Domain | Windows | Mean candidate - stock KLD |
| --- | ---: | ---: |
| General | 8 | -0.00007642319309426188 |
| Legal | 8 | -0.00039484783991359466 |
| Code/agentic | 8 | +0.002172029605298759 |
| Reasoning/termination | 8 | -0.00047065869276827427 |

Thus three domain means favored the candidate, but the code/agentic regression
dominated the overall result. This breakdown is descriptive and post-hoc; it was
not used to alter or reroll v6.

This is a wrong-direction null, not evidence that the candidate improves KLD.
The conditional-fit32 panel was already-open adaptive data, so even a favorable
result would not have constituted protected qualification.

## Receipt hashes

| File | SHA-256 |
| --- | --- |
| `FULL_BUILD.json` | `e1075e2a0ff13a9514a656968d6a155702a11a4340aab4f01a97e144a3d7496f` |
| `FULL_BUILD_PT.json` | `4258943a493cc4a7091dbba75e35e214f593ea62625b1d9cf720db4f26990d35` |
| `cf32-v3-paired-analysis.json` | `0e9be417290750e95dfa5769de35b5fd57daf1711d3a0f4e70b780a6570072ac` |
| `run-stock-cf32-v3.json` | `830132822d7f341eff1caa9e5ef863ed6b0fbf29e3740dfc29fa255d392db447` |
| `run-candidate-cf32-v3.json` | `155fb19dff42d0ce31c81b808200dae1e204198692bd90ae8536e5f6cdcae15f` |

The v1 candidate failed before scoring because external chunk symlinks were not
mounted in the container. V2 fixed that mount and then failed before scoring on
the reserved metadata key. Both failures opened zero candidate windows. V3
fixed both infrastructure defects and completed all 32 windows in each arm.

## Next adaptive hypothesis

Rotation v7 uses the same one-byte descriptor budget but assigns identity to
index 0 and 255 procedural H16 transforms to the remaining entries. This lets
each expert/projection/K16 block decline rotation when no H16 candidate beats
identity under the fitted block-Hessian objective. Its plan was frozen only
after this v6 failure and remains adaptive conditional-fit work.
