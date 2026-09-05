# Rotation v7: identity-inclusive selective H16

Status: **adaptive conditional-fit failure; do not promote**.

V7 tested the most direct correction to V6's forced-rotation design. Each
expert/projection/physical-K16 block selected from one identity entry and 255
procedural D-P-H16 transforms. The descriptor remained one uint8 per block, so
the physical rate stayed 4.500207265218099 bpw. Gate, up, and down selections
were independent. LDLQ and BlockLDLQ were excluded.

## Learned selection pattern

| Projection | Identity blocks | H16 blocks | Mean fitted gain vs identity |
| --- | ---: | ---: | ---: |
| Gate | 96.5875% | 3.4125% | 0.0488% |
| Up | 95.7845% | 4.2155% | 0.0586% |
| Down | 7.9915% | 92.0085% | 31.4390% |

This is strong evidence that the fitted block-Hessian objective assigns nearly
all apparent rotation value to the down projection. The end-to-end result below
shows that this large surrogate gain did not transfer to KLD.

## Physical closure

- 288 experts x 3 independently selected projections
- 7,247,757,312 logical weights
- 4,077,051,264 physical bytes
- 184,320 descriptor bytes
- Four of four saved-payload slices passed bitwise BF16 equality after physical
  NVFP4 decode and indexed inverse rotation
- All generated safetensors carried `format=pt` and the separate
  `weight_format=ModelOpt NVFP4 E2M1/E4M3-per-16/FP32-global` metadata

## End-to-end BF16-teacher KLD

Both arms used the exact same 32 conditional-fit windows, image digest,
InstantTensor loader, TP4/EP4/DCP4 topology, Humming backend, BF16 activations,
FP8 MLA KV cache, eager execution, and no MTP.

| Arm | Mean KLD |
| --- | ---: |
| Stock ModelOpt NVFP4 | 0.03845049456035629 |
| Selective H16 candidate | 0.039266882350162284 |

- Paired mean delta (candidate - stock): +0.0008163877898059943
- Relative improvement: -2.1232179173261256% (candidate is worse)
- Paired BCa 95% interval: [-0.00025460288280870953,
  +0.002047754258027317]
- Windows improved: 13/32
- Decision: fail; do not open protected selection

Post-hoc domain mean paired deltas were approximately 0.0000000 general,
+0.0000658 legal, +0.0004099 code/agentic, and +0.0027899
reasoning/termination. This breakdown is descriptive and was not used to alter
or reroll V7.

## Receipt hashes

| File | SHA-256 |
| --- | --- |
| `FULL_BUILD.json` | `4dbea03aa83f6e9b4dd36a4b7b539a29d98283a541d744acff51c8067c80cd87` |
| `cf32-execution.json` | `1376910ab55d26ba66c1e83bba59dfd51c4772baeb6259a6487e6bbb24fd2b5c` |
| `cf32-paired-analysis.json` | `fc19204f2abdaeb037460cd0afefe0381b7ba4bcddf2158cb79d5f4e278beb72` |
| `run-stock-cf32-v1.json` | `2a97792f122d1fc25d1b7ee6edd45fc2d96a75e2c0811640bab1e0a3ccebdece` |
| `run-candidate-cf32-v1.json` | `f78ca87c4feb4f4f93da5e010e7aec93a0cd3ae2e8a696d3c2c29e50237f2dea` |

The result is adaptive evidence only because conditional-fit32 had already been
opened. It does not qualify a learned law, compact runtime prologue, or final
model claim.
