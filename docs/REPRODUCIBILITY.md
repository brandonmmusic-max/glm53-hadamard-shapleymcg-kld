# Reproducibility record

## Exact identities

| Component | Immutable identity |
|---|---|
| Implementation snapshot before public documentation | `15d6a2cb3f7145fb7d08a6fd66fba9407270380a` |
| Qwen reference implementation | `2207fbc769023444770c3e4a3c40d445abf7b7fe` |
| GLM-5.3-Flash BF16 source | `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` |
| Teacher logits | `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa` |
| Stock NVFP4 carrier | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Base runtime image | `sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe` |
| H16/MXFP6 v80 image | `sha256:8e6856039c449aa202f113c3834af09e4e1dcb4c9c9b7e7eb009608f3772627a` |
| Role manifest SHA-256 | `32c2f068c78ab54190ab930b84f984e22193321d150c5d87a1231332e6fbfae8` |
| Hardware | 4 x NVIDIA RTX PRO 6000 Blackwell 96 GB, SM120, no NVLink |
| Runtime regime | TP4 / EP4 / DCP4, BF16 activations, packed NVFP4 routed weights, FP8 MLA KV, no MTP |

The v80 Dockerfile pins its parent digest and installs the exact B12X plugin,
FP6 serving binding, and dynamic CuTe kernel used by the mixed 6-bpw campaign.
Launchers independently compare the resolved image ID with the expected ID and
disable the unmodified FP6 micro-kernel path.

## Statistical estimand

KLD is `KL(teacher || student)` evaluated per token and summarized as an
equal-window mean. Candidate-versus-control uncertainty uses a paired
window-level bootstrap with 20,000 replicates and seed `20260902`. A rotation
passes the conditional-fit gate only if both the mean delta and the upper bound
of the BCa 95% interval are below zero.

## Current replay

The paths below are the exact local campaign paths. The code currently assumes
this workstation layout; use bind mounts or edit the path constants when
reproducing elsewhere.

```bash
cd /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53

# Verify code and the published evidence snapshot.
python3 -m pytest -q
python3 scripts/verify_public_evidence.py

# Build all 42 routed layers using the exact Qwen H16 recipe.
./scripts/run_qwen_exact_had16_full_v9.sh

# Run conditional-fit, open a new selection wave only after passing, and freeze.
./scripts/run_qwen_exact_h16_qualification_v10.sh

# After the H16 winner receipt exists, run the 6-bpw Shapley campaign.
./scripts/run_shapley_6bpw_campaign_v10.sh
```

The quantizer uses 16-column blocks, shared `R_in` for gate/up, separate
`R_mid` for down, full `R^T H R`, static activation order within groups,
sequential GPTQ across 128-column slabs, and a cap of 256 routed samples per
expert for the exact-Qwen scale-up.

## Published evidence boundary

Included now:

- opened layer-3 H16, learned, identity, stock, and 256-versus-10k paired KLD;
- quantization receipts and layer validation for the exact 256-sample build;
- source, runtime patch, Dockerfile, launchers, tests, and analysis code;
- a checksum manifest and reproducibly generated figure.

Excluded from Git:

- BF16 and quantized weight tensors;
- raw teacher logits and token arrays;
- raw calibration JSONL and activation/Hessian captures;
- JIT caches and Docker layers;
- unopened selection and confirmation inputs or outputs;
- credentials and Hugging Face/GitHub tokens.

Those exclusions prevent accidental disclosure, avoid redistributing
third-party weights, and preserve the sealed evaluation boundary. The final
snapshot can append newly opened results and receipts without rewriting this
history.

## Known validity threats

- Current GLM evidence changes one routed layer, not all 42.
- The 16-window conditional-fit panel estimates within-panel uncertainty but
  not between-quantization-run variance.
- The serving endpoint is W4A16. Native FP4 activation input was not qualified
  in the pinned Humming runtime.
- MoE routing changes make layer effects non-additive, so the one-layer gain
  cannot be multiplied by 42.
- Learned rotations optimized a calibration objective and did not generalize
  better than fixed H16 on the opened KLD panel.
