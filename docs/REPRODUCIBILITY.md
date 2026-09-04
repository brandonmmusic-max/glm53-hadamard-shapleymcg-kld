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
equal-window mean. Candidate-versus-control uncertainty uses paired
window-level BCa bootstrap intervals. Legacy H16 plans use 20,000 replicates;
the codec-v2 plans use 50,000 and record their seed, multiplicity correction,
minimum-effect rule, and role boundary in the immutable analysis plan written
before each run. Numerical gates compare BF16 pseudoquant candidate and BF16
decoded-GPTQ control overlays through the identical InstantTensor/framework
execution path.

As of decision 12, paired window uncertainty is a recorded development
control rather than an engineering stop. An interval crossing zero may permit
continued codec and kernel work, but it may not support a comparative quality
claim. A final statement that P8/P4 beats NVFP4 still requires matched
end-to-end teacher KLD with the paired BCa upper bound below zero. Shapley is
used only after a viable base codec is identified and cannot substitute for
that comparison.

For the layer-22 gate, both arms redirect the same 864 routed-expert tensors
in exactly one layer. The control is a BF16 decode of full-Hessian GPTQ
NVFP4; the candidate is the BF16 reference decode of the 4.25-bpw P8 stream.
The analysis plan was written before either n=32 run and is identified by
SHA-256 `4499f08c8b653a0ac5ac47e036ec2f1033cf3bf8fcc4fe8325c295a7b708ea97`.

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

# Recheck bit-exact P4/P8 reference codec and endpoint export behavior.
python3 -m pytest -q \
  tests/test_trellis_mxf.py \
  tests/test_trellis_nvfp4.py \
  tests/test_output_aware.py \
  tests/test_export_native_endpoint.py \
  tests/test_endpoint_codec.py
```

The quantizer uses 16-column blocks, shared `R_in` for gate/up, separate
`R_mid` for down, full `R^T H R`, static activation order within groups,
sequential GPTQ across 128-column slabs, and a cap of 256 routed samples per
expert for the exact-Qwen scale-up.

The codec-v2 P8 encoder instead preserves physical 16x16 trellis order,
decodes a K4 stream to E4M3 with physical K32 UE8M0 scales, applies static
within-group activation order, and carries full-Hessian GPTQ-style error
feedback between groups while jointly refitting scales. It is 4.25 physical
bpw and charged 4.5 bpw in the KLD gates. No LDLQ or BlockLDLQ implementation
or result is used. P8 targets `mxf8f6f4`, which has twice the MMA issue count
of NVFP4; it is a quality product, not an NVFP4-speed claim.

## Published evidence boundary

Included now:

- opened layer-3 H16, learned, identity, stock, and 256-versus-10k paired KLD;
- all 42 layers of exact-256 H16 quantization receipts and validation, plus the
  completed full-model conditional-fit KLD and runtime proof;
- source, runtime patch, Dockerfile, launchers, tests, and analysis code;
- a checksum manifest and reproducibly generated figure.

The live klcstore campaign additionally contains the codec-v2 compressed
streams, dense BF16 reconstructions, matched decoded-GPTQ overlays, per-window
Parquet records, preregistered plans, and SHA-256 receipts. Weight-bearing
artifacts remain excluded from Git; the next evidence snapshot should include
only plans, analyses, manifests, and hashes.

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

- The uniform all-42-layer H16 candidate has been measured, but its 1.326%
  lower mean KLD did not pass because the paired BCa interval crossed zero.
- Only layer 3 has a statistically qualified H16 improvement; selective
  multi-layer placement remains adaptive conditional-fit work.
- The 16-window conditional-fit panel estimates within-panel uncertainty but
  not between-quantization-run variance.
- The serving endpoint is W4A16. Native FP4 activation input was not qualified
  in the pinned Humming runtime.
- MoE routing changes make layer effects non-additive, so the one-layer gain
  cannot be multiplied by 42.
- Learned rotations optimized a calibration objective and did not generalize
  better than fixed H16 on the opened KLD panel.
- The first P8 KLD comparison mixed a BF16 candidate path with a packed-NVFP4
  Humming control and is invalid for encoder attribution. The corrected
  matched-BF16 layer attribution found conditional-fit improvements at layers
  3 and 20, but their frozen combination improved the reused V5 wave by only
  1.383% with a BCa interval crossing zero and opposite signs across domains.
- The earlier integrated W6A8 activation path regressed layer-3 KLD even with
  near-lossless weights. The corrected monolithic P8 prologue closes on eight
  actual full-size layer-3 expert payloads (cosine 0.998953, relative L2
  0.045753). The split/prefill specialization now also closes for K3/K4
  synthetic payloads and for the actual full-size K4 layer-3 payload at M33 and
  M64 (real-payload cosine 0.998940–0.998948). The former failure was an
  omitted mandatory K32 MMA lane permutation in one materialization branch;
  it remains preserved. The physical TP4 kernel subsequently closed against
  decoded pseudoquant on all 32 opened conditional-fit windows: mean delta
  `+0.0003952`, BCa 95% CI `[-0.0018896,+0.0029882]`, passing the relaxed
  absolute-mean engineering margin of `0.0014`. This is an end-to-end device
  closure, not evidence that P8 beats NVFP4. The current correctness run uses
  the materialized dynamic arm for small M because the first E=288 small-M
  diagnostic emitted non-finite values.
  Independently, the
  strongest new layer-22 candidate improved causal
  full-expert NMSE by 17.07% but regressed matched-path end-to-end KLD by
  1.074%, with its paired interval crossing zero. Local routed-output NMSE is
  therefore not sufficient promotion evidence for the current encoder.
  P4 remains blocked by its exact-E2M1 encoder screen. No native-kernel quality
  or speed claim is made.
