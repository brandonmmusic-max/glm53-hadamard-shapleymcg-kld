# Full 42-layer coupled TrellisMX P8: build, fidelity protocol and measured results

Status 2026-09-06 08:30 EDT. Phase 0 is complete and measured. Phase 1 (same-size K3/K4/K5
reallocation) is running; its measured result will be appended, not substituted for a prediction.

Everything below is read from a receipt on the campaign volume. Where a number is a prediction
from a proxy rather than a measurement it says so in the same sentence.

## 1. What was built

A coupled TrellisMX P8 checkpoint covering **every routed layer, 3 through 44**, replacing the
earlier three-layer pilot. Each layer is stored as four tensor-parallel rank files.

| Quantity | Value |
|---|---:|
| Routed layers | 42 |
| Rank files | 168 |
| Logical routed weights | 304,405,807,104 |
| Weight payload | 161,715,585,024 B (4.25 bpw exactly) |
| Coupled metadata | 151,436,544 B (0.00398 bpw) |
| safetensors headers | 551,040 B |
| Total file bytes | 161,867,572,608 |
| Stored rate including metadata | 4.253979859528719 bpw |
| Manifest sha256 | `2b81f12f1730b12ed66a21407c81d281b939a4efa7bbfe46efacb06e34033978` |

Boundary: `coupled-h512-h128-suh-svh-v1`, that is an H512 input Hadamard, the exact EXL3 suh/svh
scales, an H128 output Hadamard, sign draw 0 and a SiLU cap of 10. Transform predeclaration
sha256 `093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.

Two encoder designs are pinned in the manifest, and the runtime accepts only these two hashes:

| Design sha256 | Layers |
|---|---:|
| `23fba55049279860d0038583660ecad6625885df08f6a521217ce9e8fe71aa74` | 39 |
| `4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695` | 3 |

The three layers on the second design (3, 20 and 22) were encoded during the pilot and are
hard-linked rather than re-encoded; their reuse receipts record the source directory and the
sha256 of every linked file.

### Byte constraint that governs Phase 1

The coupled checkpoint is **151,865,280 B larger** than the identity K4 checkpoint files
(161,715,707,328 B), because the coupled metadata is real signed scale data and is not
sign-packable. At layer granularity one rate step is 905,969,664 B, so holding the identity
budget requires at least one more K3 layer than K5 layers. The solver derives this rather than
assuming it: the maximum admissible net offset `#K5 - #K3` is **-1**.

## 2. Fidelity protocol

Quality is measured with the sealed CF32 conditional-fit protocol, unchanged from earlier work:

* 32 windows, roles sha256 `b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14`,
  8 windows in each of four domains;
* 2,047 forced-decode rows per window; row 0 is excluded from the true-decode metric because it
  is the prefill position;
* teacher logits are the stored BF16 reference; KL(teacher || student) is accumulated in FP64 on
  CPU, never on device;
* the window interval is a bias-corrected and accelerated bootstrap over windows, B = 20,000,
  seed 20260902;
* raw logit captures are streamed **one window at a time** (1,268,157,440 B each) and each is
  unlinked only after its score and hash receipt are durable, so the protocol never needs the
  dense 32-window capture on disk.

Execution is sealed before it runs. The preparer writes a runtime manifest binding the image id,
the checkpoint manifest, every sidecar hash, the design hashes, the transform, the roles and the
teacher root; the executor authenticates that manifest, writes a seal, and refuses to run if the
window order, arm order or image id differs from the seal. For this run:

| Artifact | sha256 |
|---|---|
| Runtime manifest | `ba431792ca46dd4f613f1ee74f133e3699c4766b509a6a5b7c9889894308e32b` |
| Execution seal | `2b027c2495f2f0f88e9f3cf256af850d0afc153d5f03ac80f3b241ed779c1d42` |
| Analysis | `cf12e503eca9491d792a1fb9b30341e97f13938870be95f07218f61294619213` |

The runtime log gate additionally verifies all 168 layer/rank pairs loaded, the per-layer design
hash and the per-layer stored rate, so a checkpoint cannot be measured under a runtime that
silently substituted anything.

### Runtime conditions actually recorded

| Condition | Value |
|---|---|
| Image | `sha256:859d180f14740ef70ef452e1a07a57895bd87500c41559ac5cffb1708dab5fbb` |
| Attention | B12X_MLA_SPARSE |
| KV cache | nvfp4_ds_mla |
| MoE backend | native-p8-mxf8f6f4-n128-coupled-h512-h128-suh-svh-v1 |
| Activations | E4M3, UE8M0 K32 scales, at native P8 MMA boundaries |
| Layers | 3-44, stored rate K4 |

Production was off for the entire campaign: the backend service and the model-stack timer stayed
inactive and port 8000 stayed unbound.

## 3. Measured result: uniform coupled K4

| Metric | Value |
|---|---:|
| True-decode mean KLD | **0.036967452439595275** |
| Mean including row 0 | 0.037951788543825330 |
| Window BCa 95% | 0.031308303958108 to 0.044597445871382 |
| Windows | 32 |

Per domain:

| Domain | KLD |
|---|---:|
| axis1_general | 0.037460 |
| axis2_legal | 0.053656 |
| axis3_code_agentic | 0.033627 |
| axis4_reasoning_termination | 0.023126 |

### Paired against prior arms on the identical 32 windows

Both reference arms were measured with the **same KV cache (nvfp4_ds_mla) and the same attention
backend**, so the comparison is not confounded by the cache. They ran under the tail-repair v2
image rather than the v10 coupled image, so this is a matched-window comparison, not a
matched-condition one, and it is reported as descriptive rather than as a preregistered contrast.
The reference values are the mean of the three product-validation repeats.

| Comparison | Mean | Paired difference | 95% interval | Coupled better in |
|---|---:|---:|---|---:|
| Coupled K4 (this run) | 0.036967 | — | — | — |
| Identity K4 P8 | 0.037310 | -0.000342 (-0.92%) | -0.003438 to +0.002154 | 17 of 32 |
| EXL3 4.0 bpw | 0.031300 | +0.005667 (+18.1%) | +0.001882 to +0.009357 | 9 of 32 |

Interval is a percentile bootstrap of the paired per-window difference, 20,000 resamples,
seed 20260906.

**Reading.** Against the identity P8 the interval spans zero: the coupled boundary is **level**,
not better. The honest claim is that coupling all 42 layers cost nothing, which matters because
the rebuild existed to make coupling possible at full model scope. Against EXL3 at 4.0 bpw the
interval excludes zero: EXL3 is genuinely ahead by about 18% relative on this path, and the
coupled boundary did not close that gap.

### Per window

| Window | Coupled K4 | Identity P8 | EXL3 4bpw | Coupled minus identity |
|---|---:|---:|---:|---:|
| 0003 | 0.025792 | 0.027817 | 0.029270 | -0.002026 |
| 0008 | 0.082805 | 0.086353 | 0.058019 | -0.003548 |
| 0009 | 0.052474 | 0.052213 | 0.033092 | +0.000262 |
| 0010 | 0.016554 | 0.017403 | 0.024262 | -0.000850 |
| 0011 | 0.018423 | 0.018976 | 0.018808 | -0.000553 |
| 0021 | 0.048571 | 0.045595 | 0.029681 | +0.002976 |
| 0030 | 0.025483 | 0.025570 | 0.037592 | -0.000087 |
| 0031 | 0.018279 | 0.020648 | 0.019321 | -0.002369 |
| 0032 | 0.040619 | 0.024201 | 0.019419 | +0.016418 |
| 0035 | 0.034341 | 0.042824 | 0.026428 | -0.008483 |
| 0040 | 0.048365 | 0.043390 | 0.029945 | +0.004976 |
| 0041 | 0.091949 | 0.084970 | 0.075859 | +0.006979 |
| 0046 | 0.067506 | 0.066718 | 0.058816 | +0.000788 |
| 0047 | 0.022345 | 0.023402 | 0.025080 | -0.001057 |
| 0051 | 0.014229 | 0.016011 | 0.019251 | -0.001782 |
| 0055 | 0.022519 | 0.022812 | 0.022038 | -0.000293 |
| 0056 | 0.042927 | 0.042746 | 0.035134 | +0.000181 |
| 0060 | 0.020507 | 0.019369 | 0.014176 | +0.001138 |
| 0062 | 0.030274 | 0.030267 | 0.025838 | +0.000007 |
| 0063 | 0.029081 | 0.033704 | 0.025971 | -0.004623 |
| 0074 | 0.048627 | 0.085245 | 0.029017 | -0.036618 |
| 0080 | 0.043521 | 0.048032 | 0.029742 | -0.004511 |
| 0081 | 0.028723 | 0.022449 | 0.052494 | +0.006274 |
| 0083 | 0.034011 | 0.035517 | 0.033193 | -0.001506 |
| 0090 | 0.045199 | 0.051138 | 0.039649 | -0.005939 |
| 0094 | 0.021712 | 0.023294 | 0.016786 | -0.001582 |
| 0098 | 0.049120 | 0.049472 | 0.038676 | -0.000352 |
| 0099 | 0.016895 | 0.012871 | 0.013902 | +0.004024 |
| 0100 | 0.026918 | 0.024562 | 0.024577 | +0.002357 |
| 0113 | 0.060976 | 0.048378 | 0.038902 | +0.012598 |
| 0118 | 0.015827 | 0.013658 | 0.010768 | +0.002168 |
| 0123 | 0.038385 | 0.034301 | 0.045908 | +0.004084 |

## 4. Shapley attribution method and its verification

Per-layer payoff is the fit-role routed-output damage

    D_L = sum_t || sum_e w_te ( f_q,e(x_t) - f_e(x_t) ) ||^2

on 3,072 fit tokens (48 per window, 64 windows of the fit role), in residual-stream units with
quantized E4M3 carriers. Within a layer the per-expert credit is the **exact** Shapley value of
the per-token quadratic game v(C) = || sum_{e in C} z_e ||^2, which has the closed form
psi_e = sum_t <z_e(t), S(t)>, so the expert shares sum to the layer damage exactly. The
implementation is checked against brute-force enumeration over all player orderings in the test
suite, and every damage receipt carries a closure assertion that fails the run if it does not
hold. Observed closure on the first three layers:

| Layer | Sum of expert shares | Layer damage |
|---|---:|---:|
| 3 | 28510.713174493954 | 28510.713174493950 |
| 4 | 578.6717597712234 | 578.6717597712233 |
| 5 | 744.9510276175723 | 744.9510276175722 |

The decode path used by the scorer is independently validated: for layers with encoder chunk
receipts it recomputes all **864 per-projection NMSE values** and compares them to the encoder's
own receipts, and the K3 and K5 chunk-stream decode was checked on CPU against the smoke-encode
receipts to a worst relative difference of 5.5e-8.

**Boundary of the claim.** These shares sum exactly to the layer's routed-output damage. They do
**not** sum to the model KLD, and no claim is made that they do. Across layers the allocation
treats the damage as additive; the end-to-end KLD of the installed allocation is the only quality
claim, which is why it is measured rather than predicted.

Protocol hashes for the damage pass: fit roles
`85cb6f8d29151863457830e41a141743feabcd69706a231d89cd01a3e02175c9`, capture manifest
`f1a6fe7b8828b3461e81ee533d417dd1524355ef6a205145850116560197f81a`, EXL3 scale source
`092be1ffa8db66bf02d4c370d0433a57aa48d4a6e5ce89723ef6a3bb7ca32643`.

## 5. Measured layer damage, and how concentrated it is

All 42 layers were scored at K4 with no new encoding, since the K4 sidecars already exist.
The headline is not the total but its distribution.

| Quantity | Value |
|---|---:|
| Total K4 routed-output damage | 715,466 |
| Highest layer (29) | 115,230 |
| Lowest layer (12) | 442 |
| Ratio, highest to lowest | 261 |

### Across layers

| Layers, worst first | Share of total damage |
|---|---:|
| top 4 | 46.1% |
| top 8 | 69.2% |
| top 16 | 89.5% |
| top 21 | 95.1% |

By depth, the damage sits in the second half of the stack, with a secondary early peak:

| Layer band | Damage | Share |
|---|---:|---:|
| 3-9 | 32,687 | 4.6% |
| 10-16 | 8,768 | 1.2% |
| 17-23 | 61,164 | 8.5% |
| 24-30 | 181,431 | 25.4% |
| 31-37 | 103,401 | 14.5% |
| 38-44 | 328,015 | 45.8% |

### Per layer

K5 damage is shown where that layer was encoded and scored at K5; the ratio is measured,
not assumed.

| Layer | K4 damage | Share | Top expert's share of the layer | K5 damage | K5/K4 |
|---:|---:|---:|---:|---:|---:|
| 29 | 115,230 | 16.11% | 97.0% | 31,710 | 0.2752 |
| 40 | 79,710 | 11.14% | 88.3% | 27,996 | 0.3512 |
| 41 | 71,328 | 9.97% | 81.9% | 23,883 | 0.3348 |
| 43 | 63,471 | 8.87% | 78.5% | 23,000 | 0.3624 |
| 38 | 59,975 | 8.38% | 88.5% | 18,815 | 0.3137 |
| 34 | 42,588 | 5.95% | 89.7% | 13,898 | 0.3263 |
| 25 | 34,114 | 4.77% | 95.9% | 9,129 | 0.2676 |
| 3 | 28,511 | 3.98% | 6.1% | 11,714 | 0.4109 |
| 33 | 26,826 | 3.75% | 81.1% | 11,247 | 0.4193 |
| 42 | 25,192 | 3.52% | 34.7% | 14,833 | 0.5888 |
| 44 | 21,030 | 2.94% | 40.1% | 11,566 | 0.5500 |
| 21 | 18,969 | 2.65% | 94.6% | 4,347 | 0.2292 |
| 19 | 18,722 | 2.62% | 95.3% | 5,996 | 0.3203 |
| 22 | 14,802 | 2.07% | 95.0% | 4,170 | 0.2817 |
| 36 | 10,060 | 1.41% | 47.4% | 4,347 | 0.4321 |
| 30 | 9,483 | 1.33% | 57.5% | 3,031 | 0.3196 |
| 28 | 8,910 | 1.25% | 72.7% | 3,285 | 0.3687 |
| 24 | 8,797 | 1.23% | 87.8% | 2,469 | 0.2806 |
| 31 | 8,485 | 1.19% | 49.7% | 3,831 | 0.4515 |
| 39 | 7,309 | 1.02% | 11.8% | not scored | - |
| 37 | 6,814 | 0.95% | 13.8% | not scored | - |
| 32 | 4,431 | 0.62% | 7.1% | not scored | - |
| 35 | 4,197 | 0.59% | 5.9% | not scored | - |
| 18 | 3,601 | 0.50% | 57.4% | not scored | - |
| 26 | 3,078 | 0.43% | 40.9% | not scored | - |
| 23 | 2,446 | 0.34% | 69.2% | not scored | - |
| 16 | 2,190 | 0.31% | 58.2% | not scored | - |
| 14 | 2,000 | 0.28% | 9.3% | not scored | - |
| 27 | 1,819 | 0.25% | 5.4% | not scored | - |
| 15 | 1,520 | 0.21% | 3.1% | not scored | - |
| 17 | 1,513 | 0.21% | 46.7% | not scored | - |
| 13 | 1,239 | 0.17% | 3.8% | not scored | - |
| 20 | 1,111 | 0.16% | 23.7% | not scored | - |
| 6 | 1,038 | 0.15% | 11.6% | not scored | - |
| 10 | 849 | 0.12% | 6.5% | not scored | - |
| 5 | 745 | 0.10% | 14.6% | not scored | - |
| 7 | 671 | 0.09% | 15.8% | not scored | - |
| 4 | 579 | 0.08% | 2.8% | not scored | - |
| 8 | 572 | 0.08% | 12.9% | not scored | - |
| 9 | 572 | 0.08% | 8.8% | not scored | - |
| 11 | 528 | 0.07% | 2.5% | not scored | - |
| 12 | 442 | 0.06% | 11.2% | not scored | - |

### Inside a layer, the damage is concentrated on individual experts

Each layer has 288 routed experts and the per-expert credit is the exact Shapley value of
the per-token quadratic game, so these shares sum to the layer damage exactly.

| Measure | Value |
|---|---:|
| Layers where one expert holds more than half the layer's damage | 17 of 42 |
| Largest single-expert share (layer 29, expert 2) | 97.02% |
| Share held by the top 100 of all 12,096 layer-expert pairs | 80.12% |
| Share held by the top 1,000 pairs | 89.67% |

The concentrated layers are exactly the high-damage ones. In the diffuse low-damage layers
the top expert holds only 2.5% to 16%.

### What the concentration is not

Four candidate explanations were tested against layer 29, where one expert holds 97.02% of the
layer's damage, and all four are ruled out.

| Candidate cause | Measurement | Verdict |
|---|---|---|
| The expert quantizes badly | per-expert weight NMSE spans 0.00700-0.00790 across all 288; the dominant expert sits at 0.00746, 203rd of 288 | ruled out |
| The expert has larger weights, so the same relative error is larger in absolute terms | Frobenius norms 54.6 / 55.2 / 54.2 against an ordinary-expert median of 54.9 / 55.3 / 54.2 | ruled out |
| The expert is routed far more often | 79 of 24,576 routed slots, against a median of 80 per expert; 147th of 288 by count | ruled out |
| The expert sees unusually large activations | hidden-state norms are essentially constant across the fit tokens, mean 24.95 and median 24.96, and no token exceeds five times the median | ruled out |

The dominant expert index also differs from layer to layer (2, 21, 5, 204, 259, 109, 223, ...), so
this is not one global outlier expert.

### What the concentration tracks instead

The concentrated layers are the domain-skewed ones. Layer 29's damage is 103.07 per token on the
reasoning-termination axis against 1.25 on the code and agentic axis, a factor of 83. Across all
42 layers the correlation between a layer's top-expert share and the log of its domain skew is
**0.784**.

That points at token concentration rather than expert concentration: a small set of tokens, drawn
mostly from one domain, produces most of the routed-output error in the late layers, and the
Shapley credit lands on whichever expert happened to be routed on those tokens. The credit
assignment is exact for the game as defined; the question it leaves open is whether the *expert
identity* is a stable property of the model or an artifact of which tokens this calibration set
contains.

**This has not been tested and should not be assumed.** The test is to recompute the shares on a
disjoint token sample and see whether the same expert dominates. Until that is done, no
per-expert claim is made here, and the byte-efficiency arithmetic below is a hypothesis about
what per-expert allocation could buy, not a measured result.

### Consequence for allocation granularity, if the expert identity proves stable

| Strategy | Bytes to upgrade | Damage addressed on this calibration set |
|---|---:|---:|
| 16 whole layers to K5 (what the ABI allows today) | 14,495,514,624 | 89.5% |
| Top 100 layer-expert pairs to K5 | 314,572,800 | 80.1% |

One expert at one rate step costs 3,145,728 B against 905,969,664 B for a whole layer, so reaching
80% of the damage would cost about 46 times fewer bytes per-expert than per-layer. That is worth
chasing only if the disjoint-sample test holds; if the dominance is token-driven and the expert
identity moves, per-expert allocation would overfit the calibration set and the layer-granular
choice stands.

The running campaign remains layer-granular regardless, because the sidecar ABI stores one rate
per layer and the compiled small-M kernel reads one rate per layer. Every damage number in this
document is fit-role routed-output error on 3,072 calibration tokens, not KLD.

## 6. Assumed versus measured rate ratios

The candidate screen used ratios taken from single-layer weight error: K3 at 3.76x and K5 at
0.285x the K4 damage. The K5 side has now been measured on 19 layers:

| Statistic | Measured K5/K4 damage ratio |
|---|---:|
| Minimum | 0.2292 |
| Median | 0.3348 |
| Mean | 0.3623 |
| Maximum | 0.5888 |

The assumed 0.285 was optimistic by roughly 18%. Recomputing the predicted swap with **measured**
K5 damage and the still-assumed K3 penalty gives a predicted change in the proxy of **-51.2%**
against -56.3% under the assumption. The K3 side is being measured now; it is the half of the
ledger that could still overturn the sign.

## 7. Controls that gate the campaign

* **Storage.** Every build step passes a guard that recomputes the campaign's apparent bytes and
  refuses to proceed unless the total plus the step's reservation stays under the owner-set
  ceiling of 400,000,000,000 B and free space exceeds the need plus a 10 GB floor. Hard-linked
  files are charged once across campaign paths, so link-only assembly cannot double-count the
  161 GB checkpoint. Two reclaim receipts record every deletion with file counts and byte totals.
* **Thermal.** Encoder workers are stopped and resumed by signal at 90 C and 85 C; every event is
  journaled with layer, GPU, pid and temperature.
* **Production.** Every guard invocation re-checks that the backend service and model-stack timer
  are inactive; the model-stack lock is held for the duration of each GPU phase.
* **Overwrite safety.** Encoder, packer, verifier and scorer all refuse to overwrite an existing
  output; a partially written layer is detected and the run stops rather than silently repairing.
* **Determinism.** Layer completion re-derives the exact 4.25 bpw arithmetic from file sizes
  (`weight_bytes * 32 == 17 * logical_elements`) rather than trusting metadata, and cross-checks
  the metadata bpw against it.

## 8. Incidents during the campaign

Recorded because they affect provenance, not because they changed a result.

1. **Host low-memory policy** stopped the encode and orchestrator at 23:21 on 09-05 while 445 GB
   of memory was available. Layer 16's four chunks and receipts were intact; the interrupted pack
   left three partial rank files, which were removed and journaled in
   `<k4 root>/receipts/pack-restarts.jsonl` before the pack was rerun from the retained chunks.
   All long-running jobs were relaunched detached and were never interrupted again.
2. **Preparer defect,** introduced when the arm selector was added: the canonical-path check
   tried to resolve the arm name as a path, aborting the first handoff at 02:56 on 09-06. Fixed,
   with a regression test that drives the preparer entry point with an arm argument. The failure
   happened before any output was written, so no partial capture root existed and the retry was
   clean.

3. **Mixed-rate serving defect, caught by the measurement itself.** The first allocated-model
   run (18 K3 / 7 K4 / 17 K5) measured a true-decode KLD of 1.8146 on every window while the
   runtime gate confirmed all 168 layer-rank pairs loaded at their expected rates with no
   fallback. Encode error is monotone in rate, so a served result that violates rate
   monotonicity by 49x is a runtime defect, not a quality finding, and it is not reported as
   one. Cause: the explicit compile spec that keys the kernel JIT cache did not include the
   stored trellis rate, so within each tensor-parallel rank the first rate compiled was served
   for every layer. Single-rate device closures cannot observe this. The spec now carries the
   rate and its version was bumped to retire rate-less cache entries; the defective outputs are
   preserved under a DEFECT prefix and the closures and measurement are re-run on the rebuilt
   image.
4. **K5 decode read past its window.** A lane's eight overlapping windows span 16 + 7 x rate
   bits; at K5 that crosses into a third ring word for half the lanes, which the two-word merge
   never read. Proved with a pure-Python model of the lane geometry against the CPU reference
   (now a permanent test) and fixed for K5 only, so K3 and K4 code generation is unchanged.
   The device closure caught it: input carriers exact, post-FC1 payload wrong in 4,030 of 4,096
   bytes.

## 9. Phase 1 measured results

All arms below were measured on the same sealed 32 windows, the same teacher, the same
nvfp4_ds_mla KV cache and attention backend, and, for the mixed-rate arms, the rebuilt image
whose compile spec is keyed on the stored rate (`sha256:c121590d…`). Both device closures pass on
that image. Intervals on differences are percentile bootstraps of the paired per-window
difference, 20,000 resamples, seed 20260906.

| Arm | Layers K3 / K4 / K5 | Stored bpw | File bytes | True-decode KLD | Paired vs uniform K4 | Better in |
|---|---|---:|---:|---:|---|---:|
| Uniform coupled K4 | 0 / 42 / 0 | 4.2540 | 161,867,572,608 | 0.036967 | — | — |
| Same-size allocation | 18 / 7 / 17 | 4.2302 | 160,961,603,488 | 0.050484 | +0.013517 (+36.6%), [+0.008596, +0.018973] | 1 of 32 |
| Upgrades only | 0 / 25 / 17 | 4.6587 | 177,269,057,440 | **0.034181** | **−0.002786 (−7.5%), [−0.004723, −0.000961]** | 25 of 32 |

**Reading.** Upgrading the 17 highest-damage layers to K5 lowers end-to-end KLD, and the
interval excludes zero; rate monotonicity holds, which is the evidence that the mixed-rate
serving path is correct after the defects in section 8. The same-size allocation, which pays
for those upgrades by downgrading 18 low-damage layers to K3, is a large loss. By subtraction
the 18 downgrades cost about +0.0163 of KLD, against the routed-output proxy's estimate that
those layers hold 2.8% of the damage. The proxy is a usable screen for which layers deserve
bits and a poor price for what removing bits costs. The K3-only arm that would have priced the
downgrades directly was not run, by the owner's decision.

Per-domain KLD of the upgrades-only arm: general 0.033236, legal 0.049390, code and agentic
0.030089, reasoning termination 0.024009. Against EXL3 at 4.0 bpw (0.031300 on these windows)
the upgrades-only arm closes about half the gap the uniform model showed, at 0.66 bpw more.

### Against the stock NVFP4 carrier, with a caveat

No stock-carrier measurement exists on this exact serving path (the pilot's stock arm failed
before scoring). The only stock run on the same windows and teacher is the rotation
experiment's arm, served with FP8 KV, DCP4, eager execution and scored from a single 2,048-token
prefill pass rather than 2,047 forced-decode rows. Same windows, different path, so this is
context only.

| Arm | Mean over the 32 windows | vs stock | Windows worse than stock |
|---|---:|---:|---:|
| Stock NVFP4 (FP8 KV, prefill-scored) | 0.039789 | — | — |
| Uniform coupled K4 P8 | 0.036967 | −0.002821 | 12 |
| Upgrades-only P8 | 0.034181 | −0.005607 | 6 |

The six windows where the upgrades-only arm sits above stock are 0032 (+0.0064), 0055 (+0.0007),
0056 (+0.0185), 0063 (+0.0031), 0081 (+0.0020) and 0094 (+0.0009). A matched stock arm on the
nvfp4-KV forced-decode path is a 35-minute measurement once the harness gains a no-sidecar arm.

### Byte cost of the upgrade

Each layer moved from K4 to K5 adds 905,969,664 B; the 17 upgrades add 15,401,484,832 B, 9.5%
over the uniform checkpoint and 15,553,350,112 B over the identity budget. The stored rate goes
from 4.2540 to 4.6587 bpw. The tensor-core path is unchanged: every rate decodes in registers to
E4M3 and issues the same mxf8f6f4 MMA, and the closure gate requires the native small-M
materialized N128 owner path with no fallback. Grouped M64 prefill kernels remain K4-only, so
K5 layers serve batches larger than one row by row.

## 10. Status and what remains

The campaign's measurements are complete as of 2026-09-06 16:20 EDT: the uniform coupled K4
model, the same-size allocation, and the upgrades-only arm are all measured on the sealed
protocol. Kernel note: the mixed-rate lineage differs from the pinned K4 kernel in accepting
rates 3, 4 and 5, rescaling the shared-memory staging, reading a third ring word at K5, sizing
the descriptor carriers from the rate, and keying the compile spec on the rate; at rate 4 every
one of those evaluates to the inherited K4 behaviour. K3 and K5 both pass the M1 device closure
against the CPU reference on the image that produced the numbers above.

Not done, and worth doing next:

* a matched stock-carrier arm on the nvfp4-KV forced-decode path, so the comparison in section 9
  stops carrying a caveat (harness needs a no-sidecar arm; one 35-minute run);
* a two-rate-in-one-process device closure, which is the test that would have caught the
  compile-cache defect before a full measurement did;
* the disjoint-token-sample test of expert identity from section 5;
* online K6 encoding of the carrier's remaining BF16 linears at load (19.09 GB is 128-aligned
  and eligible, about 11.8 GB model-wide or 2.96 GB per GPU recoverable), validated in stages
  because attention projections at K6 have no quality measurement on this model yet.
