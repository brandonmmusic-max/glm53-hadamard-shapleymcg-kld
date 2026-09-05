# GLM-5.3-Flash decision log

Times and numerical values in this file are backed by JSON under `evidence/`
or by the hash-addressed campaign receipts on `klcstore`. Lower KLD is better.
Conditional-fit was an adaptive development role; protected selection,
confirmation, and final roles remained unopened for this redesign branch.

## 2026-09-04: Decision 20 — matched-path layer-22 P8 end-to-end KLD gate

- **Decision before result:** after layer 22 passed the exact full-Hessian
  causal screen, build all 288 experts and compare its no-LDLQ P8
  reconstruction against decoded full-Hessian GPTQ NVFP4. Both arms replace
  exactly layer 22 as BF16 through the identical InstantTensor/framework
  path. On the fixed, four-domain conditional-fit panel of 32 windows, pass
  required mean paired delta at most `-0.0014` nats and a paired BCa 95%
  upper bound below zero.
- **Outcome:** failed. Candidate mean KLD was `0.0391435578` versus
  `0.0387275135` for the matched GPTQ control: delta `+0.0004160443`, a
  `1.07429%` regression, BCa 95% CI
  `[-0.0002491743,+0.0011677178]`, with 14/32 window wins. Legal improved,
  while general, code/agentic, and reasoning/termination regressed. Thus the
  layer's `17.0735%` causal full-expert NMSE gain did not predict end-to-end
  KLD direction. P8 device closure, speed, and determinism remain blocked;
  no protected confirmation logits were opened. Plan SHA-256
  `4499f08c8b653a0ac5ac47e036ec2f1033cf3bf8fcc4fe8325c295a7b708ea97`;
  analysis SHA-256
  `a8b7ab00b56c38e2bd63710f3c569bf77ca6f04ef82a9823501f486632f28f23`.

## 2026-09-04: Decision 19 — full-Hessian causal qualification of layer 22

- **Decision before result:** after layer 22 ranked least-bad in the broad
  block-Hessian screen, acquire its pinned full REAP activation capture and
  apply the same causal gate/up/SwiGLU/down rule used for layers 3, 19, and 20.
  A build required at least 10% lower geometric-mean full-expert NMSE, BCa
  upper log-ratio below zero, and 12/16 expert wins.
- **Outcome:** passed decisively. Full-expert NMSE improved `17.0735%`, all
  16 experts won, and the BCa mean-log-ratio interval was
  `[-0.1910927,-0.1828700]`. This reversal from the block-Hessian result is
  evidence that full cross-group/casual calibration geometry is material.
  Capture verification SHA-256
  `8ebfa6a1ce54e26c62ae49e9f83459807656e84d465e02772e8646d0f9e11feb`;
  analysis SHA-256 `d59c064522bf003198bb88d70bc59d0d74a3ad5bb875d913d7bea19c533e37ab`.

## 2026-09-04: Decision 18 — broad block-Hessian P8 layer ranking

- **Decision before result:** because only three full activation captures were
  materialized, screen all 38 layers covered by the immutable 16x16 Hessian
  set using P8 versus block-Hessian GPTQ at charged-equal bpw. Treat the result
  only as a gate/up projection design prior; require at least 10% improvement,
  a paired expert BCa upper bound below zero, and 12/16 expert wins.
- **Outcome:** no layer passed. The least-bad layer was 22, where P8 was still
  `2.7454%` worse with 10/16 expert wins and CI crossing zero. Layers 23 and
  26 were `3.0469%` and `5.8484%` worse; early layers were much worse. This
  rules out the old block Hessians as a shortcut to the full-Hessian gain.
  Plan SHA-256 `ac7ba58542bdbee269e5de87e6d9d220a72195a48a5f443e15634a1eecbaae81`;
  result SHA-256 `359386462bf62c6630586156a157bf43cf932fd52e6d5c781695fdb135cbed38`.

## 2026-09-04: Decision 17 — invalidate the infeasible causal all-layer sweep

- A 42-layer full-Hessian causal sweep was frozen before execution, but exact
  byte validation found complete 10 GB activation files only for layers 3,
  19, and 20. Layer 4 failed before scoring. All workers were stopped, no
  partial `raw.json` was accepted, and production serving was restored. The
  frozen plan is preserved rather than rewritten. Invalidation SHA-256
  `71fcd5ce80790d0d6590311ff13cd104eeae6693f464886b6ecf7bc3b64583bb`.

## 2026-09-04: Decision 16 — external V5 per-layer decomposition

- **Decision before result:** after the combined layer-3/20 subset failed,
  decompose its two layers on the same opened V5 role with a two-comparison
  Bonferroni correction. This is exploratory attribution, never a protected
  claim.
- **Outcome:** neither layer survived. Layer 3 was effectively flat
  (`-0.0656%` improvement); layer 20 improved `0.6996%`, but its 97.5% BCa
  interval crossed zero and legal/code means were not both favorable. Result
  SHA-256 `20b688c8765b758904cb4a0b90d7b3e3dcaf62d38b11804f66051cbb2d26ec06`.

## 2026-09-04: Decision 15 — external V5 gate for the P8 layer-3/20 subset

- **Decision before result:** after BF16-matched conditional-fit attribution
  supported layers 3 and 20, freeze that two-layer subset and compare it with
  the decoded-GPTQ control on all 63 domain-balanced V5 selection-wave
  windows. Both arms use BF16 overlays through the identical
  InstantTensor/framework path. Provisional validation required at least 3%
  lower arithmetic-mean KLD, a paired BCa 95% upper bound below zero, and a
  negative mean delta in each of general, legal, and code/agentic domains.
- **Outcome:** failed. Mean KLD improved from `0.0447961714` to
  `0.0441764187`, a `1.38349%` reduction, but mean delta was only
  `-0.0006197527` with BCa 95% CI
  `[-0.0024961352,+0.0006606803]`. General improved by `-0.0026108913`
  nats while legal and code/agentic regressed by `+0.0004313914` and
  `+0.0003202418`. The result is not a generalized codec win. The V5 role had
  previously been used for H16, so this is external reused-role evidence, not
  pristine confirmation; the 28 confirmation logits remain unopened. Plan
  SHA-256 `7392a69e8a69099bebd651513cf14040acb9f020dd0f9f1a5348285e48acc6c8`;
  result SHA-256 `13d8dc8824acca8f9f1e4d6768e937741d81da58fba233892032246c96cbf89e`.

## 2026-09-04: Decision 14 — matched-path P8 layer attribution

- **Decision before result:** compare layer 3, 19, and 20 separately against
  one shared decoded-GPTQ BF16-overlay control on conditional-fit n=32. A
  layer required mean delta at most `-0.0004` nats and a
  Bonferroni-adjusted 98.333% paired BCa upper bound below zero.
- **Outcome:** layers 3 and 20 passed. Layer 3 reduced mean KLD by `4.15725%`
  (delta `-0.0016171098`, adjusted CI
  `[-0.0040551835,-0.0002973811]`); layer 20 reduced it by `3.08581%`
  (delta `-0.0012003354`, adjusted CI
  `[-0.0034453658,-0.0003380131]`). Layer 19 improved only `0.74949%` and
  failed its interval gate. These are adaptive conditional-fit results, not
  protected claims. Result SHA-256
  `56811eb6c35cfc94c81d1aefe685e7ba0eaab951aa5aad2c84485de242ea2055`.

## 2026-09-04: Decision 13 — repair the mixed execution-path P8 comparison

- **Evidence before decision:** Decision 10 compared a BF16 pseudoquant
  candidate with a packed NVFP4/Humming control. That changes both weight
  values and the execution pipeline, so it cannot attribute KLD to the
  encoder and is invalid for the stated gate.
- **Decision before result:** decode the exact GPTQ NVFP4 control weights to
  BF16 and run both arms as InstantTensor BF16 overlays on the same n=32 role.
  Retain the predeclared `-0.0014`-nat and BCa-upper-below-zero gate.
- **Outcome:** directionally favorable but still a gate failure. The P8
  layers-3/19/20 candidate reduced mean KLD by `1.63810%`; paired delta was
  `-0.0006371990`, BCa 95% CI `[-0.002456,+0.000675]`. Result SHA-256
  `ac0a142145f720916f407b51d3125117b462491295aa86abe29bb32a83f62839`.
  Decision 10's numerical row is retained as a diagnostic receipt but is
  retracted as encoder evidence.

## 2026-09-04: Decision 12 — powered V5 check of down-only H16

- **Decision before result:** rerun the layer-3 down-only H16 candidate on all
  63 balanced V5 selection-wave windows against its same-path stock control.
- **Outcome:** failed and reversed the smaller V4 indication. Candidate mean
  KLD was `0.0404740175` versus stock `0.0399280357`, a `1.3674%` regression;
  paired mean delta was `+0.00101680`, BCa 95% CI
  `[+0.0000962,+0.0028456]`, and every domain worsened. Result SHA-256
  `adcfa636e26c8c17ffc3cd190cfd496cdf030f119d29f5c18423a1a7d92f612a`.

## 2026-09-04: Decision 11 — stop the learned 4 KiB T12 law

- **Decision before result:** fit one checkpoint-family 4096-byte monotone E4M3
  law on 16 experts, freeze it, then validate causal full-expert NMSE on 16
  disjoint experts against both procedural MCG and GPTQ NVFP4. A new layer/KLD
  build required at least 15% improvement versus MCG, 20% versus GPTQ, BCa
  upper bounds below zero for both, and at least 12/16 MCG wins.
- **Outcome:** failed decisively. The learned law was 56.40% worse than MCG,
  32.01% worse than GPTQ, and won 0/16 experts. Both paired expert BCa
  intervals were entirely unfavorable. No second full-layer artifact was
  built. Plan SHA-256 `0eef41f7c484e4502b7f7eaa1f1f21e323bb6dd742ea5f727f843b611e867064`;
  result SHA-256 `4f2e75ef7cb673b5c08ff7286b69d41df505fec8e21f036a594632a6874cfcab`.

## 2026-09-04: Decision 10 — conditional-fit KLD gate for Hessian P8

- **Decision before result:** compare the dense BF16 pseudoquant layer-3 P8
  reconstruction with a freshly regenerated, domain-balanced full-Hessian
  GPTQ NVFP4 layer-3 control on the exact n=32 conditional-fit role. The P8
  codec is physically 4.25 bpw but was charged 4.5 bpw for this gate. Pass
  required mean paired delta at most `-0.0014` nats and paired BCa upper below
  zero. P8 kernel work remained blocked until a pass.
- **Outcome (subsequently invalidated for encoder attribution by Decision
  13):** the arms used different execution pipelines. Candidate mean KLD was
  `0.0548696998`; GPTQ was `0.0546027319`; delta was `+0.0002669679`, BCa 95%
  CI `[-0.0029615132,+0.0069333575]`, with 15/32 window wins. Result SHA-256
  `dbb5501165e94ba8c3b1e365d5f596e7c482273ff4e3e04de65af4fd01a5caf6`.
  P4/P8 prologues, device closure, speed, and determinism gates were not run.
  Preserve this row only as a mixed-pipeline diagnostic, not as evidence that
  the P8 encoder itself is null.

## 2026-09-04: Decision 9 — build the full Hessian P8 layer only after causal NMSE

- **Decision before result:** a 288-expert layer build required the 16-expert
  causal screen to improve full-expert NMSE by at least 10% versus GPTQ, with
  paired expert BCa upper below zero and at least 12 wins.
- **Outcome:** passed. Causal full-expert geometric-mean NMSE improved 11.87%,
  15/16 experts won, and the mean-log-ratio BCa interval was
  `[-0.1586380,-0.0603206]`. Four-GPU build then produced 864 bit-exact dense
  reconstructions plus four compressed codec shards. Screen result SHA-256
  `26489a089e1f179d766c17df1035a42eaa51b1fb096f10013af4790f72f68b78`.

## 2026-09-04: Decision 8 — qualify Hessian feedback across experts

- **Decision before result:** advance beyond the one-expert screen only at 10%
  or greater held-out projection-NMSE improvement versus GPTQ NVFP4, paired
  expert BCa upper below zero, and at least 12/16 expert wins.
- **Outcome:** passed: 14.95% geometric-mean improvement, 16/16 wins, BCa
  mean-log-ratio interval `[-0.1720710,-0.1520428]`. Hessian feedback reduced
  plain-MCG output NMSE from `0.00402248` to `0.00218785`; GPTQ NVFP4 was
  `0.00257230`. Result SHA-256
  `c166ebd2439dd3b86f6da90e680e4244a162a4989fa53a3f6435931d01f17b3c`.

## 2026-09-04: Decision 7 — output-aware GPTQ-style Viterbi, without LDLQ

- The new encoder preserves physical 16x16 trellis order, applies static
  within-group activation ordering, and carries full-Hessian GPTQ-style error
  between native groups while iteratively refitting UE8M0 scales.
- A route-weighted ridge correction for E4M3 carrier mismatch overfit in v1
  and became null under nested regularization in v2. It was not promoted.
- The Hessian-only encoder unexpectedly beat GPTQ NVFP4 on both held-out
  expert-0 projections, motivating Decision 8. V1 and v2 result SHA-256 values:
  `1f823d62784fdc3bd72ade1675667684c920c2042eb66f33e39e42213e2a8735`,
  `3280d92a0e8d5606c3c8a74924bb4c2571510cec664bac82bdc279f993876208`.
  No LDLQ path was implemented or used.

## 2026-09-04: Decision 1 — diagnose the W6A8 carrier before P8

- **Decision before result:** run the fixed arms `R0,W0,W1,W2,W3`, once and in
  that order, on the exact domain-balanced conditional-fit panel of 32 windows.
  `R0` serves the decoded MXFP6 weights through the BF16 framework path; the
  W arms use the identical weights and sequentially isolate routed reduction,
  fast activation math, and dynamic FC2 activation scaling.
- The primary estimand is paired mean `KLD(W3) - KLD(R0)`. The carrier is
  repaired only if the paired BCa 95% upper bound is at most `+0.0014` nats. A
  mean delta of at least `+0.01` blocks P8; an intermediate result requires
  calibrated-input-scale and staged-epilogue diagnosis before P8.
- Attempt v1 seal `64e55a179041bf0e24d68fc2fc582463bf1078e77ac12d522c50b4af2098f7c5`
  was invalidated after R0 because its v80 image has an H16-only MXFP6
  plugin; W0 failed before scoring any window. Corrected attempt v2 pins the
  identity-capable v79 image for every arm. V2 record seal:
  `8758559f4b196b8779e52e517ec9cc7368b26e8fda8fc812eed8194402d37ccf`.
  V2 analysis plan: `5cfb8fbecc68fe1a47f4baefcf144239f970743b086ac0c382a74659a3746a86`.
  Role manifest: `b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14`.
- **Outcome:** failed and P8 remained blocked. `R0 = 0.0515559935`, while the
  best fixed kernel arm was `W3 = 0.1132481179`; `W3-R0 = +0.0616921245`
  nats with paired BCa 95% CI `[+0.0504003196,+0.0744450147]`. All 32 windows
  worsened. Result SHA-256:
  `274b4bda537b2256f98307d2a149368c281b0d02b032edb0d901d8518548844b`.
  Selection, confirmation, and final roles remain closed. The 28 confirmation
  logits were not opened.
- P8 uses `mxf8f6f4` at twice the MMA issue count of NVFP4 and is the quality
  product. P4 is the `mxf4nvf4` speed product. No LDLQ is used in this program.

## 2026-09-04: Decision 2 — restore GLM's clipped SwiGLU in W6A8

- **Evidence before decision:** gate 3 failed with `W3-R0 = +0.0616921245`
  nats, paired BCa 95% CI `[+0.0504003196,+0.0744450147]`; all 32 windows
  worsened. Its three fixed switches were individually null-sized.
- Code inspection then found a specific architecture mismatch: the checkpoint
  declares `text_config.swiglu_limit = 10`, the GLM BF16 path uses
  `SiluAndMulWithClamp`, and B12X's kernel accepts this limit, but the v79
  MXFP6 bridge omitted it when constructing `fused_moe.Caps`.
- **Decision before result:** run one `C0` arm with W3's settings plus the
  model-locked limit 10. Carrier closure still requires paired BCa upper
  `C0-R0 <= +0.0014` nats; mean delta `>= +0.01` keeps P8 blocked.
- Seal: `56d94af04816aa04a0a9aa89d89b77d973c1aa5b45e7bb8148e5f411a4c19246`.
  Analysis plan: `6bf4769a1b21df6e28ea9f945d997af3b27cc7f6bc8393d1c04efe69b933227d`.
- **Outcome:** failed and P8 remains blocked. Restoring the clamp reduced mean
  KLD by `0.0015020556` nats relative to W3, but `C0 = 0.1117460623` versus
  `R0 = 0.0515559935`; `C0-R0 = +0.0601900688`, paired BCa 95% CI
  `[+0.0492044300,+0.0730315321]`. Result SHA-256:
  `7196e9a4ebe91476e16a78bb5d5b530d357e6dc590c5589eae38eaa0bc0a46aa`.
  Protected roles remain closed. No LDLQ is used.

## 2026-09-04: Decision 3 — calibrate FC1 E4M3 input-scale phase on REAP

- **Evidence before decision:** clamp restoration left `C0-R0 = +0.0601900688`
  nats, while a random-input phase sweep improved routed-output NMSE by only
  about 1.85%. Random inputs are not a valid calibration substitute.
- **Decision before result:** on 16 fixed layer-3 experts, use 32
  domain-balanced routed REAP fit samples per expert to select one shared FC1
  phase from the eight values `2^(j/8), j=0..7`, then evaluate that frozen
  phase on the next 32 fit samples per expert. FC2 remains unit-scaled.
- The scale is material only if validation geometric-mean expert-output NMSE
  improves by at least 5% and at least 12 of 16 experts improve. Otherwise the
  unit phase is retained and diagnosis moves to FC2 activation/epilogue
  isolation. Plan SHA-256:
  `776ed2d93dbb285b6cd323f3c46e340f1a7ceadd84769335802f5d370dbc4284`.
- **Outcome:** FC1 phase was not material. The selected √2 phase improved
  selection geometric-mean NMSE by 10.99%, but validation worsened by 1.23%
  and only 8/16 experts improved. Unit a1 is retained. Result SHA-256:
  `371741bb6c693b2456343c417d73462555749f4b53bfc7a70471d4b51dfe96ca`.
  Protected roles remain closed. No LDLQ is used.

## 2026-09-04: Decision 4 — decompose the two E4M3 activation hops from fusion

- **Evidence before decision:** √2 improved the FC1 scale-selection split by
  10.99%, then worsened the disjoint validation split by 1.23%, with only 8/16
  expert wins. Unit input scale is retained.
- **Decision before result:** on the fixed validation routes, compare decoded
  BF16 weights under BF16 activations, FC1-only E4M3, FC2-only E4M3, both E4M3
  hops, and the fused W6A8 kernel. Static FC2 scaling is used so the Torch and
  fused stage boundaries match exactly.
- If fused-versus-both-activation NMSE exceeds both-activation-versus-BF16
  NMSE, the fused carrier/staging is dominant. The reverse identifies native
  activation quantization as dominant; fused-versus-both `<= 1e-4` is closure.
  Plan SHA-256:
  `a817a68828ca1dbb64b2fae95b9d058488134f85d25c3343a914365c1782cdba`.
- **Outcome:** activation quantization dominates. Geometric-mean NMSE was
  `0.00018513` for FC1-only E4M3, `0.00054948` for FC2-only, `0.00076482`
  for both, and `0.00010181` for fused-versus-both. Thus the native activation
  loss is 7.51x the residual fused discrepancy and FC2 is 2.97x FC1. The fused
  residual narrowly missed the `1e-4` closure line. Result SHA-256:
  `ce126d0dcecb94f93dae9457a994e5635ac8af82bf4e642d704fb2f9ff1b9d52`.
  Protected roles remain closed. No LDLQ is used.

## 2026-09-04: Decision 5 — calibrate per-expert FC2 E4M3 scale phase

- **Evidence before decision:** the stage decomposition identifies FC2 E4M3
  activation quantization as the largest carrier component; a shared FC1 phase
  did not validate, but the checkpoint/runtime ABI supports per-expert scales.
- **Decision before result:** tune a2 independently for the same 16 experts on
  offset-0 REAP routes over `2^(j/8), j=0..7`, freeze those phases, and compare
  them with unit a2 on offset-32 routes. A result is material only at at least
  5% validation geometric-mean NMSE improvement and 12/16 expert wins. Only a
  material result permits expansion to all 288 experts and end-to-end KLD.
  Plan SHA-256:
  `9d13011bb6f8e1eb6612aa19c4ca26fd76074dbda8dfec8bebb0b4ca77e45881`.
- **Outcome:** failed. The per-expert phases worsened validation geometric-mean
  NMSE by 8.21% and improved only 6/16 experts. Unit a2 is retained. Result
  SHA-256:
  `557cfaa52feb2cd22d54283c451bed6ee85263542a84183c02f986495a7c6953`.
  Protected roles remain closed. No LDLQ is used.

## 2026-09-04: Decision 6 — search the native E4M3 scale per K32 block

- **Evidence before decision:** both shared and per-expert global-scale phases
  failed validation, while E4M3 activation quantization accounts for 7.51x the
  remaining fused discrepancy and FC2 is the largest stage.
- **Decision before result:** in pseudoquant, choose for each activation K32
  block among the ordinary containment exponent and offsets `-1,-2` by minimum
  local E4M3 reconstruction SSE. The payload remains ordinary E4M3 and the
  selected scale remains one UE8M0 byte, so no MMA format changes.
- Implement this search in the fused prologue only if geometric-mean routed
  output NMSE improves at least 15% and at least 12/16 fixed experts improve.
  Plan SHA-256:
  `6a2c1de721b3dfe77c8d33f9bb97dd320b53711d81f6104985d31b3d42df53f2`.
- **Outcome:** null; no kernel implementation. Candidate geometric-mean NMSE
  was `0.0007648181` versus baseline `0.0007648172` (relative change
  `-0.000112%`) with only 5/16 expert wins. Result SHA-256:
  `9a45010679c68e3b9f02ea601b2a8ca1d05fe58582cb0d35f8a7707e958344f4`.
  Protected roles remain closed. No LDLQ is used.

## 2026-09-04: Decision 7 — scaled-H128 MCG layer-3 pseudoquant KLD

- **Decision before result:** compare one scaled-H128 procedural-MCG K4 P8
  layer-3 build with one decoded full-Hessian GPTQ NVFP4 control on the exact
  32 conditional-fit windows. Both arms use the same InstantTensor BF16
  loader and dynamic E4M3-per-K32 input/post-SwiGLU carrier. Pass only if
  candidate minus control mean KLD is at most `-0.0014` nats and the paired
  BCa 95% upper bound is below zero. No exclusions, substitutions, or rerolls.
  Execution-manifest SHA-256:
  `a4ed3de32a8adbeb6716441ffcd87aece984600f8a20da4b327c8751e4ac1c0c`.
- **Outcome:** failed/null. Candidate mean KLD was `0.0534951744` versus
  `0.0533314176`; delta `+0.0001637568`, relative change `-0.3071%`, paired
  BCa 95% CI `[-0.0021343931,+0.0059936601]`, with 22/32 window wins. Three
  domain means improved, but the legal mean regressed by `+0.0058844981` and
  `conditional-fit-0074` regressed by `+0.0487102441`. Analysis SHA-256:
  `e4e19f9f73763b48cd73439c7114cd21291ae51938d457c2a4da2af3a3f198e2`.
  This is decoded-weight pseudoquant evidence, not native P8 closure.
  Confirmation logits remain unopened. No LDLQ or BlockLDLQ was used.
- **Next frozen direction:** use disjoint fit routes to estimate effects over
  all 288 experts and test a sub-4-KiB per-expert identity/H128 policy. Reuse
  of these 32 windows is adaptive tuning evidence only and cannot qualify a
  redesigned candidate.

## 2026-09-04: Decision 8 — all-expert identity/H128 attribution

- **Decision before result:** evaluate the physical K4 procedural-MCG P8
  identity and scaled-H128 encoders on all 288 layer-3 experts using disjoint,
  domain-balanced REAP fit calibration and evaluation slices. Advance only if
  H128 improves full routed-output geometric-mean NMSE by at least 5%, wins at
  least 173 experts, and the paired expert-bootstrap interval excludes zero.
- **Outcome:** pass for policy construction. Full routed-output NMSE improved
  by 28.0315% with 252/288 wins and mean-log-ratio BCa 95% CI
  `[-0.367496,-0.293519]`. Gate and up weight NMSE improved by 18.8058% and
  19.2387% with 288/288 wins each; down weight NMSE worsened by 0.9386% with
  116/288 wins. The preregistered 5% rule made 242 experts eligible and kept
  the expert policy at 39 bytes. Analysis SHA-256:
  `70c9313ace09bd9810ea83bff574eddcdaf5015466e51b4ae5fe3526c4743e50`.
  Protected roles remained closed. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 9 — disjoint fit validation of 39-byte policy

- **Decision before result:** on one fit-only selection slice choose H128 only
  for earlier-eligible experts where it beats the newly built identity P8 by
  at least 2%; then evaluate that frozen mask on a disjoint fit-only
  validation slice. Advance to adaptive KLD only for at least 10% improvement
  versus GPTQ NVFP4, at least 173/288 wins, and a paired BCa upper bound below
  zero.
- **Outcome:** pass to adaptive linkage only. The frozen mask assigns H128 to
  98 experts and identity to 190. Validation geometric-mean routed-output NMSE
  was `0.0048759244` for the hybrid versus `0.0056111689` for GPTQ NVFP4, a
  13.1032% improvement with 201/288 wins and mean-log-ratio BCa 95% CI
  `[-0.187872,-0.097337]`. Policy SHA-256:
  `04e65ecd95a3bc04130de35e5a268150a5d685fa383a4df5cdeb7c406e56acc3`;
  validation SHA-256:
  `0ee36449eed9fcbc6a7640794955cfc503dbe0dc14514733171828cdae72da96`.
  This is fit evidence, not end-to-end qualification. No LDLQ was used.

## 2026-09-04: Decision 10 — adaptive end-to-end policy linkage

- **Decision before result:** reuse the exact 32 already-opened
  conditional-fit windows once, with no exclusions or rerolls, comparing the
  4.250443-bpw masked P8 codec against the existing matched decoded GPTQ
  NVFP4 control. Treat the result as developmental regardless of outcome.
  Numerical signal requires mean delta at most `-0.0014` nats and paired BCa
  upper bound below zero. Plan SHA-256:
  `8383c3b69eb4356d297a1859caa4a40594d1e212b223a1a83def0ec162d2e3fa`.
- **Outcome:** adaptive null/fail. Candidate mean KLD was `0.0527762731`
  versus `0.0533314176`; delta `-0.0005551445`, a 1.0409% improvement, with
  paired BCa 95% CI `[-0.0024023246,+0.0011452178]` and 19/32 wins. Code
  improved by `-0.00382252` nats while legal regressed by `+0.00377694`.
  Analysis SHA-256:
  `3feb4a8ddeffa047ea780b6ba1d6fc6f14bc40133648950fb54152bf141e40cb`.
  The protected 28 confirmation logits remain unopened. No LDLQ or BlockLDLQ
  was used.

## 2026-09-04: Decision 11 — optimize the actual routed top-8 sum

- **Decision before result:** replace independent per-expert mask scoring with
  the squared error of the actual route-weighted top-8 MoE sum, including
  co-routed cross-expert terms. Use eight fit windows for selection and eight
  disjoint fit windows for validation, two per domain in each phase. Advance
  only for at least 10% improvement versus GPTQ NVFP4, 6/8 wins, and a BCa
  upper bound below zero. Plan SHA-256:
  `3b8d5bceae7ee239315bfb179649c044c652c5ba11fa91ca8ab375a874b642dd`.
- **Outcome:** fail/null. The optimized 101-H128/187-identity mask improved
  aggregate validation NMSE by 1.1388% and won 6/8 windows, but mean log ratio
  was `+0.141697` with BCa 95% CI `[-0.129819,+0.937355]`. One code window
  exposed a large GPTQ-specific advantage. Analysis SHA-256:
  `9ae85c8f35370821ce7ba2a0013050419a6097e33a321cab7c51e940bd10f843`.
  Protected roles remained closed. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 12 — separate FC1 H128 from identity down

- **Evidence before decision:** across all 288 experts, H128 improved gate and
  up weight NMSE for 288/288 experts but worsened down on average. The binary
  joint-route policy was a null.
- **Decision before result:** add an algebraically closed third state that
  uses H128-quantized gate/up, inverse-H128 before SwiGLU, and identity-MCG
  down weights in the original basis. Optimize one 75-byte two-bit expert
  policy on eight new fit windows and validate on eight more untouched fit
  windows. Use the same 10%, 6/8, BCa-upper-below-zero gate. Plan SHA-256:
  `ea735343a64815b0ea7cdbe9dad241027b7b5493274aadf1617c09f4f074fe3b`.
- **Outcome:** promising but failed the full rule. The 104-identity,
  94-full-H128, 90-FC1-only mask improved aggregate validation NMSE by
  24.0491% and won 6/8 windows. Mean log ratio was `-0.162510`, but BCa 95%
  CI `[-0.325965,+0.019159]` narrowly crossed zero. Policy SHA-256:
  `3e653916a5628b0600089a2a595caa44851bc21e0bf44dac3c0d2584cf9a6612`;
  analysis SHA-256:
  `6b7463191e7180b6f43f5b01699262ae9ec4768e1b8562de87177b50528f0bee`.
  No protected role was opened and no LDLQ was used.

## 2026-09-04: Decision 13 — powered validation of frozen FC1 policy

- **Decision before result:** freeze Decision 12's policy unchanged and use
  every one of the 32 remaining fit windows, eight per domain. Advance to one
  previously unopened KLD role only for at least 10% aggregate improvement,
  24/32 wins, and a paired BCa upper bound below zero. No exclusions,
  substitutions, policy edits, or rerolls. Plan SHA-256:
  `5a630dd5422aa96bbb9138388f04536110eac4593fdda5c0b083339e4a079737`.
- **Outcome:** failed the uncertainty gate despite a replicated large point
  effect. Aggregate routed-output NMSE improved by 12.0290% and 25/32 windows
  won. Code won 8/8, general and legal each won 7/8, and reasoning won 3/8.
  Mean log ratio was `+0.0400501`, BCa 95% CI
  `[-0.125090,+0.356753]`, driven by rare windows where GPTQ error was nearly
  an order of magnitude lower. Analysis SHA-256:
  `7c9c2b7d30ace28b068b0f91c12383ea2468bee6efa364d7449427845b8ad9d9`.
  The 28 confirmation logits remain unopened. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 15 — correct scalar-E4M3 rate labeling

- Code audit found that the early `scalar-e4m3-k32` upper-bound diagnostic
  selected independently from all finite E4M3 values. Its dense symbol stream
  therefore requires 8 bits/weight plus 0.25 bpw of UE8M0 scales: **8.25
  bpw**, not the 4.25 bpw recorded in those historical screen rows.
- This does not affect any physical K4 trellis payload, KLD candidate, or
  4.250443-bpw claim; all of those store four trellis edge bits per weight.
  The historical JSON is retained immutably, and future screen code now emits
  8.25 bpw for that upper bound. Any new scalar fallback must use at most 16
  E4M3 reconstruction levels and explicit 4-bit indices to qualify at 4.25
  bpw.

## 2026-09-04: Decision 16 — genuine scalar16 hybrid and MCG alpha refit

- **Decision before result:** replace the invalid unrestricted scalar-E4M3
  fallback with a fitted 16-entry E4M3 codebook carrying exactly four indices
  per weight, then test scalar16 alone and a per-projection scalar16/MCG hybrid
  at 4.25 bpw. Separately refit the procedural MCG family compander over the
  fixed projection grid. No LDLQ or BlockLDLQ is permitted.
- **Outcome:** scalar16 alone failed. The hybrid improved GPTQ NVFP4 weight
  NMSE by 13.16%, but was 2.886% worse than procedural MCG, so it was rejected.
  The MCG compander refit selected alpha `2.0` independently for gate, up, and
  down; every projection retained the existing value. MCG alpha `2.0` remains
  frozen and the runtime uses native ties-to-even E4M3 rounding.

## 2026-09-04: Decision 17 — relax the noisy window gate only as a control

- **Evidence before decision:** replicated routed-output gains of 12.03% and
  25/32 window wins did not translate into a resolved end-to-end KLD effect;
  the adaptive KLD point estimate was only 1.04% and its paired interval
  crossed zero. The W6A8 carrier control also failed because native E4M3
  activation quantization dominated, not because of an identified missing
  scale factor.
- **Decision before subsequent device result:** the 32-window uncertainty rule
  no longer blocks developmental encoder or kernel work. A failed or noisy
  window screen is retained as a control and may inform later Shapley
  allocation, but it cannot qualify a candidate. The scientific claim remains
  unchanged: a codec beats NVFP4 only when paired end-to-end teacher KLD has a
  BCa upper confidence bound below zero on an eligible role. Shapley allocation
  cannot convert an inferior base arm into a win.

## 2026-09-04: Decision 18 — procedural-MCG P8 device arithmetic closure

- **Decision before repeat:** after an unsealed bring-up, freeze one fresh-seed
  K3/K4 repeat. Require zero decoder mismatches over 32,768 values per rate,
  cosine above `0.995`, relative L2 below `0.12`, finite outputs, identical
  runtime-source hashes, and no reroll. Plan SHA-256:
  `cafc8963560fda0e7202c1f8487991bdcf0ebea048870a3b6b65c3df7b153017`.
- **Outcome:** passed this bounded gate. K3 and K4 each had zero decoder
  mismatches. Native SM120 MoE arithmetic through `mxf8f6f4` measured cosine
  `0.9986532` / `0.9987590` and relative L2 `0.0552597` / `0.0564104`.
  The decoder is procedural MCG alpha 2.0 with zero runtime table bytes.
- **Boundary:** this closes bit-exact decode and small-M device arithmetic only.
  It is not split-prefill closure, end-to-end KLD, speed, determinism, or
  product qualification. P8 remains a quality product whose MMA issues at
  twice the NVFP4 count. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 19 — split-prefill P8 arithmetic closure

- **Decision before repeat:** after one unsealed M33/M64 bring-up, freeze a
  fresh seed and require all four K3/K4 by M33/M64 cells to have finite output,
  cosine above `0.995`, relative L2 below `0.12`, and exact runtime-source
  identity. The source transformer is fail-closed against the pinned B12X FC1
  and FC2 hashes. Plan SHA-256:
  `34134dbd58e540ba0b55fa5c9951c9cfedcfe7d6b37a6cb789aacbe2f44c5dc1`.
- **Outcome:** passed. Cosine ranged `0.9988336..0.9989196`; relative L2
  ranged `0.0464960..0.0483086`. Both split FC1 and split FC2 decode the K3/K4
  MCG stream in their MMA warps. The unused SQG shared-table reservation and
  copy were removed, so procedural MCG now has zero runtime table bytes.
  Result SHA-256:
  `16ccbac5ef027ef8814e1a7be2d151c465fccb3c315934327b89e70eaae15a11`.
- **Boundary:** this is prefill arithmetic closure, not full-model KLD or
  speed qualification. P8 still has twice the NVFP4 MMA issue count. No LDLQ
  or BlockLDLQ was used.

## 2026-09-04: Decision 20 — five-run split determinism

- **Decision before result:** run the identical K3/K4 by M33/M64 split probe
  in five cold containers and require every output tensor SHA-256 to match.
  Plan SHA-256:
  `61c46b3cd3ca7ede322a4b92eca56b0ea02cad9651dcf1e11d84a7a9fcf8b7bd`.
- **Outcome:** failed. All 20 arithmetic cells passed and their metrics were
  stable, but every output hash differed across runs. The probe exercised the
  ordinary unordered BF16 atomic-scatter specialization, not the runtime's
  separate deterministic route-output plus fixed top-k reduction path.
  This preserves the failure and leaves gate 4d open until that specialization
  is run five times. It does not reverse Decisions 18–19's decoder/MMA closure.
  No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 21 — corrected deterministic P8 runtime

- **Cause:** E=288 monolithic decode read one-byte sentinels through the
  logical UE8M0 scale ABI slots. The split path passed because it consumed a
  different repacked scale plane. Cluster-count arms 4/64/188 all reproduced
  the same failure, falsifying the resident-grid hypothesis.
- **Outcome:** dual logical/repacked scale planes close monolithic M3/M33 at
  cosine `0.999579..0.999693`. Per-route output plus fixed-order top-k sum is
  bitwise identical across five runs for both monolithic and materialized
  M3/M33. The deterministic full-model layer-3 arm closes to decoded
  pseudoquant within the relaxed margin (delta `+0.0013772`, BCa
  `[-0.0005542,+0.0039721]`) but is a quality null versus GPTQ (`3.06%` worse
  point estimate; interval crosses zero).
- **Boundary:** P8 remains a twice-NVFP4-issue quality path; speed and a strict
  quality win remain open. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 22 — relaxed joint-policy gate did not transfer to KLD

- **Decision before KLD:** demote the noisy 8-window routed-NMSE interval to a
  control and advance the frozen 24.05%-better three-state policy. Require at
  least 5% mean KLD improvement for development promotion; preserve the
  paired interval and all domains.
- **Outcome:** mean KLD improved only `0.8104%` versus matched decoded GPTQ
  NVFP4 (`0.0528992` versus `0.0533314`), with 16/32 wins and BCa
  `[-0.0024595,+0.0012695]`. This fails development promotion and the strict
  quality claim gate.
- **Consequence:** Shapley may still allocate transformations, but it must use
  KLD-aware marginal values; local routed-output NMSE cannot be assumed to
  cover the mismatch. No LDLQ or BlockLDLQ was used.

## 2026-09-04: Decision 23 — uniform P8 product gate before native6 allocation

- **Decision before result:** do not execute the 84-coalition native6-v2 game.
  First encode and install K4 procedural-MCG P8 on every routed layer 3–44,
  using the pinned domain-balanced REAP fit capture, GPTQ-style output-error
  feedback with static within-group activation order, jointly refitted
  UE8M0/32 scales, and no LDLQ or BlockLDLQ. Require every TP4 sidecar to bind
  to the corrected native6-v2 design hash and to carry exactly 4.25 payload
  bpw.
- Measure full-model teacher KLD on the already-opened, domain-balanced 32
  conditional-fit windows. This is developmental evidence; no selection,
  confirmation, final, or reserved confirmation logits may be opened.
- Then measure TP4 prefill and single-request decode tokens/s against the exact
  locally served EXL3 checkpoint and runtime, with matched request shapes,
  attention/KV settings, CUDA-graph mode, and five cold runs per arm. P8 must
  be strictly faster than EXL3 on both median prefill and median decode to
  advance. Any tie or regression stops the allocation game.
- If P8 passes, estimate the necessary game size from the existing pilot's
  observed main effects and pair interactions and preregister that power-based
  design. The superseded fixed 84-endpoint count is not an execution target.
- **ISA boundary:** P8 decodes directly to E4M3 plus physical UE8M0/32 scales
  in the fused prologue and uses `mxf8f6f4`; it has twice the MMA issue count
  of NVFP4. This gate does not claim the unfinished P4/`mxf4nvf4` product.

### Thermal-control amendment during the all-layer build

- The initial runner paused a worker at 89 C and resumed at 78 C. Live layer-3
  telemetry showed repeated stop/resume cycling, and the operator explicitly
  directed that 89 C is acceptable.
- The resumable build runner now defaults to an emergency pause at 94 C and
  resume at 88 C, while leaving NVIDIA's own thermal throttling active. These
  values are configurable and are not asserted as hardware specifications.
- This changes build scheduling only. It does not change source weights,
  calibration samples, encoder objective, sidecar format, KLD roles, or any
  performance-benchmark thermal-matching requirement. The exact amendment is
  preserved in
  `evidence/opened/codec-v2/native6-v2/all42-thermal-amendment-v1.json`.

## 2026-09-04: Decision 24 — P4 must close inside GLM serving

- **Hard product gate:** a standalone CUDA launch seam is not a P4 product.
  P4 must be selected by the actual B12X GLM MoE runtime for both gate/up FC1
  and down FC2, preserve routed top-k and TP4 behavior, and consume the physical
  codec sidecars without a dense BF16/FP16 weight materialization or matmul.
- The independent CPU codec commit uses alpha-1 procedural MCG projected to
  E2M1 with native round-to-nearest-even and signed-zero preservation. The
  first kernel commit instead retained the legacy ties-to-lower-magnitude,
  zero-canonicalizing law. An exhaustive census found `5,477 / 65,536` nibble
  mismatches: `45` numerical midpoint choices plus `5,432` signed-zero choices.
  These commits are preserved as structural building blocks but are not an
  interoperable ABI and must not be used for device or KLD closure as-is.
- Freeze one new versioned RNE ABI, add the missing one-matrix-to-TP4 sidecar
  bridge, and require exhaustive state, packed-byte, scale-address, FC1/FC2,
  dispatch, and fail-closed tests before a GPU run. Static SASS containing
  `mxf4nvf4 m16n8k64` proves instruction selection only; it does not prove
  device arithmetic, integrated serving, KLD, graph safety, or speed.
- Protected evaluation roles remain unopened, and no LDLQ or BlockLDLQ is
  introduced. Device closure waits for a reconciled serving commit and an
  available GPU boundary that does not corrupt the all-layer P8 build.

## 2026-09-04: Decision 25 — full-model P8 KLD is the critical path

- **Decision before result:** complete the uniform K4 procedural-MCG P8
  checkpoint for every routed layer 3–44, verify all 168 TP4 sidecars and
  receipts at exactly 4.25 payload bpw, and then execute one resumable KLD run
  on the exact 32-window domain-balanced conditional-fit role. Do not reroll,
  exclude, or replace windows. Plan SHA-256:
  `86b415ad76b80adc8dd983edcf17d2c0b371b32ccc921643ed8a0e112aeb98d2`.
- Report the absolute full-model mean KLD, all window values, four domain
  summaries, and paired window-bootstrap intervals. The already-opened
  decoded-GPTQ result is retained as a directional context row only: its rate
  and transformed layer set do not match uniform P8, so it cannot establish a
  strict product win.
- The KLD run uses an immutable runtime worktree at
  `bb45c5bc35b218f810c667698940b06d71ba1b21`, its content-addressed patch
  manifest, TP4 with expert parallelism disabled, DCP1, fixed-order routed
  summation, and the physical all-layer P8 sidecars. The analyzer is frozen by
  SHA-256 before the target result.
- P4 encoder, kernel, and GLM serving integration remain CPU/static side work
  until the P8 build releases the GPUs. A complete finite P8 KLD measurement
  advances to the preregistered five-cold-run TP4 speed comparison against the
  exact EXL3 system; Shapley allocation remains blocked on that speed gate.
- This conditional-fit result is developmental opened-role evidence, not
  untouched final qualification. No selection, confirmation, final, or
  reserved 28 confirmation logits may be opened. No LDLQ or BlockLDLQ is used.
- **Pre-result runtime amendment 1:** the prior layer-3 proof used the BF16
  overlay method and did not exercise layers 4–44's ModelOpt pre-forward
  quant-config hook. Exact source inspection showed that hook would rebuild a
  ModelOpt config from carrier scales already released by native P8. Runtime
  commit `efd250b002e8da5a0e603253e0cd07ba6249c7b4` bypasses that builder only
  for selected P8 methods; all experimental inputs and decisions are unchanged.
  The amendment is separately sealed and preserves the original plan.

## 2026-09-04: Decision 26 — reconcile P4 structurally without delaying P8

- The P4 encoder/TP4 bridge and GLM serving/kernel lines were composed only
  after resolving their independent decoder-law and container mismatch. The
  canonical ABI is procedural MCG alpha-1, RNE E2M1, signed zero, positive
  physical E4M3/16 scales, six TP-rank tensors, and exact per-layer capture
  identities. No LDLQ or BlockLDLQ is present.
- The opt-in backend now enters through the actual GLM `FusedMoEFactory` and
  `RoutedExperts.forward_modular` call path for both routed projections while
  leaving router, shared-expert, and TP-reduction behavior with the stock
  runner. It fails closed on unsupported topology, malformed sidecars,
  DBO/microbatching, duplicate model ownership, and overlapping host dispatch.
- After integration, 185 combined CPU/static tests passed with CUDA hidden.
  Offline assembly retains 15 `mxf4nvf4` E2M1 K64 sites, 62 registers, 1,728
  bytes shared memory, and zero spills. These are structural/compiler receipts,
  not device, KLD, graph, determinism, or speed evidence.
- P4 remains side work. No P4 GPU run may contend with the all-42-layer P8
  build or its sealed 32-window full-model KLD follow-on. Device closure begins
  only after that critical path releases the GPUs. Exact reconciliation:
  `docs/P4_V2_INTEGRATION_RECONCILIATION.md`.

## 2026-09-04: Decision 27 — freeze the post-KLD P8 versus EXL3 speed gate

- Before checkpoint or KLD completion, freeze a target-only serving comparison
  using the exact local P8 and EXL3 images/checkpoints: TP4, EP off, DCP1,
  B12X sparse attention, calibrated NVFP4 MLA KV, speculation off, prefix cache
  off, CUDA graphs on, one sequence, and matched 32K/64K requests.
- Run five new server containers and worker-process sets per arm, alternating
  order by round. Compiled caches remain warm and outside the cold-process
  definition; image, runtime, model, tool, request, and cache-path identities
  are recorded. Do not mutate power limits or clocks, start each arm at or
  below 75 C, and reject a measured cell reaching 94 C.
- **Decision before result:** P8 must be strictly faster than EXL3 on both the
  five-run median 32K Prometheus-validated prefill throughput and the five-run
  median C1 32K continuous-usage decode throughput. Any tie, regression,
  incomplete arm, graph/backend mismatch, or non-finite result stops the
  native6 allocation game. Report 64K cells as secondary evidence.
- The two codec products require different exact local images. This answers
  the end-product serving question and is not a causal kernel-only attribution.
  The sealed plan is
  `experiments/p8-uniform-all42-tp4-vs-exl3-speed-v1.json` (SHA-256
  `907882659d687043c52549ae39490c0b935982d2dc3c0f81681ee70d1ed313d8`).

## 2026-09-04: Decision 28 — correct P8 image before full-model execution

- GPU-free inspection found the queued base image lacked the explicit
  `trellis_codebook`, `trellis_scaled`, and `trellis_identity_boundary`
  constructor arguments used by P8. The mounted runtime does not replace the
  installed b12x package. Thus the queue was not executable as written.
- Before checkpoint completion or target results, amendment 2 pins both P8
  follow-ons to image `5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8`,
  the exact image recorded for native deterministic layer-3 conditional-fit v2.
  Require image/source hashes and explicit constructor ABI before GPU launch.
- Preserve the original plans and amendment 1. No encoder, sidecar, calibration,
  role, metric, analyzer, topology, decision rule, or protected boundary changes.
  Keep the current build uninterrupted; only replace waiting follow-on services.
- Receipt: `experiments/p8-all42-runtime-image-amendment-2.json` and the
  GPU-free `glm53_nvfp4.preflight_p8_runtime_image` check. This establishes
  source compatibility, not all-layer device closure or KLD quality. P8 uses
  twice NVFP4's MMA issue count; its speed remains unmeasured.

## 2026-09-05: Decision 29 — report the completed native all-layer measurement

- Outcome of pre-result Decisions 25/28, not a retroactive gate: all 42 routed
  layers and 168 TP4 sidecars completed; native all-layer KLD is
  `0.04002139481676188`, window BCa 95% CI `[0.03377817, 0.04825257]`.
- Audit passed 32 windows, 65,504 causal positions, and 168/168 native
  weight-ready and forward pairs. No protected roles or LDLQ were used.
- Historical contextual comparison: 24.957189% lower, 28/32 wins, paired delta
  `-0.01331002`, BCa 95% CI `[-0.02010227, -0.00757892]`. Different rate,
  quantized-layer coverage and EP/DCP topology prevent a matched codec claim.
- Preserve the predeclared speed gate. Its first EXL3 arm failed before
  measurement with checkpoint TP2/runtime TP4 mismatch. No speed win exists;
  allocation stays stopped. Preserve the failed logs before any compatibility
  amendment or rerun. P8 remains a twice-NVFP4-MMA-issue quality path.
- Receipts and limitations: `results/P8_FULLMODEL_CF32.md` and
  `evidence/opened/codec-v2/uniform-p8-all42-v1/`.

## 2026-09-05: Decision 30 — preregister the existing-topology EXL3 product trial

- Source inspection of the exact EXL3 image shows the checkpoint is unsliced;
  `_configure_glm53_unsliced_routed_experts` synthesizes `tp=2`. The working
  four-GPU service uses EP4/DCP4, whose explicit unsliced-EP path bypasses that
  TP guard. Do not misrepresent this as physically TP2-only checkpoint data.
- No speed measurement exists from v1. Preserve that failed trial; do not
  relax or remove the reader's guard to invent a new comparison runtime.
- Before any v2 speed result, freeze `p8-uniform-all42-tp4-vs-exl3-speed-v2.json`:
  P8 TP4/no-EP/DCP1 versus existing EXL3 TP4/EP4/DCP4. Speculation remains off
  in both, so this is not the production MTP5 throughput comparison.
- Same images, models, four GPUs, requests, graph requirements, repetitions,
  numerical speed rule and thermal limits. The exact original protocol is
  retained; v2 explicitly abandons the matched-topology estimand. No causal
  kernel/codec attribution from the resulting product-speed difference.
- New output `speed-v2` and container identity; v1 is never overwritten.
  Allocation remains stopped unless both five-run primary speed medians pass.
  No new EXL3 kernel or P8 weight change is introduced by this amendment.

## 2026-09-05: Decision 31 — preserve speed failures and repair receipt checks

- V2 failed at the CPU audit because its new worktree path differed from the
  existing receipt. A1 uses a fresh audit destination without weakening the
  audit. A1 then failed in DCP profile forward; A2 restores the pinned EXL3
  compose's B12X DCP flags, with no kernel source or weight change.
- A2 measured EXL3 but its wrapper required the wrong graph progress label.
  All four ranks actually completed FULL graph capture. A3 accepts the two
  runtime label spellings and additionally requires 100% FULL capture and
  completion on every TP rank; it rejects profiling or partial capture.
- A2's 32K baseline row (6,519 server prefill tokens/s, 91.198066 C1 decode
  tokens/s) is retained as an operational diagnostic, not silently substituted
  or erased. Missing post-run receipts require a fresh full five-round series.
- Before any P8 speed result, preserve the original thresholds, requests,
  repetitions, images, topology and serving environment. No speed pass or
  allocation authorization is inferred. See `results/P8_SPEED_ATTEMPTS.md`.

## 2026-09-05: Decision 32 — profile P8 decode after the frozen speed series

- The first completed v2a3 pair shows faster P8 prefill but substantially
  slower C1 decode. It is not the five-run result. Keep all five pairs,
  thresholds, topology and runtime unchanged; no concurrent GPU profiling.
- A read-only Astra source audit identifies sparse deterministic scheduling,
  scratch clearing and procedural decoder work as testable candidates, not
  established causes. Pin its source references and first-pair hashes in
  `results/P8_DECODE_CPU_AUDIT.md`.
- After the series, retain the existing allocation stop rule and use a
  separately identified synthetic/fit-only profile to choose any runtime
  optimization. Preserve decoder bytes, BF16 accumulation order and graph
  isolation. No LDLQ, no protected logits, no implied KLD transfer to a
  changed serving path. P8 remains a 2x-MMA-issue E4M3 quality product.

## 2026-09-05: Decision 33 — queue the isolated diagnostic behind all ten speed runs

- Freeze the synthetic profile plan and implementation before collection.
  Commit `ddc22a09858ebb4091f97fcc7700ee6296b0460b` is executed from a
  separate clean worktree. The queue receipt is
  `experiments/p8-decode-profile-v1-queue.json`.
- The follow-on is live only as a waiter while speed-v2a3 runs. It cannot
  acquire the model-stack lock or launch GPUs until terminal success and
  the complete receipt audit. A numerical speed-gate failure remains a
  failure and stops allocation; it does not prohibit this declared diagnosis.
- Use a new synthetic 32K request, not conditional-fit or protected logits.
  Same weights/runtime, declared profiler-only changes, CID-owned cleanup,
  fail-closed thermal telemetry and verified restoration. Seventy-nine CPU
  tests pass; no device or trace validity is inferred from that result.

## 2026-09-05: Decision 34 — product speed fails; preserve and diagnose decode

- Speed-v2a3 completed five cold processes per arm with ten unique containers,
  complete FULL graph-capture receipts on four ranks, stable within-arm
  configurations, start temperatures at most 68 C and measured peaks at most
  90 C. The complete receipt audit passed.
- Frozen primary medians: P8/EXL3 32K prefill `6940/6252` tokens/s
  (**+11.0045%**), and C1 32K decode `20.6937/91.2645` tokens/s
  (**-77.3256%**). Both metrics had to win, so the product gate fails and the
  native6 Shapley allocation game stops. Secondary 64K cells and every run are
  retained in `results/P8_SPEED_V2A3.md`.
- This is a different-topology system comparison: P8 TP4/no-EP/DCP1 versus
  EXL3 TP4/EP4/DCP4, target-only and no MTP. Do not attribute the full decode
  gap to the codec or transfer eager/FP8-KV KLD to graph/NVFP4-KV serving.
- Read-only Astra audits establish genuine fusion: procedural MCG decoding
  produces E4M3 register fragments inside the same kernel that issues
  `mxf8f6f4`; no decoded-weight global buffer exists. At C1, deterministic
  scheduling exposes only eight useful all-slice tasks, alongside graph-replayed
  scratch clears and routing barriers. These are source-backed hypotheses,
  not measured attribution. Continue the isolated Nsight diagnostic while
  preserving this speed failure and allocation stop.

## 2026-09-03: uniform rotation

- The original KLD 2.93661 result was invalidated. A sparse overlay was loaded
  by the ordinary safetensors loader and redirected tensors were overwritten
  by later carrier shards.
- The corrected runtime uses a fail-closed InstantTensor overlay and records
  first-forward proofs for both `R_in` (gate/up input) and `R_mid` (down input).
- Fixed H16 on layer 3 improved conditional-fit KLD by 2.625% and passed the
  paired interval gate. Learned rotation improved only 0.590% against its
  stock anchor and lost directly to H16.
- Exact-Qwen H16 on all 42 routed layers improved mean conditional-fit KLD by
  1.326%, but its BCa interval crossed zero. Uniform H16 therefore failed.

## 2026-09-03: selective rotation

- Adaptive conditional-fit localized heterogeneous effects. Layers 3-9 and
  32-44 worsened KLD, while the middle region contained the positive rows.
- The best adaptive subset, layers 3/19/20, improved conditional-fit by 5.493%
  and passed there. It was frozen before selection wave 2, where it was 0.339%
  worse than stock and failed.
- The pre-ranked runner-up, layers 19/20, improved conditional-fit by 2.956%.
  It was frozen before selection wave 3, where it improved by only 0.407%; its
  BCa 95% interval `[-0.00110568, +0.00066555]` crossed zero, so it failed.
- All three V3 selection waves are now opened. No more V3 layer-subset tuning
  is permitted. There is no qualified GLM rotation win from this campaign.

## Interpretation boundary

- The calibration capture is not evaluation leakage or a substituted corpus.
  The Hub corpus receipt points to the exact local REAP JSONL SHA-256
  `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`,
  and the Hub tokenizer receipt matches the local GLM tokenizer.
- Qwen's 29-36% headline combined GPTQ, REAP calibration, and H16 against a
  shipped RTN expert-only control. Its same-calibration H16-only increment was
  about 6-15%, depending on panel.
- On the corrected fixed-loader comparison, GLM identity GPTQ was a null versus
  stock (`t = 0.24`; point estimate `+1.09%` KLD, lower is better). It rules out
  the Qwen-sized GPTQ gain within the measured interval, but the earlier causal
  "GPTQ regression" account is retracted.
- Architecture is a plausible mechanism, not yet a proven cause: GLM has 288
  routed experts per layer and sigmoid/noaux top-8 routing with scaling 2.5,
  versus Qwen's smaller routed-expert system. Quantization errors can therefore
  change later routing decisions differently. A causal router-flip experiment
  is required to establish that mechanism.
