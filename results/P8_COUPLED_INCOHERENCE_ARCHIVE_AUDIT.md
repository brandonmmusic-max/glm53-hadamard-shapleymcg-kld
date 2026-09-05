# P8 coupled-incoherence archive audit

Status: no-GPU archive audit complete. No protected role was opened.

## Verdict

The GLM P8 campaign has not tested Luke/QSRT's coupled H512/H128 topology.
The existing `p8-h128` results used uncoupled per-128 transforms only along
the intermediate dimension. Identity was selected because those candidates
did not transfer to end-to-end KLD, not because full-dimensional coupled
incoherence was disproved.

The August 29 LDLQ ablation removed only the intra-tile/intra-walk factor at
`trellis.py:357`. It is not a control for calibrated inter-block BlockLDLQ.
Inter-block BlockLDLQ remains an encoder-only, P8-ABI-preserving fallback on
layers 3, 20, and 22, and is run only if the coupled-incoherence pilot fails to
close the tail-fixed P8-versus-EXL3 gap.

## Existing uncoupled evidence

All rows are layer 3 unless stated otherwise.

| Arm | Evidence role | Candidate | Control | Relative result | Decision |
|---|---|---:|---:|---:|---|
| Fixed H128 activation-only | balanced fit diagnostic, 16 experts | 0.0008403094 | 0.0007372527 | 13.98% worse | fail |
| Tuned scaled H128 | fit selection/validation, 16 experts | 0.0006255257 | 0.0007372527 | 15.15% better | inconclusive CI |
| Signed H128 | same fit split | 0.0008430983 | 0.0007372527 | 14.36% worse | fail |
| Frozen balance-.5 | disjoint fit evaluation, 16 experts | 0.0004546180 | 0.0006830104 | 33.44% better | proxy pass |
| Frozen whiten-.5 | disjoint fit evaluation, 16 experts | 0.0004686664 | 0.0006830104 | 31.38% better | proxy pass |
| H128 + MCG K4 | fit evaluation, all 288 experts | 0.00360018 | 0.00500244 | 28.03% better | routed-output proxy pass |
| Whole-layer H128 + MCG KLD | conditional-fit32 | 0.05349517 | 0.05333142 | 0.31% worse | null, CI crosses zero |
| Binary hybrid-policy KLD | reused conditional-fit32 | 0.05277627 | 0.05333142 | 1.04% better | adaptive null |
| Three-state-policy KLD | reused conditional-fit32 | not separately retained here | identity anchor | 0.81% better | adaptive null |

The recent three-layer `p00625` result, 0.039755726 versus 0.040145899
(0.97% better), is also not coupled H128. It is a four-stage H16-wired
butterfly evaluated as a BF16 weight overlay, not native 4.25-bpw P8 serving.

## Coupled topology that remains to be tested

The archived Luke/QSRT order is:

1. shared normalized H512 over the full residual dimension;
2. full-hidden F16 `suh`, then H128 at FC1 input;
3. six intermediate segments
   `[slot0_svh, slot1_svh, down_suh, pre_signs(2I), post_signs(I)]`;
4. interleaved gate/up H128, scale, H128, signs, adjacent-pair activation;
5. post-activation signs, H128, down scale, H128;
6. down-output H128, full-hidden F16 `svh`, then shared H512.

The deterministic sign generator and reference order are in
`/home/brandonmusic/KLC_SANDBOXES/b12x-coupled-glm52-20260813/tests/moe/test_fused_moe_trellis.py`
at lines 1241 and 919 respectively. Archived commits `a4b27475559a`,
`f17e19dad2f3`, and `080510bdac2e` establish direct E4M3 decode, coupled
kernel closure, and CUDA-graph replay in their covered tests. They do not
provide coupled-versus-identity KLD evidence.

The general dynamic kernel has a `trellis_coupled` implementation, but the
active optimized P8 specialization hardcodes `trellis_coupled=False` and
`trellis_identity_boundary=True`; its fast-path contract rejects coupled
mode. Luke's outer H512 hooks are not integrated into that specialization.
Flipping the flag alone would therefore be mathematically invalid.

## Frozen next arm

After the tail-V2 CF32 P8-versus-production-EXL3 ship gate sizes the actual
gap, the fixed three-layer arm must:

- use layers 3, 20, and 22 at exact P8 K4/MCG 4.25 bpw;
- transplant and hash the GLM-5.3 EXL3 checkpoint's full-dimensional `suh`,
  `svh`, and deterministic sign metadata;
- preserve the archived H512/H128 order exactly;
- re-encode all three projections in that basis using the current inter-group
  Hessian-feedback encoder;
- wire both outer H512 and inner `trellis_coupled` runtime boundaries;
- compare with a freshly re-encoded identity P8 arm on the same CF32 windows;
- report paired BCa and require mean KLD improvement over unrotated P8;
- then require native-kernel/pseudoquant closure before any all-layer build.

No new angles or per-layer tuning are permitted. If the EXL3 metadata cannot
be mapped exactly, a new scale/sign fit must be preregistered; Luke draw 6 is
only a structural canary and has no demonstrated GLM quality superiority.
