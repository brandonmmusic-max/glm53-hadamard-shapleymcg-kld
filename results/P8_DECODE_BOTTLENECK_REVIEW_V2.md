# P8 decode bottleneck review: implemented versus measured

Status: read-only evidence audit, 2026-09-05. Three agents reviewed fusion and
graphs, B12X reuse, and the validity of performance claims. No new GPU timing
was collected by this review. Current numerical-validation work is a separate
experiment, `p8-index-order-integration-v1`; this report does not alter it.

## Findings supported by existing receipts

| Question | Finding | Scope |
|---|---|---|
| Does trellis decoding feed native MMA without a BF16 weight buffer? | Yes. Narrow FC1 decodes gate/up into register fragments consumed by E4M3 MMA in the same kernel. | Source and prior runtime evidence; hidden decoder latency is not established. |
| Were CUDA graphs replaying? | The older N128 trace has 16 complete request-correlated graphs per rank, 2,475 nodes each, with actual M1 geometry. | Its original strict trace gate failed on trailing capture artifacts; the separately disclosed complete-prefix diagnostic was reproduced exactly. |
| Did the planned N64 implementation run? | All five later P8 serving runs record 168 actual M1 dispatch pairs, covering 42 layers and four ranks, plus FULL graph capture. | No fresh measured-request Nsight replay trace exists for that N64 build. |
| Is scratch initialization inside MMA? | No. Twenty-three zero allocations were consolidated into one arena clear with zero-copy typed views. | Fewer initialization launches; not a zero-cost or MMA-hidden clear. |
| What is the current residual bottleneck? | Not established after N64/consolidated-scratch optimization. | The older FC1 attribution cannot be treated as a new-build measurement. |

The fusion reviewer independently reran the complete-prefix trace analyzer
and obtained exactly the saved diagnostic JSON, and reran runtime-audit
checks on all five P8 final logs. The main agent recomputed the five P8 decode
median from the per-process summary files: **100.5501114546 tokens/s**.

Historical N128 attribution on a critical rank: FC1 4.968 ms/token, FC2
0.524 ms, native dense MMA 2.505 ms, and explicit fills 0.551 ms. FC1 was
approximately 90% of routed-MoE time. These are correlated graph samples
from one server, not independent runs. Kernel sums can overlap. No occupancy,
stall, or instruction-counter measurement identifies the limiting FC1 work.

The subsequent five-cold-process-per-arm system comparison measured P8
decode 100.5501 versus EXL3 91.2200 tokens/s (+10.23%), and server-validated
prefill 6,959 versus 6,239 tokens/s (+11.54%). P8 is TP4/noEP/DCP1 and EXL3
is TP4/EP4/DCP4, with different images and weight representations. The result
supports a tested serving-stack advantage, not an isolated codec effect.
Those results do not transfer automatically to the index-corrected build.

## B12X reuse: do not propose completed work as new

Already implemented and exercised: direct M1 routing, narrower N64 ownership,
cropped staging, both decoded halves, shared gate/up activation fragments,
and consolidated scratch initialization. The combined N64/arena package
reduced synthetic routed-block medians by 44.11–45.42% across four GPUs.
N32 did not improve on selected N64. FC2 N256 provided only approximately
1.4–1.8% block benefit, failed its device speed gate, and is not selected in
serving. B12X's original M1 kernel is not a drop-in for P8 arithmetic.

Two source-backed, unmeasured follow-ups remain:

1. Prove the M1 scale-scratch live address set and initialization requirements,
   then consider reducing capacity. `runtime_patch/p8_smallm_schedule.py:71`
   reserves 2,433,024 scale bytes, about 80% of the 3,038,496-byte arena.
   The existing wrapper clears that entire arena. A capacity reduction is
   not yet proven safe or fast.
2. If a fresh profile identifies activation preparation or specific dense
   projections, investigate B12X's four-lanes-per-K32 quantization and M1
   direct/non-TMA policies in its `b12x/_lib/dense_gemm.py`. These are donor
   mechanisms, not routed-MoE flags or measured P8 improvements.

Do not port repeated-scale reuse that assumes identical scales across K128:
P8 permits independent K32 scales. Split-K changes accumulation order and
is not an exact replacement. Preserve the clipped FP32 SwiGLU, BF16 rounding
seam, complete ordered K4096 reduction, and deterministic route reduction.

## Next measurement and evidence boundary

Finish the separate corrected-index full-model canary and conditional-fit32
numerical/KLD work. Profile the exact final serving image/configuration on
synthetic 32K C1 requests, without the heavy index observer, in a separate
predeclared diagnostic. Require complete request-correlated graph inventory,
actual N64 geometry, and separate preparation/FC1/FC2/fill/dense/NCCL timing.
Use representative-kernel instruction/stall profiling only after the current
critical cost is localized. Any speed claim for the modified build needs a
fresh serving comparison; same-topology N128/N64 A/B is required to isolate
the scheduling package's causal contribution.

P8 decodes procedural trellis weights into E4M3 and uses `mxf8f6f4` K32,
**twice the MMA issue count of NVFP4 for equal K**. It is not the P4/native
NVFP4 speed-class endpoint. This review changes no encoder, uses no LDLQ,
opens no protected logits, and makes no new KLD or allocation claim.
ExLlamaV3, KQuant, QSRT, w4a8_trellis and B12X attribution remains in
`THIRD_PARTY_NOTICES.md`.

## Evidence index

- `results/P8_DECODE_NSYS_V1.md`: original fused-MoE bottleneck and graphs.
- `results/P8_SMALLM_PROFILE_V1.md` and
  `evidence/opened/codec-v2/p8-smallm-profile-v1/analysis-diagnostic.json`:
  N128 attribution and explicitly preserved strict capture failure.
- `results/P8_FC1_TILES_DEVICE.md`: all device variants, including failures.
- `results/P8_FC1_COLD_COMPARISON_V1.md` and
  `evidence/opened/codec-v2/p8-fc1-cold-comparison-v1/`: five-run system
  comparison, raw per-run summaries, and runtime audits.
- `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py:454`:
  register-fragment decode; E4M3 MMA consumption at line 554.
- `runtime_patch/p8_native_kernel.py:366`: single-clear arena, not in-MMA zeroing.
- `results/P8_INDEX_ORDER_INTEGRATION_V1.md`: separately sealed current
  canary; full-model pass and conditional-fit32 KLD are not yet established.
