# Coupled H512/H128 + suh/svh: matched CF32 result

**Development gate: fail.** The complete coupled candidate did not reduce
equal-window mean teacher KLD versus the matched three-layer identity P8 arm.
Both arms completed all32 windows, final runtime audits and cross-arm input
reauthentication; executor exit0. No exclusions or protected-role openings.

| Arm | True-decode KLD | Selected-projection tensor bpw |
|---|---:|---:|
| Identity P8 | 0.052847380391592064 | 4.25 |
| Coupled H512/H128 + suh/svh | 0.05499088068580576 | 4.2539798595 |

Common conditions: B12X_MLA_SPARSE attention, nvfp4_ds_mla KV, TailV2,
native-p8-mxf8f6f4-n128 MoE (identity versus coupled boundary), E4M3
UE8M0_K32 activations, TP4/DCP1/noEP, CUDA graphs, MTPoff. Only layers3/20/22
use P8; other layers are the same stock NVFP4 carrier. These bpw values are
not whole-model disk rates. P8 has twice NVFP4's MMA issue count; no speed claim.

Coupled minus identity is **+0.0021435002942136946 KLD**, a **4.0560199547%
higher mean**. Paired95% BCa interval is **[-0.0010680525204973161,
+0.007625670904706085]**, with17/32 coupled window wins. Interval spans zero;
the result does not establish a statistically resolved regression. The
predeclared nonblocking-CI rule still fails because the candidate mean is higher.

Domain mean deltas (same conditions/rates as above): general+0.0003730923,
legal+0.0046921677, code/agentic+0.0042389203,
reasoning/termination-0.0007301792. No per-domain retuning is authorized by
these post-result diagnostics.

Metric: KL(teacher||student), FP64 CPU;2047 prediction rows/window with row0
one-token prefill excluded from true decode.32CF windows,8/domain; paired
window bootstrap B20000, seed20260902. CF32 is already-opened development
data, not final qualification. This compares the whole coupled candidate;
it does not isolate Hadamard, scales, transformed Hessians or each layer.

Stock-only serving was canceled at user direction and is not a tested arm.
Do not substitute historical FP8-KV/decoded-GPTQ scores for matched stock.

Evidence: `evidence/cf32-coupled-comparison-v3/analysis.json`, SHA256
`019fe64ed14a859b852cf9f6910ba888f49e4ed9e83c271ef52a6720d8a22a52`;
both arms' raw-retirement, score and runtime receipts are archived alongside
`evidence/cf32-coupled-candidate-v3/`. Per-token score NPZs remain local,
hash-bound by receipts. Transient raw logits were retired after durable scoring.

Production remains off, all owned containers cleaned up. No all42 coupled
build is justified by this pilot's pass rule. Handoff to Fable includes the
encoder-only calibrated inter-block BlockLDLQ fallback as the previously
authorized next option, not a demonstrated cure. No new GPU job launched.
