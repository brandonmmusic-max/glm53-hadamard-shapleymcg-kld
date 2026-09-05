# P8 small-M integrated TP4 diagnostic

Integration status: **pass.** Product gate versus EXL3: **still fail.**

| 32K C1 metric | Old P8 | Small-M P8 | Fixed EXL3 reference |
|---|---:|---:|---:|
| Sustained decode | 20.6937 tok/s | **75.5962 tok/s** | 91.2645 tok/s |
| Relative to old P8 | — | **+265.31%; 3.653x** | — |
| Relative to EXL3 | — | **-17.168%** | — |

Small-M closed **77.798% of the old P8-to-EXL3 decode deficit** in one
integrated run. The 32K standalone prefill result was 7,166 client tok/s
(7,235 server-validation tok/s), versus the earlier P8 6,940 and EXL3 6,252
references. That is +3.26% and +14.62%, respectively, but this run was not a
five-cold-run product comparison.

## Serving closure

- Exact topology: TP4, no expert parallelism, DCP1, target-only, no MTP.
- FULL CUDA graph audit passed on ranks 0-3.
- All 42 routed layers x 4 ranks emitted first-forward receipts with
  `small_m_scheduler=true` (168/168).
- Same K4 procedural-MCG stream, E4M3 reconstruction, physical UE8M0/32,
  identity boundary, 4.25 routed bpw and `mxf8f6f4` path as the prior P8 run.
- Maximum measured GPU temperature was 71 C; the 90 C abort boundary was not
  approached.
- The production backend was active before the diagnostic and was restored
  active afterward. The model-stack timer remained inactive as found.
- No protected role, calibration window or teacher logits were opened. No
  LDLQ or BlockLDLQ was used.

The generated output is the same codec/checkpoint path; this diagnostic changes
the M1 work decomposition, not the stored weights or KLD objective. The
single-layer graph gate was bit-exact to the deployed monolithic arithmetic on
the frozen payload, but full-model KLD was not rerun here and is not claimed as
a new measurement.

## Decision boundary

The user-frozen rule says allocation stays stopped unless P8 beats existing
EXL3 serving. At 75.5962 versus 91.2645 tok/s, it does not. Therefore the old
product failure cannot be rewritten and the Shapley allocation game remains
stopped. The result is nevertheless strong engineering evidence: the measured
decode bottleneck was overwhelmingly the old eight-task M1 schedule, and the
new scheduler recovers most of that gap under actual GLM serving and CUDA
graphs.

Next work should profile this integrated candidate, focusing on the critical
Max-Q ranks and the two serial N128 FC2 halves that restage A, then attempt an
N256 A-reuse port from B12X while preserving the monolithic BF16 boundary.

The frozen plan, raw benchmark JSON, FULL graph audit, 168 receipt log,
container identity, thermal telemetry and restoration receipt are under
`evidence/opened/codec-v2/p8-smallm-integrated-v1/`.
