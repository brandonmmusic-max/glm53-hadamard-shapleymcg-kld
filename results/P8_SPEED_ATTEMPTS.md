# P8 product-speed attempt ledger

The full-model P8 KLD result is complete; speed qualification is separate.
The frozen v2 series requires five independent cold server starts per arm,
32K prefill and C1 decode primary medians both strictly faster for P8, and
completed CUDA graphs. No allocation pass exists yet.

| Attempt | Observed outcome | Disposition |
|---|---|---|
| speed-v1 | EXL3 reader synthesized TP2 restriction and rejected TP4/no-EP | Failed before measurement; original matched-topology estimand not achieved |
| speed-v2 | CPU audit refused an overwrite because the immutable worktree path changed | Failed before GPU launch; use a per-attempt audit destination |
| speed-v2a1 | EXL3 loaded, but omitted production DCP flags selected an incompatible combine path | Failed before measurement; restore the two flags from the pinned serving compose |
| speed-v2a2 | EXL3 measurement completed; wrapper rejected valid full graph capture due to a log-label mismatch | Preserve as receipt-incomplete operational diagnostic; restart full series under a3 |

The stored EXL3 weights are unsliced, not physically TP2-only. The existing
four-GPU serving path is TP4/EP4/DCP4. V2 compares that target-only path with
P8 TP4/no-EP/DCP1: different-topology product speed, not a matched kernel
attribution or comparison against production MTP5 throughput. P8 remains an
E4M3 `mxf8f6f4` quality path with twice NVFP4's MMA issue count, not native P4.

The saved v2a2 baseline row measured 32K server-validated prefill at 6,519
tokens/s and C1 decode at 91.198066 tokens/s. It has no paired P8 result and
is not counted in the restarted series. It was excluded for missing required
post-run receipts following the wrapper failure, not for its measured speed.
The original throughput analyzer accepts its JSON without modification.

Operational amendments a1–a3 are sealed in `experiments/`; the full v2 plan
retains the original numerical thresholds, request protocol and repetitions.
The new graph verifier accepts either runtime capture-label wording, while
requiring FULL capture at 100%, graph-enabled configuration, and completion
messages from all four TP ranks. Profiling, partial capture, or eager mode fail.

Exact terminal artifacts and per-file hashes are retained under
[p8-speed-attempts](../evidence/opened/codec-v2/p8-speed-attempts/).
`scripts/snapshot_p8_speed_attempts.py` refuses active services and refuses
to overwrite changed evidence. Each source attempt and service remains local.
Attempt v2a3 ran from commit
`33bc3de0e94dfaac49fc44c442ef4fea54f97936` and completed all ten cold runs.
Median 32K prefill improved 11.00%, while C1 decode regressed 77.33%; the
predeclared two-metric gate failed and allocation stopped. The complete result
is [P8_SPEED_V2A3](P8_SPEED_V2A3.md), without rewriting the failed attempts.

The separate read-only `glm53_nvfp4.audit_p8_speed_receipts` verifies
receipt hashes, unique cold-container identities, within-arm configuration
stability, ordered non-overlapping execution, graph/backend/dispatch evidence
and thermal observations. It does not change the frozen speed estimator.
`--allow-incomplete` reports a pending inventory as **incomplete**, never a
pass. Docker environment and bind-list ordering is canonicalized only after
rejecting duplicate keys/destinations; command ordering is preserved.
The first three completed v2a3 slots passed these individual checks, while
the overall series remained incomplete. Twenty-two focused CPU tests passed
across this auditor, the frozen speed analyzer and graph verifier.

After the service is terminal, run the auditor without `--allow-incomplete`
and preserve its output as `speed-receipt-audit.json`. The snapshot tool now
accepts v2a3 and its final analysis/audit, but still refuses active services.
