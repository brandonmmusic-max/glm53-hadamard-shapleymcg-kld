# Corrected-index P8 full32 analysis amendment

Status at seal: the target capture completed and passed full bitwise closure,
but the chained V1 CPU analyzer failed before creating its output or scoring a
teacher window. Analysis V2 is a predeclared, analysis-only correction. It
does not authorize another GPU capture or change the metric, windows,
aggregation, checkpoint, or captured logits.

## Completed capture

- Capture plan SHA-256: `6873cd1c066ad2845635cdb0db25df2352196240930d99ec213dc16b6a15133a`.
- Root execution SHA-256: `3a08163738655452c0c430a36ecb906e014b0bec7f5d8aeef80d70849ccac158`.
- Full-exact receipt SHA-256: `89b628ec6519f3a2e070a2226e62140bbd9608e2f5ea57966a1c7e4ad1f53e10`.
- Result: N128 and N64 were bitwise identical on all 32 conditional-fit
  windows, 2,047 rows per window and 154,880 logits per row: 65,504 causal
  rows total. Both serving processes completed, cleaned up, and restored the
  prior service state. Peak sampled temperature was 88 C, below the sealed
  90 C abort threshold.

## Preserved V1 analysis failure

The chained unit `glm53-p8-index-order-full-v1.service`, invocation
`c6509947a453419d900ffb39cee70b7a`, exited 1 after capture. Its
invocation-scoped journal has SHA-256
`423aa90e503e91455fcc35196414fc678ea4e38f531abfde47e43b91bec01d9d`
and terminates at `verify_execution` with
`ValueError: request does not match approved causal sequence`.

The analyzer reconstructed request model names after the launcher's scoped
full-campaign prefix had been restored. The expected captured request was
`glm53-p8-index-order-full-v1-n128`; the verifier instead derived the older
base prefix. The fixed V1 output path did not exist after the failure, proving
that it stopped before output creation and before teacher scoring. This is an
analysis identity bug, not a failed or partial GPU capture.

## Frozen V2 correction

The V2 wrapper scopes the full-campaign prefix only around the unchanged V1
receipt verifier and scorer, restores it in `finally`, and writes to a fresh
output. It authenticates the original sealed plan, all 56 frozen V1 sources,
the root/exact receipts, the exact failed systemd invocation and journal, the
absence of the V1 analysis output, and all capture artifacts before scoring.
It can neither call the capture runner nor control a service.

- Amendment source commit: `87815e447114596edda63fe8128ad622cadf6c0e`.
- V2 analysis plan: `experiments/p8-index-order-full-analysis-v2.json`.
- V2 plan SHA-256: `79e958c6463c647b24fc8d48640ff0c85f0e65393e2a7839bcf45103834abc9c`.
- Validation before seal: 119 focused CPU tests passed; `py_compile` and
  `git diff --check` passed. Two GPT-5.6 SOL high-reasoning reviews returned
  GO after the failure-evidence and partial-progress receipts were tightened.

The analysis remains conditional-fit development evidence. It cannot claim a
matched stock-NVFP4 quality improvement, protected-role qualification, or
closure of the known incomplete-KPool-tail omission. P8 still uses E4M3
`mxf8f6f4`, with twice the NVFP4 MMA issue count for equal K; it is not P4.

## Result

The single sealed CPU-only analysis completed successfully. No GPU was
launched and no protected role was opened.

- Mean KLD, N128: `0.11747562624547078` nats.
- Mean KLD, N64/fused scratch: `0.11747562624547078` nats.
- N64 minus N128: exactly `0.0`; all paired window values and NPZ outputs are
  identical, so no sampling interval is inferred for the identically-zero
  delta.
- Equal-window 95% BCa interval for absolute KLD:
  `[0.09262619134086884, 0.1508711690207086]`.
- True-decode rows 1--2046 equal-window mean: `0.1156031843903463` nats.
- Full result: 32 unique windows, eight per domain and 65,504 causal rows.
- Outer execution SHA-256:
  `a67f466084947759156737be705ccdfedc965d33c3d4bee9f172ab8e359faa23`.
- Nested analysis SHA-256:
  `83fa9f3700c8ad2ad7e2be77c42f4c682afb64d95a8bedf486ec02b3fbbfcb2c`.

An independent SOL-high replay verified all 98 outer and 97 inner inventory
entries, every per-window raw and NPZ pair, the aggregate, the terminal-success
unit state, and the claim boundaries.

The first sanitized-snapshot invocation failed closed before creating its
destination because the newly integrated packager derived the frozen V1 plan
path from the V2 checkout. The correction binds the original capture worktree
explicitly and is regression-tested; it does not touch capture or KLD output.

This is worse in absolute KLD than the earlier `0.04002139481676188` run by a
factor of `2.9353` (`+193.53%`). That difference is real between the two
observed systems, but it is not attributable to N64: N128 is identical here.
The runs share the conditional-fit32 window IDs, all 32 teacher hashes, metric
code, TP4/noEP/DCP1 topology, carrier, all 168 P8 sidecars and no-MTP serving.
The older run measured all 2,048 causal positions through one prompt-prefill
execution, eager with FP8 MLA KV. This run measured one-token prefill followed
by 2,047 serial forced decode steps, using FULL graphs, Model Runner V2,
NVFP4 MLA KV, a newer capture image and the corrected-index/small-M runtime.
Thirty of 32 windows are higher in the current run. Those variables were
changed together, so this comparison does not identify which one caused the
higher KLD. In these reports, `M1` means the small-M MoE matmul shape; it does
not mean an MTP-1 speculative model.
