# Middle-boundary index audit

Date: 2026-09-05. Structural CPU/source evidence, not device closure.
Source inspected at integration commit `821998d`:
`runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py`.

## DECISIONS

1. Repair two demonstrated index defects before the next device retry. Keep the
   frozen synthetic fixture, exact byte gates and numerical thresholds. Retain
   all prior failed receipts; do not reinterpret the source audit as GPU success.

## Findings

The full-coupled FC1 branch passes `base_col = segment*64 + (raw_idx>>6)*32
+ (raw_idx&31)` to `_scale_fc1_after_h128`, but that helper requires a column
within the entire local intermediate dimension. The missing `output_tile*128`
means tiles 1, 2 and 3 reuse tile 0's gate/up svh. All 256 scale accesses per tile
are wrong on these three tiles; tile 0 is unaffected by this defect.

The subsequent activation gather chooses one of four lane-local registers using
the destination lane's target before a source-indexed warp shuffle. The source
lane therefore supplies the register selected by its own destination calculation,
not the destination's requested register. CPU emulation of source-lane semantics
finds 64/128 gathered elements incorrect. Examples: destination lane 8/component 0
needs activation 32 but reads 96; lane 16/component 0 needs 64 but reads 0.

The repair must shuffle a uniform register selection across lanes before making
the destination-dependent choice, or use an equivalently proven mapping. Test
all 128 positions, not just selected examples. Test scale addresses across all
four output tiles and both projections with nonuniform sentinel values.

Materialized prefill inherits this same `_run_task`, so both M1 and prefill are
affected. These source errors are distinct from the already repaired harness
shape bug and input Hadamard operation-order mismatch.

## Limits

No GPU was used in this audit. No KLD, throughput or pass result was produced.
Other defects may remain; the measured 4057/4096 middle payload discrepancies and
59/128 scale discrepancies cannot yet be attributed exclusively to these bugs.
Candidate is native P8 E4M3/UE8M0-K32, K4 payload 4.25 bpw plus coupled metadata;
attention/KV are not exercised in this component test. `mxf8f6f4` has twice the
NVFP4 MMA issue count.
