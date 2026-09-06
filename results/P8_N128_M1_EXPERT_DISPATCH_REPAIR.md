# N128 M1 FC1 expert dispatch repair

2026-09-05. Independent critic/source finding; GPU verification pending.

Decision 1, before the next diagnostic run: pass `topk_ids` to every direct
small-M FC1 owner, including N128. Previously only N32/N64 selected these IDs.
The M1 phase0 path does not populate grouped `task_expert`; the wrapper zeros
that buffer, and N128 FC1 reads it by route index. Consequently that launch
selects expert 0 for all routes instead of the requested eight experts.
FC2 already selects `topk_ids` for small-M, creating a mismatched FC1/FC2 pair.

The corrected condition adds `p8_small_m` while retaining existing narrow-owner
behavior. Dense N128 prefill remains on grouped-task metadata. The regression
test evaluates the actual source predicate for N32/N64/N128 M1 and dense N128,
using nonuniform requested IDs against zeroed grouped metadata.

This is a real source defect and a credible cause of broad middle divergence,
not proof it explains every differing byte. It cannot explain input-carrier
rounding, which precedes expert routing. The next diagnostic must verify the
consumed expert mapping rather than merely repeat the requested expert list.
No new KLD, speed, or closure pass is asserted; prior failed receipts remain.

Regime: synthetic coupled P8 M1, no attention/KV, E4M3/UE8M0-K32 activations and
weights, K4 payload 4.25 bpw plus metadata; mxf8f6f4 is twice NVFP4 MMA issue.

## Contrary observation retained

Re-reading the frozen v4 NPZ shows that observed middle rows are not identical:
payload mismatch counts versus row 0 are `[0,506,509,508,506,505,503,504]`, and
scale counts `[0,8,7,8,8,4,8,3]`. A fully deterministic expert-0 computation
repeated eight times on the same input would instead agree. Therefore the source
argument above is not a complete explanation of the device behavior. Additional
dispatch state, unintended writes or another kernel defect remain possible.
The new diagnostic must capture consumed expert IDs and write completeness;
the old NPZ did not capture `task_expert`. Do not claim that eight observed
expert-0 computations have been demonstrated by the existing device receipts.
